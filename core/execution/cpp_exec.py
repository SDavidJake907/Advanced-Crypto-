from __future__ import annotations

import logging
from typing import Any, Dict

import os

from core.config.runtime import get_proposed_weight, get_runtime_setting
from core.execution.kraken_live import KrakenLiveExecutor
from core.execution.mock_exec import MockExecutor
from core.execution.order_policy import build_order_plan
from core.execution.risk_budget import estimate_entry_risk_fraction
from core.risk.fee_filter import evaluate_trade_cost
from core.risk.trade_quality import assess_trade_quality

try:
    import krakencpp
except ImportError:
    krakencpp = None

_LOGGER = logging.getLogger(__name__)


class CppExecutor:
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate
        self.mode = os.getenv("EXECUTION_MODE", os.getenv("KRAKEN_ENV", "paper")).strip().lower()
        self._fallback = MockExecutor(fee_rate=fee_rate)
        self._live = KrakenLiveExecutor(fee_rate=fee_rate) if self.mode == "live" else None
        self.engine = krakencpp.ExecutionEngine(fee_rate) if krakencpp is not None else None
        self._warned_cpp_fallback = False
        if self.mode != "live" and self.engine is None:
            _LOGGER.warning("krakencpp unavailable; using MockExecutor fallback for mode=%s", self.mode)

    def execute(
        self,
        signal: str,
        symbol: str,
        features: dict,
        state: dict,
        size_factor: float = 1.0,
    ) -> Dict[str, Any]:
        if self.mode == "live":
            assert self._live is not None
            return self._live.execute(signal, symbol, features, state, size_factor=size_factor)
        if self.engine is None:
            if not self._warned_cpp_fallback:
                _LOGGER.warning("CppExecutor executing with MockExecutor fallback because krakencpp is unavailable")
                self._warned_cpp_fallback = True
            return self._fallback.execute(signal, symbol, features, state, size_factor=size_factor)
        if signal == "FLAT":
            return {"status": "no_trade"}

        cost = evaluate_trade_cost(features, signal)
        if not cost.actionable:
            return {"status": "blocked", "reason": cost.reasons, "cost": cost.to_dict()}
        if float(cost.net_edge_pct) <= 0.0:
            return {"status": "blocked", "reason": ["net_edge_non_positive"], "cost": cost.to_dict()}

        tq = assess_trade_quality(features, str(features.get("lane", "L3")))
        _exec_features = dict(features)
        _exec_features["trade_quality_limit_offset_scale"] = tq.limit_offset_scale
        order_plan = build_order_plan(_exec_features, signal, cost)
        price = float(order_plan.price or 0.0)
        cash = float(state.get("cash", 0.0) or 0.0)
        lane = str(features.get("lane") or "")
        fallback_weight = get_proposed_weight(symbol=symbol, lane=lane)
        base_weight = float(features.get("proposed_weight", fallback_weight) or fallback_weight)
        notional = cash * base_weight * max(float(size_factor), 0.0) * tq.size_scale
        if notional <= 0 or price <= 0:
            return {"status": "rejected", "reason": "invalid_notional_or_price"}

        min_notional_usd = max(float(features.get("min_notional_usd", 0.0) or 0.0), float(get_runtime_setting("EXEC_MIN_NOTIONAL_USD")))
        risk_budget_usd = cash * (float(get_runtime_setting("EXEC_RISK_PER_TRADE_PCT")) / 100.0)
        risk_fraction_of_notional = estimate_entry_risk_fraction(
            price=price,
            atr=float(features.get("atr", 0.0) or 0.0),
            lane=lane,
        )
        if notional < min_notional_usd and (min_notional_usd * risk_fraction_of_notional) > (risk_budget_usd * float(get_runtime_setting("EXEC_MIN_TRADE_RISK_BUDGET_MULT"))):
            return {"status": "blocked", "reason": ["min_trade_exceeds_risk_budget"], "cost": cost.to_dict()}
        notional = max(notional, min_notional_usd)
        qty = notional / price

        req = krakencpp.ExecRequest()
        req.symbol = symbol
        req.side = "BUY" if signal == "LONG" else "SELL"
        req.qty = float(qty)
        req.price = float(price)

        res = self.engine.simulate(req)
        return {
            "status": res.status,
            "symbol": res.symbol,
            "side": res.side,
            "qty": res.qty,
            "price": res.price,
            "notional": res.notional,
            "fee": res.fee,
            "mark_price": res.price,
            "order_type": order_plan.order_type,
            "limit_price": order_plan.limit_price,
            "maker_preference": order_plan.maker_preference,
            "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
            "cost": cost.to_dict(),
        }

    def execute_exit(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        features: dict[str, Any],
        exit_reason: str,
    ) -> Dict[str, Any]:
        if self.mode == "live":
            assert self._live is not None
            return self._live.execute_exit(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                features=features,
                exit_reason=exit_reason,
            )
        notional = qty * price
        fee = notional * self.fee_rate
        return {
            "status": "filled",
            "symbol": symbol,
            "side": "SELL" if side == "LONG" else "BUY",
            "qty": float(qty),
            "price": float(price),
            "notional": float(notional),
            "fee": float(fee),
            "mark_price": float(price),
            "exit_reason": exit_reason,
        }
