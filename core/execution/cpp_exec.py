from __future__ import annotations

import logging
from typing import Any, Dict

import os

from core.data.kraken_rest import KrakenRestClient
from core.execution.clamp import clamp_order_size
from core.execution.kraken_rules import load_rules
from core.config.runtime import get_proposed_weight, get_runtime_setting, is_meme_lane
from core.execution.kraken_live import KrakenLiveExecutor
from core.execution.mock_exec import MockExecutor
from core.execution.order_policy import build_order_plan
from core.execution.risk_budget import estimate_entry_risk_fraction, calculate_dynamic_risk_pct
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
        self.client = KrakenRestClient()
        self._rules = load_rules(self.client)
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

        min_notional_usd = float(
            get_runtime_setting("MEME_EXEC_MIN_NOTIONAL_USD")
            if is_meme_lane(lane)
            else get_runtime_setting("EXEC_MIN_NOTIONAL_USD")
        )

        if lane == "L1":
            risk_per_trade_pct = float(get_runtime_setting("L1_EXEC_RISK_PER_TRADE_PCT"))
        elif lane == "L2":
            risk_per_trade_pct = float(get_runtime_setting("L2_EXEC_RISK_PER_TRADE_PCT"))
        elif lane == "L4" or is_meme_lane(lane):
            risk_per_trade_pct = float(get_runtime_setting("MEME_EXEC_RISK_PER_TRADE_PCT"))
        else:
            risk_per_trade_pct = float(get_runtime_setting("EXEC_RISK_PER_TRADE_PCT"))

        atr_pct = float(features.get("atr", 0.0)) / max(float(features.get("price", 1.0) or 1.0), 1e-10) * 100.0
        risk_per_trade_pct = calculate_dynamic_risk_pct(
            base_risk_pct=risk_per_trade_pct,
            cash=cash,
            atr_pct=atr_pct,
            trade_quality_scale=tq.size_scale,
        )

        kraken_symbol = self.client.resolve_order_pair(symbol).replace("/", "").upper()
        rule = self._rules.get(kraken_symbol)
        if rule is None:
             return {"status": "rejected", "reason": f"missing_pair_rule:{kraken_symbol}"}

        risk_fraction_of_notional = estimate_entry_risk_fraction(
            price=price,
            atr=float(features.get("atr", 0.0) or 0.0),
            lane=lane,
        )

        clamp = clamp_order_size(
            pair=kraken_symbol,
            price=price,
            desired_usd=notional,
            rule=rule,
            equity_usd=cash,
            max_position_usd=max(cash, notional),
            risk_per_trade_pct=risk_per_trade_pct,
            min_notional_usd=max(float(features.get("min_notional_usd", 0.0) or 0.0), min_notional_usd),
            max_min_trade_risk_mult=float(get_runtime_setting("EXEC_MIN_TRADE_RISK_BUDGET_MULT")),
            clamp_up=True,
            kelly_fraction=float(features.get("kelly_fraction", 1.0) or 1.0),
            risk_fraction_of_notional=risk_fraction_of_notional,
        )

        if not clamp.ok:
            return {
                "status": "blocked",
                "reason": [clamp.reason],
                "price": clamp.price,
                "notional": clamp.notional_usd,
                "cost": cost.to_dict(),
            }

        qty = clamp.size_base

        if self.engine is not None:
            req = krakencpp.ExecRequest()
            req.symbol = symbol
            req.side = "BUY" if signal == "LONG" else "SELL"
            req.qty = float(qty)
            req.price = float(price)

            res = self.engine.simulate(req)
            sim_status = res.status
            sim_qty = res.qty
            sim_notional = res.notional
            sim_fee = res.fee
        else:
            if not self._warned_cpp_fallback:
                _LOGGER.warning("krakencpp unavailable; using Python fallback for simulate")
                self._warned_cpp_fallback = True
            sim_status = "filled"
            sim_qty = float(qty)
            sim_notional = float(qty * price)
            sim_fee = float(sim_notional * self.fee_rate)

        return {
            "status": sim_status,
            "symbol": symbol,
            "side": "BUY" if signal == "LONG" else "SELL",
            "qty": sim_qty,
            "price": float(price),
            "notional": sim_notional,
            "fee": sim_fee,
            "mark_price": float(price),
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
