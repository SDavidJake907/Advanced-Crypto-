from __future__ import annotations

from typing import Any, Dict

import os

from core.execution.kraken_live import KrakenLiveExecutor
from core.execution.mock_exec import MockExecutor
from core.execution.order_policy import build_order_plan
from core.risk.fee_filter import evaluate_trade_cost

try:
    import krakencpp
except ImportError:
    krakencpp = None


class CppExecutor:
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate
        self.mode = os.getenv("EXECUTION_MODE", os.getenv("KRAKEN_ENV", "paper")).strip().lower()
        self._fallback = MockExecutor(fee_rate=fee_rate)
        self._live = KrakenLiveExecutor(fee_rate=fee_rate)
        self.engine = krakencpp.ExecutionEngine(fee_rate) if krakencpp is not None else None

    def execute(
        self,
        signal: str,
        symbol: str,
        features: dict,
        state: dict,
        size_factor: float = 1.0,
    ) -> Dict[str, Any]:
        if self.mode == "live":
            return self._live.execute(signal, symbol, features, state, size_factor=size_factor)
        if self.engine is None:
            return self._fallback.execute(signal, symbol, features, state, size_factor=size_factor)
        if signal == "FLAT":
            return {"status": "no_trade"}

        cost = evaluate_trade_cost(features, signal)
        if not cost.actionable:
            return {"status": "blocked", "reason": cost.reasons, "cost": cost.to_dict()}

        order_plan = build_order_plan(features, signal, cost)
        price = float(order_plan.price or 0.0)
        cash = float(state.get("cash", 0.0) or 0.0)
        base_weight = float(features.get("proposed_weight", 0.1) or 0.1)
        notional = cash * base_weight * max(float(size_factor), 0.0)
        if notional <= 0 or price <= 0:
            return {"status": "rejected", "reason": "invalid_notional_or_price"}

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
