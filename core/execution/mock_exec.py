from __future__ import annotations

from typing import Any, Dict

from core.execution.base import Executor
from core.execution.order_policy import build_order_plan
from core.risk.fee_filter import evaluate_trade_cost


class MockExecutor(Executor):
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate

    def execute(
        self,
        signal: str,
        symbol: str,
        features: dict,
        state: dict,
        size_factor: float = 1.0,
    ) -> Dict[str, Any]:
        if signal == "FLAT":
            return {"status": "no_trade"}

        cost = evaluate_trade_cost(features, signal)
        if not cost.actionable:
            return {"status": "blocked", "reason": cost.reasons, "cost": cost.to_dict()}

        order_plan = build_order_plan(features, signal, cost)
        price = float(order_plan.price or 0.0)
        cash = float(state.get("cash", 0.0) or 0.0)
        base_weight = float(features.get("proposed_weight", 0.1) or 0.1)
        trade_notional = cash * base_weight * max(float(size_factor), 0.0)
        if trade_notional <= 0 or price <= 0:
            return {"status": "rejected", "reason": "invalid_notional_or_price"}

        qty = trade_notional / price
        side = "BUY" if signal == "LONG" else "SELL"
        fee = trade_notional * self.fee_rate
        return {
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "price": float(price),
            "notional": float(trade_notional),
            "fee": float(fee),
            "mark_price": float(price),
            "order_type": order_plan.order_type,
            "limit_price": order_plan.limit_price,
            "maker_preference": order_plan.maker_preference,
            "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
            "cost": cost.to_dict(),
        }
