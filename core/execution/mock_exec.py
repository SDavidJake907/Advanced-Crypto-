from __future__ import annotations

import os
from typing import Any, Dict

from core.execution.base import Executor
from core.config.runtime import get_proposed_weight, get_runtime_setting
from core.execution.order_policy import build_order_plan
from core.execution.risk_budget import estimate_entry_risk_fraction
from core.risk.fee_filter import evaluate_trade_cost
from core.risk.trade_quality import assess_trade_quality


class MockExecutor(Executor):
    def __init__(self, fee_rate: float = 0.001, allow_short: bool | None = None):
        self.fee_rate = fee_rate
        if allow_short is None:
            raw = str(os.getenv("ALLOW_SHORTS", "false")).strip().lower()
            allow_short = raw in {"1", "true", "yes", "on"}
        self.allow_short = bool(allow_short)

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
        if signal == "SHORT" and not self.allow_short:
            return {"status": "blocked", "reason": "shorts_disabled"}

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
        trade_notional = cash * base_weight * max(float(size_factor), 0.0) * tq.size_scale
        if trade_notional <= 0 or price <= 0:
            return {"status": "rejected", "reason": "invalid_notional_or_price"}

        min_notional_usd = max(float(features.get("min_notional_usd", 0.0) or 0.0), float(get_runtime_setting("EXEC_MIN_NOTIONAL_USD")))
        risk_budget_usd = cash * (float(get_runtime_setting("EXEC_RISK_PER_TRADE_PCT")) / 100.0)
        risk_fraction_of_notional = estimate_entry_risk_fraction(
            price=price,
            atr=float(features.get("atr", 0.0) or 0.0),
            lane=lane,
        )
        if trade_notional < min_notional_usd and (min_notional_usd * risk_fraction_of_notional) > (risk_budget_usd * float(get_runtime_setting("EXEC_MIN_TRADE_RISK_BUDGET_MULT"))):
            return {"status": "blocked", "reason": ["min_trade_exceeds_risk_budget"], "cost": cost.to_dict()}
        trade_notional = max(trade_notional, min_notional_usd)
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
            "trade_quality": tq.to_dict(),
        }

    def execute_exit(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        features: dict,
        exit_reason: str,
    ) -> Dict[str, Any]:
        exit_price = float(price or features.get("price", 0.0) or 0.0)
        exit_qty = abs(float(qty or 0.0))
        if exit_qty <= 0.0 or exit_price <= 0.0:
            return {"status": "rejected", "reason": "invalid_exit_qty_or_price"}

        exit_notional = exit_qty * exit_price
        fee = exit_notional * self.fee_rate
        live_side = "SELL" if side == "LONG" else "BUY"
        return {
            "status": "filled",
            "symbol": symbol,
            "side": live_side,
            "qty": exit_qty,
            "price": exit_price,
            "notional": float(exit_notional),
            "fee": float(fee),
            "mark_price": float(exit_price),
            "exit_reason": str(exit_reason or ""),
            "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
        }
