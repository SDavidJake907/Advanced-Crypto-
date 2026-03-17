from __future__ import annotations

import time
from typing import Any

from core.config.runtime import get_runtime_setting, is_meme_lane
from core.data.kraken_rest import KrakenRestClient
from core.execution.clamp import clamp_order_size
from core.execution.kraken_rules import load_rules
from core.execution.order_policy import OrderPlan, build_order_plan
from core.risk.fee_filter import evaluate_trade_cost
from core.state.open_orders import has_active_order_for_symbol, upsert_open_order


class KrakenLiveExecutor:
    def __init__(self, fee_rate: float = 0.001) -> None:
        self.fee_rate = fee_rate
        self.client = KrakenRestClient()
        self._rules = load_rules(self.client)

    def _build_client_order_id(self, symbol: str, side: str) -> str:
        token = symbol.replace("/", "").upper()
        return f"ksk-{token}-{side.lower()}-{int(time.time() * 1000)}"

    def _submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_plan: OrderPlan,
        features: dict[str, Any],
        cost: dict[str, Any] | None = None,
        exit_reason: str | None = None,
    ) -> dict[str, Any]:
        if qty <= 0.0:
            return {"status": "rejected", "reason": "invalid_qty"}
        if has_active_order_for_symbol(symbol):
            return {"status": "no_trade", "reason": "existing_open_order"}
        client_order_id = self._build_client_order_id(symbol, side)
        kraken_symbol = self.client.resolve_order_pair(symbol).replace("/", "").upper()
        rule = self._rules.get(kraken_symbol)
        limit_price = order_plan.limit_price
        post_only = bool(get_runtime_setting("ORDER_POST_ONLY"))
        if limit_price is not None and rule is not None:
            limit_price = round(float(limit_price), int(rule.pair_decimals))
            if rule.tick_size:
                tick = float(rule.tick_size)
                if tick > 0.0:
                    limit_price = round(round(limit_price / tick) * tick, int(rule.pair_decimals))
        try:
            response = self.client.add_order(
                symbol=symbol,
                side=side.lower(),
                ordertype=order_plan.order_type,
                volume=qty,
                price=limit_price if order_plan.order_type == "limit" else None,
                oflags="post" if (post_only and order_plan.order_type == "limit" and order_plan.maker_preference) else None,
            )
        except Exception as exc:
            return {
                "status": "rejected",
                "reason": str(exc),
                "symbol": symbol,
                "side": side.upper(),
                "qty": float(qty),
                "price": float(order_plan.price),
                "notional": float(qty * float(order_plan.price)),
                "order_type": order_plan.order_type,
                "limit_price": limit_price,
                "maker_preference": order_plan.maker_preference,
                "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
                "cost": cost,
                **({"exit_reason": exit_reason} if exit_reason else {}),
            }
        result = response.get("result", {})
        txid = None
        txids = result.get("txid")
        if isinstance(txids, list) and txids:
            txid = txids[0]
        descr = result.get("descr", {}) if isinstance(result.get("descr", {}), dict) else {}
        upsert_open_order(
            client_order_id,
            {
                "symbol": symbol,
                "side": side.upper(),
                "status": "open",
                "txid": txid,
                "submitted_ts": time.time(),
                "order_type": order_plan.order_type,
                "limit_price": limit_price,
                "qty": qty,
                "exit_reason": exit_reason,
            },
        )
        return {
            "status": "submitted",
            "symbol": symbol,
            "side": side.upper(),
            "qty": float(qty),
            "price": float(order_plan.price),
            "notional": float(qty * float(order_plan.price)),
            "fee": 0.0,
            "mark_price": float(order_plan.price),
            "order_type": order_plan.order_type,
            "limit_price": limit_price,
            "maker_preference": order_plan.maker_preference,
            "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
            "cost": cost,
            "txid": txid,
            "client_order_id": client_order_id,
            "descr": descr,
            "live": True,
            **({"exit_reason": exit_reason} if exit_reason else {}),
        }

    def execute(
        self,
        signal: str,
        symbol: str,
        features: dict[str, Any],
        state: dict[str, Any],
        size_factor: float = 1.0,
    ) -> dict[str, Any]:
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
        if notional <= 0.0 or price <= 0.0:
            return {"status": "rejected", "reason": "invalid_notional_or_price"}
        lane = str(features.get("lane") or "")
        min_notional_usd = float(
            get_runtime_setting("MEME_EXEC_MIN_NOTIONAL_USD")
            if is_meme_lane(lane)
            else get_runtime_setting("EXEC_MIN_NOTIONAL_USD")
        )
        risk_per_trade_pct = float(
            get_runtime_setting("MEME_EXEC_RISK_PER_TRADE_PCT")
            if is_meme_lane(lane)
            else get_runtime_setting("EXEC_RISK_PER_TRADE_PCT")
        )
        kraken_symbol = self.client.resolve_order_pair(symbol).replace("/", "").upper()
        rule = self._rules.get(kraken_symbol)
        if rule is None:
            return {"status": "rejected", "reason": f"missing_pair_rule:{kraken_symbol}"}
        clamp = clamp_order_size(
            pair=kraken_symbol,
            price=price,
            desired_usd=notional,
            rule=rule,
            equity_usd=cash,
            max_position_usd=max(cash, notional),
            risk_per_trade_pct=risk_per_trade_pct,
            min_notional_usd=max(float(features.get("min_notional_usd", 0.0) or 0.0), min_notional_usd),
            clamp_up=True,
            kelly_fraction=float(features.get("kelly_fraction", 1.0) or 1.0),
        )
        if not clamp.ok:
            return {
                "status": "rejected",
                "reason": clamp.reason,
                "price": clamp.price,
                "notional": clamp.notional_usd,
            }
        qty = clamp.size_base
        side = "BUY" if signal == "LONG" else "SELL"
        return self._submit_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_plan=order_plan,
            features=features,
            cost=cost.to_dict(),
        )

    def execute_exit(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        features: dict[str, Any],
        exit_reason: str,
    ) -> dict[str, Any]:
        order_plan = OrderPlan(
            order_type="market",
            price=price,
            limit_price=None,
            maker_preference=False,
            reasons=["forced_market_exit"],
        )
        live_side = "SELL" if side == "LONG" else "BUY"
        return self._submit_order(
            symbol=symbol,
            side=live_side,
            qty=qty,
            order_plan=order_plan,
            features=features,
            cost=None,
            exit_reason=exit_reason,
        )
