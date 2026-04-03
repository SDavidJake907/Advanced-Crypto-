from __future__ import annotations

import time
from typing import Any

from core.config.runtime import get_proposed_weight, get_runtime_setting, is_meme_lane
from core.data.kraken_rest import KrakenRestClient
from core.execution.clamp import clamp_order_size
from core.execution.risk_budget import estimate_entry_risk_fraction, calculate_dynamic_risk_pct
from core.execution.kraken_rules import load_rules
from core.execution.order_policy import OrderPlan, build_exit_order_plan, build_order_plan
from core.risk.fee_filter import evaluate_trade_cost
from core.risk.trade_quality import assess_trade_quality
from core.state.open_orders import has_active_order_for_symbol, has_recent_buy_fill_for_symbol, reconcile_open_orders, upsert_open_order


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
        order_role: str = "entry",
    ) -> dict[str, Any]:
        if qty <= 0.0:
            return {"status": "rejected", "reason": "invalid_qty"}
        reconcile_open_orders(
            client=self.client,
            symbol=symbol,
            timeout_sec=int(order_plan.stale_ttl_sec or get_runtime_setting("ORDER_OPEN_TTL_SEC")),
        )
        if has_active_order_for_symbol(symbol):
            return {"status": "no_trade", "reason": "existing_open_order"}
        # Block duplicate buys when a limit order just filled but portfolio state hasn't synced yet
        if side.upper() == "BUY" and has_recent_buy_fill_for_symbol(symbol, cooldown_sec=600.0):
            return {"status": "no_trade", "reason": "recent_fill_cooldown"}
        client_order_id = self._build_client_order_id(symbol, side)
        kraken_symbol = self.client.resolve_order_pair(symbol).replace("/", "").upper()
        rule = self._rules.get(kraken_symbol)
        limit_price = order_plan.limit_price
        post_only = bool(get_runtime_setting("ORDER_POST_ONLY"))
        tick = 0.0
        initial_limit_price = limit_price
        max_chase_bps = max(float(order_plan.max_chase_bps or 0.0), 0.0)
        if limit_price is not None and rule is not None:
            limit_price = round(float(limit_price), int(rule.pair_decimals))
            if rule.tick_size:
                tick = float(rule.tick_size)
                if tick > 0.0:
                    limit_price = round(round(limit_price / tick) * tick, int(rule.pair_decimals))
            initial_limit_price = limit_price
        response: dict[str, Any] | None = None
        retry_count = 0
        max_retries = max(int(order_plan.max_retries), 0)
        current_limit_price = limit_price
        while True:
            try:
                response = self.client.add_order(
                    symbol=symbol,
                    side=side.lower(),
                    ordertype=order_plan.order_type,
                    volume=qty,
                    price=current_limit_price if order_plan.order_type == "limit" else None,
                    oflags="post" if (post_only and order_plan.order_type == "limit" and order_plan.maker_preference) else None,
                )
                break
            except Exception as exc:
                message = str(exc)
                if (
                    order_plan.order_type == "limit"
                    and order_plan.maker_preference
                    and retry_count < max_retries
                    and any(
                        fragment in message.lower()
                        for fragment in (
                            "post only",
                            "post-only",
                            "would immediately execute",
                            "immediate or cancel",
                            "would match and take",
                        )
                    )
                ):
                    retry_count += 1
                    if tick <= 0.0:
                        tick = max(float(current_limit_price or order_plan.price or 0.0) * 0.0005, 1e-8)
                    if side.upper() == "BUY":
                        current_limit_price = max(float(current_limit_price or 0.0) - tick, tick)
                    else:
                        current_limit_price = float(current_limit_price or 0.0) + tick
                    if rule is not None:
                        current_limit_price = round(float(current_limit_price), int(rule.pair_decimals))
                    if initial_limit_price and max_chase_bps > 0.0:
                        drift_bps = abs((float(current_limit_price) - float(initial_limit_price)) / float(initial_limit_price)) * 10_000.0
                        if drift_bps > max_chase_bps:
                            return {
                                "status": "no_trade",
                                "reason": "post_only_reprice_exhausted",
                                "symbol": symbol,
                                "side": side.upper(),
                                "qty": float(qty),
                                "price": float(order_plan.price),
                                "notional": float(qty * float(order_plan.price)),
                                "order_type": order_plan.order_type,
                                "limit_price": current_limit_price,
                                "maker_preference": order_plan.maker_preference,
                                "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
                                "retry_count": retry_count,
                                "max_chase_bps": max_chase_bps,
                                "cost": cost,
                                "order_role": order_role,
                                **({"exit_reason": exit_reason} if exit_reason else {}),
                            }
                    continue
                return {
                    "status": "rejected",
                    "reason": message,
                    "symbol": symbol,
                    "side": side.upper(),
                    "qty": float(qty),
                    "price": float(order_plan.price),
                    "notional": float(qty * float(order_plan.price)),
                    "order_type": order_plan.order_type,
                    "limit_price": current_limit_price,
                    "maker_preference": order_plan.maker_preference,
                    "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
                    "retry_count": retry_count,
                    "cost": cost,
                    "order_role": order_role,
                    **({"exit_reason": exit_reason} if exit_reason else {}),
                }
        assert response is not None
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
                "limit_price": current_limit_price,
                "qty": qty,
                "exit_reason": exit_reason,
                "retry_count": retry_count,
                "reprice_count": retry_count,
                "order_role": order_role,
                "stale_ttl_sec": int(order_plan.stale_ttl_sec),
                "order_plan_reasons": list(order_plan.reasons),
                "spread_pct_at_submit": float(features.get("spread_pct", 0.0) or 0.0),
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
            "limit_price": current_limit_price,
            "maker_preference": order_plan.maker_preference,
            "spread_pct": float(features.get("spread_pct", 0.0) or 0.0),
            "retry_count": retry_count,
            "cost": cost,
            "txid": txid,
            "client_order_id": client_order_id,
            "descr": descr,
            "live": True,
            "order_role": order_role,
            "stale_ttl_sec": int(order_plan.stale_ttl_sec),
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
        if float(cost.net_edge_pct) <= 0.0:
            return {"status": "blocked", "reason": ["net_edge_non_positive"], "cost": cost.to_dict()}
        bid = float(features.get("bid", 0.0) or 0.0)
        ask = float(features.get("ask", 0.0) or 0.0)
        book_valid = bool(features.get("book_valid", False))
        if not book_valid or bid <= 0.0 or ask <= 0.0:
            return {"status": "blocked", "reason": ["book_invalid_entry"], "cost": cost.to_dict()}
        tq = assess_trade_quality(features, str(features.get("lane", "L3")))
        _exec_features = dict(features)
        _exec_features["trade_quality_limit_offset_scale"] = tq.limit_offset_scale
        order_plan = build_order_plan(_exec_features, signal, cost)
        price = float(order_plan.price or 0.0)
        cash = float(state.get("cash", 0.0) or 0.0)
        lane = str(features.get("lane") or "")
        fallback_weight = get_proposed_weight(symbol=symbol, lane=lane)
        base_weight = float(features.get("proposed_weight", fallback_weight) or fallback_weight)
        notional = cash * base_weight * max(float(size_factor), 0.0)
        if notional <= 0.0 or price <= 0.0:
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
                "status": "rejected",
                "reason": clamp.reason,
                "price": clamp.price,
                "notional": clamp.notional_usd,
                "order_plan_reasons": order_plan.reasons,
            }
        qty = clamp.size_base
        side = "BUY" if signal == "LONG" else "SELL"
        result = self._submit_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_plan=order_plan,
            features=features,
            cost=cost.to_dict(),
            order_role="entry",
        )
        result["trade_quality"] = tq.to_dict()
        result["order_plan_reasons"] = order_plan.reasons
        return result

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
        exit_features = dict(features)
        if not exit_features.get("price"):
            exit_features["price"] = price
        order_plan = build_exit_order_plan(exit_features, side=side, exit_reason=exit_reason)
        live_side = "SELL" if side == "LONG" else "BUY"
        return self._submit_order(
            symbol=symbol,
            side=live_side,
            qty=qty,
            order_plan=order_plan,
            features=features,
            cost=None,
            exit_reason=exit_reason,
            order_role="exit",
        )
