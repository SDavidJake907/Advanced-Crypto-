from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.config.runtime import get_runtime_setting
from core.risk.fee_filter import TradeCostAssessment


@dataclass
class OrderPlan:
    order_type: str
    price: float
    limit_price: float | None
    maker_preference: bool
    reasons: list[str]
    max_retries: int = 0
    stale_ttl_sec: int = 0
    max_chase_bps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maker_limit_price(signal: str, *, bid: float, ask: float) -> float:
    return bid if signal == "LONG" else ask


def _taker_limit_price(signal: str, *, bid: float, ask: float, offset_bps: float) -> float:
    offset = offset_bps / 10_000.0
    if signal == "LONG":
        return max(ask * (1.0 + offset), bid)
    return min(bid * (1.0 - offset), ask)


def build_order_plan(features: dict[str, Any], signal: str, cost: TradeCostAssessment) -> OrderPlan:
    lane = str(features.get("lane", "L3") or "L3").upper()
    promotion_tier = str(features.get("promotion_tier", "skip") or "skip").lower()
    entry_recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
    order_preference = str(get_runtime_setting("ORDER_PREFERENCE")).strip().lower()
    aggressive_taker_enabled = bool(get_runtime_setting("ENTRY_AGGRESSIVE_TAKER_ENABLED"))
    aggressive_taker_max_spread_pct = float(get_runtime_setting("ENTRY_AGGRESSIVE_TAKER_MAX_SPREAD_PCT"))

    lane_offset_keys = {
        "L1": "L1_ORDER_LIMIT_OFFSET_BPS",
        "L2": "L2_ORDER_LIMIT_OFFSET_BPS",
        "L3": "L3_ORDER_LIMIT_OFFSET_BPS",
        "L4": "MEME_ORDER_LIMIT_OFFSET_BPS",
    }
    limit_offset_key = lane_offset_keys.get(lane, "ORDER_LIMIT_OFFSET_BPS")
    limit_offset_bps = float(get_runtime_setting(limit_offset_key))
    limit_offset_bps *= max(float(features.get("trade_quality_limit_offset_scale", 1.0)), 0.5)

    market_price = float(features.get("price", 0.0) or 0.0)
    bid = float(features.get("bid", 0.0) or 0.0)
    ask = float(features.get("ask", 0.0) or 0.0)
    spread_pct = max(float(features.get("spread_pct", 0.0) or 0.0), 0.0)
    book_valid = bool(features.get("book_valid", bid > 0.0 and ask > 0.0 and ask >= bid))

    order_type = "market"
    limit_price: float | None = None
    maker_preference = False
    reasons: list[str] = []
    max_retries = int(get_runtime_setting("ORDER_REPRICE_MAX_RETRIES"))
    stale_ttl_sec = int(get_runtime_setting("ORDER_REPRICE_TTL_SEC"))
    max_chase_bps = float(get_runtime_setting("ORDER_REPRICE_MAX_CHASE_BPS"))

    can_price_limit = book_valid and bid > 0.0 and ask > 0.0 and ask >= bid
    aggressive_entry = (
        aggressive_taker_enabled
        and signal == "LONG"
        and can_price_limit
        and spread_pct > 0.0
        and spread_pct <= aggressive_taker_max_spread_pct
        and (
            lane == "L4"
            or (lane == "L1" and promotion_tier == "promote")
            or (promotion_tier == "promote" and entry_recommendation in {"BUY", "STRONG_BUY"})
        )
    )
    if not book_valid:
        reasons.append("book_invalid")

    if order_preference == "market":
        reasons.append("market_forced")
    elif not can_price_limit:
        reasons.append("no_bid_ask_fallback_market")
    elif aggressive_entry:
        order_type = "limit"
        limit_price = _taker_limit_price(signal, bid=bid, ask=ask, offset_bps=limit_offset_bps)
        maker_preference = False
        reasons.append("aggressive_entry_taker_limit")
    elif lane == "L1":
        order_type = "limit"
        limit_price = _maker_limit_price(signal, bid=bid, ask=ask)
        maker_preference = True
        reasons.append("lane1_patient_maker")
    elif lane == "L2":
        order_type = "limit"
        limit_price = _maker_limit_price(signal, bid=bid, ask=ask)
        maker_preference = True
        reasons.append("lane2_maker_first")
    elif lane == "L4":
        order_type = "limit"
        limit_price = _maker_limit_price(signal, bid=bid, ask=ask)
        maker_preference = True
        reasons.append("lane4_maker_first")
    else:
        order_type = "limit"
        limit_price = _maker_limit_price(signal, bid=bid, ask=ask)
        maker_preference = True
        reasons.append("lane3_balanced_maker")

    # Nudge BUY limit slightly above bid (improves queue priority / fill rate).
    # Clamp below ask so post-only orders are never rejected for crossing the spread.
    if limit_price and limit_price > 0.0 and can_price_limit and signal == "LONG" and limit_offset_bps > 0.0:
        nudged = limit_price * (1.0 + limit_offset_bps / 10_000.0)
        limit_price = min(nudged, ask * (1.0 - 1e-4))

    execution_price = limit_price if limit_price and limit_price > 0.0 else market_price
    return OrderPlan(
        order_type=order_type,
        price=execution_price,
        limit_price=limit_price,
        maker_preference=maker_preference,
        reasons=reasons,
        max_retries=max_retries,
        stale_ttl_sec=stale_ttl_sec,
        max_chase_bps=max_chase_bps,
    )


def build_exit_order_plan(features: dict[str, Any], side: str, exit_reason: str) -> OrderPlan:
    bid = float(features.get("bid", 0.0) or 0.0)
    ask = float(features.get("ask", 0.0) or 0.0)
    price = float(features.get("price", 0.0) or 0.0)
    book_valid = bool(features.get("book_valid", bid > 0.0 and ask > 0.0 and ask >= bid))
    reason = str(exit_reason or "").lower()
    emergency_exit = (
        "hard_fail" in reason
        or "l4_momentum_collapse" in reason
        or "breakout_failed" in reason
    )
    if not book_valid or emergency_exit:
        reasons = ["forced_market_exit"]
        if emergency_exit:
            reasons.append("emergency_exit")
        if not book_valid:
            reasons.append("book_invalid")
        return OrderPlan(
            order_type="market",
            price=price,
            limit_price=None,
            maker_preference=False,
            reasons=reasons,
            max_retries=0,
            stale_ttl_sec=0,
            max_chase_bps=0.0,
        )

    take_profit_exit = "take_profit" in reason
    protective_exit = (
        reason == "stop_loss"
        or "trail" in reason
        or "exit_posture" in reason
        or "stale" in reason
    )
    exit_offset_bps = float(get_runtime_setting("EXIT_ORDER_REPRICE_MAX_CHASE_BPS"))
    if side.upper() == "LONG":
        limit_price = _taker_limit_price("SHORT", bid=bid, ask=ask, offset_bps=exit_offset_bps)
    else:
        limit_price = _taker_limit_price("LONG", bid=bid, ask=ask, offset_bps=exit_offset_bps)
    stale_ttl_sec = int(get_runtime_setting("EXIT_ORDER_REPRICE_TTL_SEC"))
    if protective_exit:
        stale_ttl_sec = int(get_runtime_setting("EXIT_PROTECTIVE_ORDER_TTL_SEC"))
    elif take_profit_exit:
        stale_ttl_sec = int(get_runtime_setting("EXIT_TAKE_PROFIT_ORDER_TTL_SEC"))
    return OrderPlan(
        order_type="limit",
        price=limit_price,
        limit_price=limit_price,
        maker_preference=False,
        reasons=["limit_exit_preferred", "crossing_limit_exit"] + (["protective_exit"] if protective_exit else []) + (["take_profit_exit"] if take_profit_exit else []),
        max_retries=int(get_runtime_setting("EXIT_ORDER_REPRICE_MAX_RETRIES")),
        stale_ttl_sec=stale_ttl_sec,
        max_chase_bps=float(get_runtime_setting("EXIT_ORDER_REPRICE_MAX_CHASE_BPS")),
    )
