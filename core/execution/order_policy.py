from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.config.runtime import get_runtime_setting, is_meme_lane
from core.risk.fee_filter import TradeCostAssessment


@dataclass
class OrderPlan:
    order_type: str
    price: float
    limit_price: float | None
    maker_preference: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_order_plan(features: dict[str, Any], signal: str, cost: TradeCostAssessment) -> OrderPlan:
    lane = str(features.get("lane", "L3"))
    order_preference = str(get_runtime_setting("ORDER_PREFERENCE")).strip().lower()
    offset_key = "MEME_ORDER_LIMIT_OFFSET_BPS" if is_meme_lane(lane) else "ORDER_LIMIT_OFFSET_BPS"
    limit_offset_bps = float(get_runtime_setting(offset_key))
    market_price = float(features.get("price", 0.0) or 0.0)
    bid = float(features.get("bid", 0.0) or 0.0)
    ask = float(features.get("ask", 0.0) or 0.0)

    order_type = "market"
    limit_price: float | None = None
    maker_preference = False
    reasons: list[str] = []

    can_price_limit = bid > 0.0 and ask > 0.0 and ask >= bid
    prefer_limit = order_preference == "limit" or (order_preference == "auto" and can_price_limit)

    if prefer_limit and can_price_limit:
        maker_preference = True
        order_type = "limit"
        offset = limit_offset_bps / 10_000.0
        if signal == "LONG":
            limit_price = min(ask * (1.0 - offset), ask)
            limit_price = max(limit_price, bid)
        else:
            limit_price = max(bid * (1.0 + offset), bid)
            limit_price = min(limit_price, ask)
        reasons.append("maker_limit_preferred")
    elif order_preference == "market":
        reasons.append("market_forced")
    elif not can_price_limit:
        reasons.append("no_bid_ask_fallback_market")

    execution_price = limit_price if limit_price and limit_price > 0.0 else market_price
    return OrderPlan(
        order_type=order_type,
        price=execution_price,
        limit_price=limit_price,
        maker_preference=maker_preference,
        reasons=reasons,
    )
