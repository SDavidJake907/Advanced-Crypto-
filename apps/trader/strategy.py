from __future__ import annotations

from apps.trader.features import Features
from apps.trader.types import CandidateOrder


def generate_candidate(
    *,
    symbol: str,
    price: float,
    features: Features,
    desired_usd: float,
) -> CandidateOrder | None:
    # Long-only v1
    if features.trend != "up":
        return None
    if not features.breakout:
        return None
    if not features.volume_confirm:
        return None
    if features.stretch == "stretched":
        return None

    return CandidateOrder(
        symbol=symbol,
        side="buy",
        order_type="limit",
        price=price,
        size_usd=desired_usd,
        stop_atr_mult=1.8,
        reason="15m_breakout_trend_vol_confirm",
    )
