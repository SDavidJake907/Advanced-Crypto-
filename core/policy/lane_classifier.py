from __future__ import annotations

from typing import Any

from core.config.runtime import is_meme_symbol


def classify_lane(symbol: str, features: dict[str, Any]) -> str:
    if symbol in {"BTC/USD", "XBTUSD", "XXBTZUSD", "XBT/USD"}:
        return "L1"
    if is_meme_symbol(symbol):
        return "L4"

    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    momentum_30 = float(features.get("momentum_30", 0.0) or 0.0)
    trend_1h = int(features.get("trend_1h", 0) or 0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    price_zscore = float(features.get("price_zscore", 0.0) or 0.0)
    regime_7d = str(features.get("regime_7d", "unknown") or "unknown").lower()
    rsi = float(features.get("rsi", 50.0) or 50.0)
    price = float(features.get("price", 0.0) or 0.0)
    bb_low = float(features.get("bb_lower", 0.0) or 0.0)
    bb_up = float(features.get("bb_upper", 0.0) or 0.0)

    # Fast-mover breakout — strong short-term surge + volume → treat as meme/L4
    if momentum_5 > 0.02 and volume_ratio >= 1.5:
        return "L4"

    if (
        (volume_ratio >= 1.6 and momentum_5 > 0.006 and trend_1h >= 0)
        or (trend_1h >= 1 and momentum_14 > 0.012 and momentum_30 > 0.0 and volume_ratio >= 1.05 and 45.0 <= rsi <= 78.0)
    ):
        return "L1"

    near_lower = price > 0 and bb_low > 0 and price <= bb_low * 1.01
    near_upper = price > 0 and bb_up > 0 and price >= bb_up * 0.99
    if (
        (
            trend_1h >= 0
            and momentum_5 >= 0.001
            and momentum_14 > -0.001
            and rotation_score > 0.0
            and volume_ratio >= 0.5
            and 35.0 <= rsi <= 78.0
            and price_zscore > -1.5
        )
        or (
            regime_7d == "choppy"
            and (price_zscore <= -0.75 or near_lower or near_upper)
        )
    ):
        return "L2"

    if (
        trend_1h >= 0
        and momentum_14 >= 0.0
        and momentum_30 >= 0.0
        and volume_ratio >= 0.6
        and momentum_5 <= 0.008
        and rotation_score <= 0.06
        and -0.75 <= price_zscore <= 1.0
        and 42.0 <= rsi <= 68.0
    ):
        return "L3"

    return "L3"
