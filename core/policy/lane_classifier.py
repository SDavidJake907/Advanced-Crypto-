from __future__ import annotations

from typing import Any

from core.config.runtime import is_meme_symbol


def classify_lane(symbol: str, features: dict[str, Any]) -> str:
    if is_meme_symbol(symbol):
        return "L4"

    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    momentum_30 = float(features.get("momentum_30", 0.0) or 0.0)
    trend_1h = int(features.get("trend_1h", 0) or 0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    price_zscore = float(features.get("price_zscore", 0.0) or 0.0)
    regime_7d = str(features.get("regime_7d", "unknown") or "unknown").lower()
    rsi = float(features.get("rsi", 50.0) or 50.0)
    price = float(features.get("price", 0.0) or 0.0)
    bb_low = float(features.get("bb_lower", 0.0) or 0.0)
    bb_up = float(features.get("bb_upper", 0.0) or 0.0)

    if (volume_ratio >= 1.8 and momentum_5 > 0.006) or (trend_1h >= 1 and momentum_14 > 0.01 and momentum_5 > 0):
        return "L1"

    if regime_7d == "choppy":
        near_lower = price > 0 and bb_low > 0 and price <= bb_low * 1.01
        near_upper = price > 0 and bb_up > 0 and price >= bb_up * 0.99
        if price_zscore <= -0.75 or near_lower or near_upper:
            return "L2"

    if trend_1h >= 0 and (momentum_14 > -0.002 or momentum_30 > 0) and 35.0 <= rsi <= 75.0:
        return "L3"

    return "L3"
