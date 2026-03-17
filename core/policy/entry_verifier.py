from __future__ import annotations

from typing import Any

import numpy as np

from core.config.runtime import get_runtime_setting, get_symbol_lane


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean_abs_correlation(correlation_row: Any) -> float:
    if correlation_row is None:
        return 0.0
    row = np.asarray(correlation_row, dtype=np.float64).ravel()
    if row.size == 0:
        return 0.0
    filtered = [abs(float(v)) for v in row if np.isfinite(v) and abs(float(v)) < 0.999]
    if not filtered:
        return 0.0
    return float(np.mean(filtered))


def compute_entry_verification(
    features: dict[str, Any],
    finbert_score: float = 0.0,
    xgb_score: float = -1.0,
) -> dict[str, Any]:
    lane = str(features.get("lane") or get_symbol_lane(str(features.get("symbol", ""))))
    lane_filter_pass = bool(features.get("lane_filter_pass", True))
    lane_filter_reason = str(features.get("lane_filter_reason", "") or "")
    regime_state = str(features.get("regime_state", "unknown") or "unknown").lower()
    momentum_5 = float(features.get("momentum_5", 0.0))
    momentum_14 = float(features.get("momentum_14", features.get("momentum", 0.0)))
    momentum_30 = float(features.get("momentum_30", 0.0))
    rotation_score = float(features.get("rotation_score", 0.0))
    trend_1h = int(features.get("trend_1h", 0))
    rsi = float(features.get("rsi", 50.0))
    volume_ratio = float(features.get("volume_ratio", 1.0))
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    price_zscore = float(features.get("price_zscore", 0.0))
    regime_7d = str(features.get("regime_7d", "unknown"))
    macro_30d = str(features.get("macro_30d", "sideways"))
    hurst = float(features.get("hurst", 0.5))
    autocorr = float(features.get("autocorr", 0.0))
    entropy = float(features.get("entropy", 0.5))
    crowding = _mean_abs_correlation(features.get("correlation_row"))
    book_imbalance = float(features.get("book_imbalance", 0.0) or 0.0)
    book_wall_pressure = float(features.get("book_wall_pressure", 0.0) or 0.0)
    sentiment_market_cap_change = float(features.get("sentiment_market_cap_change_24h", 0.0) or 0.0)
    sentiment_symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    # Accept finbert_score and xgb_score from features dict if not passed directly
    finbert_score = float(features.get("finbert_score", finbert_score) or finbert_score)
    xgb_score = float(features.get("xgb_score", xgb_score))

    score = 50.0
    reasons: list[str] = []

    score += _clamp(momentum_5 * 900.0, -20.0, 25.0)
    score += _clamp(momentum_14 * 500.0, -12.0, 16.0)
    score += _clamp(momentum_30 * 300.0, -8.0, 10.0)
    score += _clamp(rotation_score * 25.0, -10.0, 15.0)
    score += trend_1h * 6.0
    score += _clamp((volume_ratio - 1.0) * 10.0, -4.0, 10.0)
    score += _clamp(volume_surge * 8.0, 0.0, 8.0)
    score -= _clamp(abs(price_zscore) * 3.0, 0.0, 10.0)
    score += _clamp(book_imbalance * 8.0, -6.0, 6.0)
    score += _clamp(book_wall_pressure * 4.0, -3.0, 3.0)
    score += _clamp(sentiment_market_cap_change * 0.4, -2.0, 2.0)

    if lane == "L4":
        score += _clamp(momentum_5 * 1200.0, -10.0, 18.0)
        score += _clamp(volume_surge * 10.0, 0.0, 10.0)
        if sentiment_symbol_trending:
            score += 4.0
        if momentum_5 > 0.008 and volume_ratio >= 1.0:
            score += 4.0
            reasons.append("meme_heat")
        if book_imbalance >= 0.1:
            score += 2.0
            reasons.append("meme_bid_pressure")

    if regime_7d == "trending":
        score += 6.0
        reasons.append("regime_trending")
    elif regime_7d == "choppy":
        score -= 8.0
        reasons.append("regime_choppy")

    if regime_state == "bullish":
        score += 4.0
        reasons.append("regime_state_bullish")
    elif regime_state == "bearish":
        score -= 8.0
        reasons.append("regime_state_bearish")
    elif regime_state == "volatile":
        score -= 5.0
        reasons.append("regime_state_volatile")

    if macro_30d == "bull":
        score += 4.0
        reasons.append("macro_bull")
    elif macro_30d == "bear":
        score -= 6.0
        reasons.append("macro_bear")

    if hurst >= 0.58:
        score += 4.0
        reasons.append("persistent_trend")
    elif hurst <= 0.45:
        score -= 4.0
        reasons.append("mean_reverting")

    score += _clamp(autocorr * 8.0, -4.0, 4.0)
    score -= _clamp((entropy - 0.75) * 10.0, 0.0, 4.0)
    score -= _clamp((crowding - 0.8) * 12.0, 0.0, 5.0)

    rsi_penalty_threshold = 85.0 if lane == "L4" else 75.0
    if rsi > rsi_penalty_threshold:
        score -= _clamp((rsi - rsi_penalty_threshold) * 0.75, 0.0, 12.0)
        reasons.append("overbought_penalty")
    elif rsi < 30.0:
        score -= _clamp((30.0 - rsi) * 0.6, 0.0, 10.0)
        reasons.append("weak_rsi")

    if momentum_5 > 0.01:
        reasons.append("fast_momentum")
    if volume_ratio >= 1.5:
        reasons.append("volume_expansion")
    if volume_surge >= 0.5:
        reasons.append("volume_surge")
    if rotation_score >= 0.1:
        reasons.append("rotation_leader")
    if book_imbalance >= 0.2:
        reasons.append("bid_pressure")
    elif book_imbalance <= -0.2:
        reasons.append("ask_pressure")
    if sentiment_symbol_trending:
        reasons.append("symbol_trending")

    # FinBERT sentiment scoring
    score += _clamp(finbert_score * 12.0, -6.0, 8.0)
    if finbert_score >= 0.2:
        reasons.append("finbert_positive")
    elif finbert_score <= -0.2:
        reasons.append("finbert_negative")

    reversal_risk = "LOW"
    high_reversal_zscore = 3.1 if lane == "L4" else 2.4
    medium_reversal_zscore = 1.9 if lane == "L4" else 1.5
    if (
        price_zscore >= high_reversal_zscore
        or (rsi > (rsi_penalty_threshold + 7.0) and momentum_5 < 0.0 and momentum_5 < momentum_14)
        or crowding >= 0.92
    ):
        reversal_risk = "HIGH"
    elif price_zscore >= medium_reversal_zscore or regime_7d == "choppy" or momentum_5 < -0.005:
        reversal_risk = "MEDIUM"

    # Escalate reversal_risk if FinBERT sentiment is strongly negative
    if finbert_score < -0.6 and reversal_risk == "LOW":
        reversal_risk = "MEDIUM"

    # XGBoost score blending (only when score >= 0, i.e. model has a prediction)
    if xgb_score >= 0.0:
        score = score * 0.65 + xgb_score * 0.35
        if xgb_score > 60.0:
            reasons.append("xgb_confirmed")

    score = _clamp(score, 0.0, 100.0)
    strong_buy_threshold = float(get_runtime_setting("MEME_ENTRY_SCORE_STRONG_BUY_THRESHOLD")) if lane == "L4" else 68.0
    buy_threshold = float(get_runtime_setting("MEME_ENTRY_SCORE_BUY_THRESHOLD")) if lane == "L4" else 52.0
    recommendation = "WATCH"
    if score >= strong_buy_threshold and reversal_risk == "LOW":
        recommendation = "STRONG_BUY"
    elif score >= buy_threshold and reversal_risk != "HIGH":
        recommendation = "BUY"
    elif score <= 35.0 or reversal_risk == "HIGH":
        recommendation = "AVOID"

    if not lane_filter_pass:
        score = max(score - 12.0, 0.0)
        if lane_filter_reason:
            reasons.append(lane_filter_reason)
        if recommendation == "STRONG_BUY":
            recommendation = "BUY"
        elif recommendation == "BUY":
            recommendation = "WATCH"
        elif recommendation == "WATCH":
            recommendation = "AVOID"
        if reversal_risk == "LOW":
            reversal_risk = "MEDIUM"

    return {
        "entry_score": round(score, 2),
        "entry_recommendation": recommendation,
        "reversal_risk": reversal_risk,
        "entry_reasons": list(dict.fromkeys(reasons)),
    }
