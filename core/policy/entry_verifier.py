from __future__ import annotations

from typing import Any

import numpy as np

from core.config.runtime import get_runtime_setting, get_symbol_lane


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _profile_value(features: dict[str, Any], key: str, default: float = 50.0) -> float:
    profile = features.get("coin_profile")
    if isinstance(profile, dict):
        try:
            return float(profile.get(key, default))
        except (TypeError, ValueError):
            return default
    try:
        return float(features.get(key, default))
    except (TypeError, ValueError):
        return default


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


def _lane_entry_thresholds(lane: str) -> tuple[float, float]:
    lane = str(lane or "L3").upper()
    if lane == "L1":
        return 72.0, 56.0
    if lane == "L2":
        return 70.0, 52.0
    if lane == "L4":
        return (
            float(get_runtime_setting("MEME_ENTRY_SCORE_STRONG_BUY_THRESHOLD")),
            float(get_runtime_setting("MEME_ENTRY_SCORE_BUY_THRESHOLD")),
        )
    return 68.0, 54.0


def _lane_probe_threshold(lane: str) -> float:
    lane = str(lane or "L3").upper()
    if lane == "L1":
        return 60.0
    if lane == "L2":
        return 54.0
    if lane == "L4":
        return 52.0
    return 60.0


def _structure_confirmation_scale(
    *,
    structure_profile_present: bool,
    structure_quality: float,
    continuation_quality: float,
    ema9_above_ema20: bool,
    price_above_ema20: bool,
    range_breakout_1h: bool,
    pullback_hold: bool,
    higher_low_count: int,
    pivot_break: bool,
) -> float:
    if not structure_profile_present:
        return 1.0

    strong_structure = (
        (structure_quality >= 62.0 and continuation_quality >= 60.0)
        or (ema9_above_ema20 and price_above_ema20 and (range_breakout_1h or pullback_hold))
        or (higher_low_count >= 3 and (ema9_above_ema20 or pivot_break))
    )
    if strong_structure:
        return 1.0

    developing_structure = (
        structure_quality >= 56.0
        or continuation_quality >= 56.0
        or (ema9_above_ema20 and price_above_ema20)
        or pullback_hold
        or range_breakout_1h
    )
    if developing_structure:
        return 0.7
    return 0.45


def compute_entry_verification(
    features: dict[str, Any],
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
    atr_pct = float(features.get("atr_pct", 0.0) or 0.0)
    volatility_percentile = float(features.get("volatility_percentile", 50.0) or 50.0)
    compression_score = float(features.get("compression_score", 50.0) or 50.0)
    expansion_score = float(features.get("expansion_score", 50.0) or 50.0)
    volatility_state = str(features.get("volatility_state", "normal") or "normal").lower()
    hurst = float(features.get("hurst", 0.5))
    autocorr = float(features.get("autocorr", 0.0))
    entropy = float(features.get("entropy", 0.5))
    crowding = _mean_abs_correlation(features.get("correlation_row"))
    book_imbalance = float(features.get("book_imbalance", 0.0) or 0.0)
    book_wall_pressure = float(features.get("book_wall_pressure", 0.0) or 0.0)
    sentiment_market_cap_change = float(features.get("sentiment_market_cap_change_24h", 0.0) or 0.0)
    sentiment_symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    ema9_above_ema20 = bool(features.get("ema9_above_ema20", False))
    price_above_ema20 = bool(features.get("price_above_ema20", False))
    ema_slope_9 = float(features.get("ema_slope_9", 0.0) or 0.0)
    short_tf_ready_5m = bool(features.get("short_tf_ready_5m", False))
    short_tf_ready_15m = bool(features.get("short_tf_ready_15m", False))
    range_pos_1h = float(features.get("range_pos_1h", 0.5) or 0.5)
    range_pos_4h = float(features.get("range_pos_4h", 0.5) or 0.5)
    range_breakout_1h = bool(features.get("range_breakout_1h", False))
    higher_low_count = int(features.get("higher_low_count", 0) or 0)
    pivot_break = bool(features.get("pivot_break", False))
    pullback_hold = bool(features.get("pullback_hold", False))
    structure_build = bool(features.get("structure_build", False))
    atr_expanding = bool(features.get("atr_expanding", False))
    structure_quality = _profile_value(features, "structure_quality")
    momentum_quality = _profile_value(features, "momentum_quality")
    volume_quality = _profile_value(features, "volume_quality")
    trade_quality = _profile_value(features, "trade_quality")
    market_support = _profile_value(features, "market_support")
    continuation_quality = _profile_value(features, "continuation_quality")
    risk_quality = _profile_value(features, "risk_quality")
    coin_profile = features.get("coin_profile")
    structure_profile_present = bool(
        isinstance(coin_profile, dict)
        or "structure_quality" in features
        or "continuation_quality" in features
    )

    score = 48.0
    reasons: list[str] = []
    structure_confirmation_scale = _structure_confirmation_scale(
        structure_profile_present=structure_profile_present,
        structure_quality=structure_quality,
        continuation_quality=continuation_quality,
        ema9_above_ema20=ema9_above_ema20,
        price_above_ema20=price_above_ema20,
        range_breakout_1h=range_breakout_1h,
        pullback_hold=pullback_hold,
        higher_low_count=higher_low_count,
        pivot_break=pivot_break,
    )

    # CoinProfile now owns more of attractiveness. Raw features remain support and gating.
    score += _clamp((structure_quality - 50.0) * 0.12, -6.0, 6.0)
    score += _clamp((momentum_quality - 50.0) * 0.14, -7.0, 7.0)
    score += _clamp((volume_quality - 50.0) * 0.08, -4.0, 4.0)
    score += _clamp((trade_quality - 50.0) * 0.08, -4.0, 4.0)
    score += _clamp((market_support - 50.0) * 0.10, -5.0, 5.0)
    score += _clamp((continuation_quality - 50.0) * 0.12, -6.0, 6.0)
    score += _clamp((risk_quality - 50.0) * 0.10, -5.0, 5.0)

    # Raw momentum/volume stay as lighter confirmation to avoid double-counting.
    momentum_1 = float(features.get("momentum_1", 0.0))
    score += _clamp(momentum_5 * 400.0, -10.0, 12.0) * structure_confirmation_scale
    score += _clamp(momentum_1 * 400.0, -3.0, 4.0) * structure_confirmation_scale
    score += _clamp(momentum_14 * 100.0, -2.0, 2.0) * structure_confirmation_scale
    score += trend_1h * 4.0
    volume_ratio_log = float(np.log1p(max(volume_ratio, 0.0)))
    score += _clamp((volume_ratio_log - 0.693) * 8.0, -3.0, 6.0) * structure_confirmation_scale
    score += _clamp(volume_surge * 5.0, 0.0, 5.0) * structure_confirmation_scale
    if structure_confirmation_scale < 1.0:
        reasons.append("confirmation_capped_by_structure")

    # Microstructure remains mostly raw because it governs real execution quality.
    score -= _clamp(abs(price_zscore) * 3.0, 0.0, 10.0)
    score += _clamp(book_imbalance * 6.0, -4.0, 4.0)
    score += _clamp(book_wall_pressure * 3.0, -2.0, 2.0)
    score += _clamp(sentiment_market_cap_change * 0.4, -2.0, 2.0)

    if ema9_above_ema20 and price_above_ema20:
        score += 3.0
        reasons.append("ema_aligned")
    elif ema9_above_ema20 or price_above_ema20:
        score += 1.0
    if not ema9_above_ema20 and not price_above_ema20:
        score -= 2.0

    score += _clamp(ema_slope_9 * 500.0, -2.0, 4.0)

    if range_pos_1h >= 0.60:
        score += _clamp((range_pos_1h - 0.60) * 12.0, 0.0, 3.0)
        if range_pos_1h > 0.92:
            score -= 5.0
            reasons.append("range_exhausted_1h")
    elif range_pos_1h < 0.35:
        score -= 3.0

    if range_pos_4h < 0.85:
        score += 1.5
    elif range_pos_4h >= 0.92:
        score -= 3.0
        reasons.append("range_extended_4h")

    if range_breakout_1h:
        score += 4.0
        reasons.append("range_breakout_1h")

    trend_confirmed = bool(features.get("trend_confirmed", False))

    if volatility_state == "compressed":
        if range_breakout_1h or pullback_hold or structure_build:
            score += 2.5
            reasons.append("vol_compression_setup")
        else:
            score += 1.0
    elif volatility_state == "expanding":
        if trend_confirmed and continuation_quality >= 60.0:
            score += 2.0
            reasons.append("vol_expansion_confirmed")
        elif not trend_confirmed:
            score -= 1.5
            reasons.append("vol_expansion_unconfirmed")
    elif volatility_state == "overheated":
        score -= 3.0
        reasons.append("vol_overheated")
        if not pullback_hold:
            score -= 2.0
            reasons.append("late_vol_extension")

    if compression_score >= 72.0 and not atr_expanding:
        score += 1.0
    if expansion_score >= 75.0 and atr_expanding and volume_ratio >= 1.0:
        score += 1.0
    if volatility_percentile >= 90.0 and atr_pct >= 0.04 and not pullback_hold:
        score -= 2.0
        reasons.append("high_vol_chase")

    if higher_low_count >= 5:
        score += 3.0
        reasons.append("strong_higher_lows")
    elif higher_low_count >= 3:
        score += 1.5
        reasons.append("higher_lows")

    if pivot_break:
        score += 3.0
        reasons.append("pivot_break")

    if pullback_hold:
        score += 4.0
        reasons.append("pullback_hold")

    if lane == "L1":
        score += _clamp((structure_quality - 55.0) * 0.08, -2.0, 4.0)
        score += _clamp((continuation_quality - 55.0) * 0.10, -2.0, 5.0)
        if trend_1h > 0 and momentum_14 > 0.0:
            score += 4.0
            reasons.append("continuation_structure")
        if ema9_above_ema20 and trend_1h > 0:
            score += 2.0
        if momentum_5 < 0.001 and trend_1h <= 0:
            score -= 5.0
    elif lane == "L2":
        score += _clamp((momentum_quality - 52.0) * 0.12, -2.0, 6.0)
        score += _clamp((continuation_quality - 50.0) * 0.08, -2.0, 4.0)
        score += _clamp((market_support - 48.0) * 0.06, -2.0, 3.0)
        if volume_surge >= 0.2:
            score += 4.0 * structure_confirmation_scale
            reasons.append("rotation_release")
        if rotation_score >= 0.08:
            score += 5.0 * structure_confirmation_scale
            reasons.append("rotation_leader_score")
        elif rotation_score >= 0.05:
            score += 2.5 * structure_confirmation_scale
        if volume_ratio >= 0.8 and momentum_5 > 0.0:
            score += 1.5 * structure_confirmation_scale
        if regime_7d == "choppy":
            score -= 3.0
    elif lane == "L4":
        score += _clamp((momentum_quality - 50.0) * 0.12, -2.0, 6.0)
        score += _clamp((trade_quality - 50.0) * 0.08, -2.0, 4.0)
        score += _clamp(volume_surge * 7.0, 0.0, 7.0) * structure_confirmation_scale
        if sentiment_symbol_trending:
            score += 4.0 * structure_confirmation_scale
        if momentum_5 > 0.008 and volume_ratio >= 1.0:
            score += 4.0 * structure_confirmation_scale
            reasons.append("meme_heat")
        if book_imbalance >= 0.1:
            score += 2.0 * structure_confirmation_scale
            reasons.append("meme_bid_pressure")
    else:
        score += _clamp((trade_quality - 50.0) * 0.08, -2.0, 4.0)
        score += _clamp((risk_quality - 52.0) * 0.06, -2.0, 3.0)
        if trend_1h >= 0 and momentum_14 > 0.0 and momentum_30 > 0.0:
            score += 2.0 * structure_confirmation_scale
            reasons.append("balanced_alignment")
        if volume_ratio >= 0.9:
            score += 1.5 * structure_confirmation_scale

    if regime_7d == "trending":
        score -= 6.0  # top-of-move chasing — 22% win rate in historical data
        reasons.append("regime_trending_penalty")
    elif regime_7d == "choppy":
        score -= 6.0
        reasons.append("regime_choppy")

    if regime_state == "bullish":
        score -= 3.0  # chasing extended moves — 25% win rate in historical data
        reasons.append("regime_bullish_penalty")
    elif regime_state == "bearish":
        score -= 6.0
        reasons.append("regime_state_bearish")
    elif regime_state == "volatile":
        score -= 5.0
        reasons.append("regime_state_volatile")

    if macro_30d == "bull":
        score += 3.0
        reasons.append("macro_bull")
    elif macro_30d == "bear":
        score -= 6.0
        reasons.append("macro_bear")

    if hurst >= 0.58:
        score += 3.0
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

    if pullback_hold and reversal_risk == "HIGH":
        reversal_risk = "MEDIUM"

    late_chase = (
        price_zscore >= (2.3 if lane == "L4" else 1.8)
        and range_pos_1h >= 0.88
        and range_pos_4h >= 0.80
        and not pullback_hold
        and not range_breakout_1h
    )
    if late_chase:
        score -= 8.0 if lane in {"L1", "L2"} else 6.0
        reasons.append("late_chase_penalty")
        if reversal_risk == "LOW":
            reversal_risk = "MEDIUM"
        elif lane != "L4":
            reversal_risk = "HIGH"

    vwio = float(features.get("vwio", 0.0) or 0.0)
    if vwio >= 0.15:
        score += _clamp(vwio * 8.0, 0.0, 5.0)
        reasons.append("vwio_buy_pressure")
    elif vwio <= -0.15:
        score += _clamp(vwio * 8.0, -5.0, 0.0)
        reasons.append("vwio_sell_pressure")
        if reversal_risk == "LOW":
            reversal_risk = "MEDIUM"

    obv_divergence = int(features.get("obv_divergence", 0) or 0)
    if obv_divergence == -1:
        if reversal_risk == "LOW":
            reversal_risk = "MEDIUM"
            reasons.append("obv_bearish_div")
        elif reversal_risk == "MEDIUM":
            reversal_risk = "HIGH"
            reasons.append("obv_bearish_div_high")
    elif obv_divergence == 1:
        score += 4.0
        reasons.append("obv_bullish_div")
        if reversal_risk == "HIGH":
            reversal_risk = "MEDIUM"
            reasons.append("obv_risk_reduced")

    bullish_divergence = bool(features.get("bullish_divergence", False))
    bearish_divergence = bool(features.get("bearish_divergence", False))
    rsi_divergence = int(features.get("rsi_divergence", 0) or 0)
    if bullish_divergence and not bearish_divergence:
        rsi_divergence = 1
    elif bearish_divergence and not bullish_divergence:
        rsi_divergence = -1
    rsi_div_strength = float(features.get("divergence_strength", features.get("rsi_divergence_strength", 0.0)) or 0.0)
    rsi_div_age = int(features.get("divergence_age_bars", features.get("rsi_divergence_age", 99)) or 99)
    if rsi_divergence != 0:
        freshness = 1.0 if rsi_div_age <= 2 else 0.5
        base = 2.0 + min(4.0, rsi_div_strength * 4.0)
        adjustment = base * freshness
        if lane == "L4":
            adjustment *= 0.6
        elif lane == "L3":
            adjustment *= 1.1
        elif lane == "L1":
            adjustment *= 0.8
        adjustment = _clamp(adjustment, 0.0, 6.0)
        if rsi_divergence == 1:
            score += adjustment
            reasons.append("rsi_bullish_div")
            if reversal_risk == "HIGH":
                reversal_risk = "MEDIUM"
        else:
            score -= adjustment
            reasons.append("rsi_bearish_div")
            if reversal_risk == "LOW":
                reversal_risk = "MEDIUM"

    if not structure_profile_present:
        if trend_1h > 0 and momentum_14 > 0.0 and volume_ratio >= 1.0:
            score += 4.0
            reasons.append("sparse_structure_trend_support")
        if rotation_score >= 0.08 and momentum_5 > 0.0 and volume_ratio >= 1.25:
            score += 2.0
            reasons.append("sparse_structure_mover_support")
        if lane == "L2" and rotation_score >= 0.08 and volume_surge >= 0.2:
            score += 7.0
            reasons.append("sparse_structure_rotation_support")
        elif lane == "L4" and momentum_5 > 0.008 and volume_surge >= 0.5:
            score += 4.0
            reasons.append("sparse_structure_heat_support")

    score = _clamp(score, 0.0, 100.0)

    # --- Phase 1: Cost penalty — integrate execution costs into pre-score ---
    # Coins with terrible net edge get penalised early, before Nemo sees them.
    # Coins with strong net edge get a small bonus.
    _spread_pct = max(float(features.get("spread_pct", 0.0) or 0.0), 0.0)
    _price = float(features.get("price", 0.0) or 0.0)
    _atr = float(features.get("atr", 0.0) or 0.0)
    _atr_norm = (_atr / _price) if _price > 0.0 else 0.0
    _is_meme = lane == "L4"
    _maker_fee = float(get_runtime_setting("EXEC_MAKER_FEE_PCT"))
    _taker_fee = float(get_runtime_setting("EXEC_TAKER_FEE_PCT"))
    _slippage_mult = float(get_runtime_setting(
        "MEME_EXEC_SLIPPAGE_ATR_MULT" if _is_meme else "EXEC_SLIPPAGE_ATR_MULT"
    ))
    _slippage_pct = max(_atr_norm * _slippage_mult * 100.0, 0.02)
    _fee_rt = _maker_fee + max(_maker_fee, _taker_fee * 0.75)
    _total_cost_pct = _spread_pct + _fee_rt + _slippage_pct
    if lane == "L1":
        _tp_mult = float(get_runtime_setting("L1_EXIT_ATR_TAKE_PROFIT_MULT"))
    elif lane == "L2":
        _tp_mult = float(get_runtime_setting("L2_EXIT_ATR_TAKE_PROFIT_MULT"))
    elif _is_meme:
        _tp_mult = float(get_runtime_setting("MEME_EXIT_ATR_TAKE_PROFIT_MULT"))
    else:
        _tp_mult = float(get_runtime_setting("EXIT_ATR_TAKE_PROFIT_MULT"))
    _expected_move_pct = _atr_norm * _tp_mult * 100.0
    _quality_scale = max(min(
        (0.35 * structure_quality + 0.35 * continuation_quality
         + 0.15 * momentum_quality + 0.15 * trade_quality) / 100.0,
        1.2,
    ), 0.35)
    _expected_edge_pct = _expected_move_pct * _quality_scale
    _net_edge_pct = _expected_edge_pct - _total_cost_pct
    if _net_edge_pct >= 1.0:
        _cost_penalty = -5.0  # strong edge bonus
        reasons.append("edge_strong")
    elif _net_edge_pct >= 0.0:
        _cost_penalty = 0.0   # acceptable, no adjustment
    elif _net_edge_pct >= -0.5:
        _cost_penalty = 5.0   # marginal edge, light penalty
        reasons.append("cost_marginal")
    elif _net_edge_pct >= -1.0:
        _cost_penalty = 10.0  # bad edge, significant penalty
        reasons.append("cost_high")
    else:
        _cost_penalty = 15.0  # terrible edge, heavy penalty
        reasons.append("cost_prohibitive")
    score = _clamp(score - _cost_penalty, 0.0, 100.0)
    # --- End cost penalty ---

    strong_buy_threshold, buy_threshold = _lane_entry_thresholds(lane)
    probe_threshold = _lane_probe_threshold(lane)
    recommendation = "WATCH"
    if score >= strong_buy_threshold and reversal_risk == "LOW":
        recommendation = "STRONG_BUY"
    elif score >= buy_threshold and reversal_risk != "HIGH":
        recommendation = "BUY"
    elif score <= 35.0 and reversal_risk != "HIGH":
        recommendation = "MEDIUM"
    elif reversal_risk == "HIGH":
        recommendation = "MEDIUM"

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
        if lane_filter_reason.endswith("_vol_low") and recommendation == "BUY":
            recommendation = "WATCH"
        if reversal_risk == "LOW":
            reversal_risk = "MEDIUM"

    indicators_ready = bool(features.get("indicators_ready", True))
    lane_filter_severity = str(features.get("lane_filter_severity", "ok") or "ok").lower()
    short_tf_ready = bool(short_tf_ready_5m or short_tf_ready_15m)
    descending_short_structure = (
        momentum_5 < 0.0
        and momentum_14 <= 0.0
        and higher_low_count < 2
        and not short_tf_ready
        and not pullback_hold
        and not range_breakout_1h
        and ((not ema9_above_ema20) or ema_slope_9 < 0.0 or trend_1h < 0)
    )
    mover_signal = rotation_score > 0.0 or momentum_5 > 0.0 or volume_surge >= 0.2 or trend_confirmed
    leader_signal = (
        bool(features.get("leader_takeover", False))
        or float(features.get("leader_urgency", 0.0) or 0.0) >= 6.0
    )
    promotion_ready = (
        short_tf_ready
        or trend_confirmed
        or pullback_hold
        or range_breakout_1h
        or (ema9_above_ema20 and price_above_ema20 and ema_slope_9 >= 0.0)
    )
    trigger_quality_ready = (
        promotion_ready
        and (momentum_5 > 0.0 or volume_surge >= 0.2 or range_breakout_1h or pullback_hold)
        and (trade_quality >= 54.0 or continuation_quality >= 58.0 or structure_quality >= 60.0)
    )

    promotion_tier = "skip"
    promotion_reason = "not_qualified"
    if not indicators_ready:
        promotion_reason = "feature_warmup"
    elif descending_short_structure:
        recommendation = "AVOID"
        reversal_risk = "HIGH"
        promotion_reason = "falling_short_structure"
        reasons.append("falling_short_structure")
    elif lane_filter_severity == "hard":
        promotion_reason = lane_filter_reason or "hard_filter"
    elif recommendation == "STRONG_BUY" and reversal_risk == "LOW" and trigger_quality_ready and not late_chase:
        promotion_tier = "promote"
        promotion_reason = "strong_buy_low_risk"
    elif recommendation == "BUY" and reversal_risk != "HIGH" and trigger_quality_ready and not late_chase:
        promotion_tier = "promote"
        promotion_reason = "buy_candidate"
    elif (
        recommendation in {"BUY", "WATCH"}
        and reversal_risk != "HIGH"
        and (range_breakout_1h or pullback_hold)
        and ema9_above_ema20
        and trend_1h >= 0
        and score >= probe_threshold
    ):
        promotion_tier = "promote" if recommendation == "BUY" else "probe"
        promotion_reason = "channel_breakout" if range_breakout_1h else "channel_retest"
    elif (
        recommendation == "WATCH"
        and reversal_risk != "HIGH"
        and score >= (probe_threshold + 2.0 if lane_filter_pass else probe_threshold - 2.0)
        and mover_signal
        and short_tf_ready
        and (volume_ratio >= 0.9 or volume_surge >= 0.25 or leader_signal)
        and (trade_quality >= 55.0 or continuation_quality >= 60.0)
        and not late_chase
        and (lane in {"L2", "L4"} or leader_signal)
    ):
        promotion_tier = "probe"
        promotion_reason = "watch_mover_probe"
    elif recommendation == "WATCH":
        promotion_reason = "watch_not_promoted"
    elif recommendation == "AVOID":
        promotion_reason = "avoid_recommendation"
    else:
        promotion_reason = "insufficient_score"

    return {
        "entry_score": round(score, 2),
        "entry_recommendation": recommendation,
        "reversal_risk": reversal_risk,
        "entry_reasons": list(dict.fromkeys(reasons)),
        "promotion_tier": promotion_tier,
        "promotion_reason": promotion_reason,
        "point_breakdown": {
            "spread_pct": round(_spread_pct, 4),
            "total_cost_pct": round(_total_cost_pct, 3),
            "expected_edge_pct": round(_expected_edge_pct, 3),
            "net_edge_pct": round(_net_edge_pct, 3),
            "cost_penalty_pts": round(_cost_penalty, 1),
        },
    }
