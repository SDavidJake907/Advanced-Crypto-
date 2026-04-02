from __future__ import annotations

from typing import Any

import pandas as pd

from core.features.pattern_log_schema import PatternEvidence, SwingPoint


def pct_diff(a: float, b: float) -> float:
    if abs(b) <= 1e-12:
        return 0.0
    return abs(a - b) / abs(b)


def find_local_swings(highs: list[float], lows: list[float], lookback: int = 2) -> list[SwingPoint]:
    swings: list[SwingPoint] = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        hi = highs[i]
        lo = lows[i]
        if all(hi >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i):
            swings.append(SwingPoint(index=i, price=float(hi), kind="high"))
        if all(lo <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i):
            swings.append(SwingPoint(index=i, price=float(lo), kind="low"))
    swings.sort(key=lambda item: item.index)
    return swings


def _volume_confirmation(volumes: list[float], window: int = 10) -> float:
    if not volumes:
        return 0.0
    lookback = volumes[-window:] if len(volumes) >= window else volumes
    avg = sum(lookback) / max(len(lookback), 1)
    if avg <= 1e-12:
        return 0.0
    return min(float(volumes[-1]) / avg, 2.0) / 2.0


def detect_double_bottom(
    *,
    symbol: str,
    timeframe: str,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    tolerance: float = 0.015,
) -> PatternEvidence | None:
    swings = find_local_swings(highs, lows, lookback=2)
    low_swings = [item for item in swings if item.kind == "low"]
    high_swings = [item for item in swings if item.kind == "high"]
    if len(low_swings) < 2 or len(high_swings) < 1:
        return None

    left_low, right_low = low_swings[-2], low_swings[-1]
    if right_low.index <= left_low.index + 2:
        return None

    mid_highs = [item for item in high_swings if left_low.index < item.index < right_low.index]
    if not mid_highs:
        return None
    neckline = max(mid_highs, key=lambda item: item.price)
    if pct_diff(left_low.price, right_low.price) > tolerance:
        return None

    last_close = float(closes[-1])
    breakout_confirmed = last_close > neckline.price
    denominator = max(neckline.price - right_low.price, 1e-9)
    breakout_score = 1.0 if breakout_confirmed else max(0.0, (last_close - right_low.price) / denominator)
    symmetry_score = 1.0 - min(pct_diff(left_low.price, right_low.price) / tolerance, 1.0)
    volume_score = _volume_confirmation(volumes)
    height = neckline.price - min(left_low.price, right_low.price)
    confidence = (0.35 * symmetry_score) + (0.30 * breakout_score) + (0.20 * volume_score) + 0.15

    return PatternEvidence(
        pattern="double_bottom",
        bias="bullish",
        timeframe=timeframe,
        symbol=symbol,
        confidence_raw=round(confidence, 4),
        symmetry_score=round(symmetry_score, 4),
        breakout_score=round(breakout_score, 4),
        volume_confirmation_score=round(volume_score, 4),
        neckline_level=round(neckline.price, 8),
        support_level=round(min(left_low.price, right_low.price), 8),
        breakout_level=round(neckline.price, 8),
        stop_level_hint=round(min(left_low.price, right_low.price) * 0.995, 8),
        target_level_hint=round(neckline.price + height, 8),
        breakout_confirmed=breakout_confirmed,
        swing_points=[left_low, neckline, right_low],
        notes=[
            "Two comparable swing lows detected.",
            "Neckline derived from highest swing high between lows.",
        ],
    )


def detect_double_top(
    *,
    symbol: str,
    timeframe: str,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    tolerance: float = 0.015,
) -> PatternEvidence | None:
    swings = find_local_swings(highs, lows, lookback=2)
    high_swings = [item for item in swings if item.kind == "high"]
    low_swings = [item for item in swings if item.kind == "low"]
    if len(high_swings) < 2 or len(low_swings) < 1:
        return None

    left_high, right_high = high_swings[-2], high_swings[-1]
    if right_high.index <= left_high.index + 2:
        return None

    mid_lows = [item for item in low_swings if left_high.index < item.index < right_high.index]
    if not mid_lows:
        return None
    neckline = min(mid_lows, key=lambda item: item.price)
    if pct_diff(left_high.price, right_high.price) > tolerance:
        return None

    last_close = float(closes[-1])
    breakout_confirmed = last_close < neckline.price
    denominator = max(right_high.price - neckline.price, 1e-9)
    breakout_score = 1.0 if breakout_confirmed else max(0.0, (right_high.price - last_close) / denominator)
    symmetry_score = 1.0 - min(pct_diff(left_high.price, right_high.price) / tolerance, 1.0)
    volume_score = _volume_confirmation(volumes)
    height = max(left_high.price, right_high.price) - neckline.price
    confidence = (0.35 * symmetry_score) + (0.30 * breakout_score) + (0.20 * volume_score) + 0.15

    return PatternEvidence(
        pattern="double_top",
        bias="bearish",
        timeframe=timeframe,
        symbol=symbol,
        confidence_raw=round(confidence, 4),
        symmetry_score=round(symmetry_score, 4),
        breakout_score=round(breakout_score, 4),
        volume_confirmation_score=round(volume_score, 4),
        neckline_level=round(neckline.price, 8),
        resistance_level=round(max(left_high.price, right_high.price), 8),
        breakout_level=round(neckline.price, 8),
        stop_level_hint=round(max(left_high.price, right_high.price) * 1.005, 8),
        target_level_hint=round(neckline.price - height, 8),
        breakout_confirmed=breakout_confirmed,
        swing_points=[left_high, neckline, right_high],
        notes=[
            "Two comparable swing highs detected.",
            "Neckline derived from lowest swing low between highs.",
        ],
    )


def detect_top_pattern_from_frame(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if frame is None or frame.empty:
        return None
    required = {"close", "high", "low", "volume"}
    if not required.issubset(frame.columns):
        return None

    tail = frame.tail(80).copy()
    if "open" not in tail.columns:
        synthetic_open = tail["close"].shift(1).fillna(tail["close"])
        tail["open"] = synthetic_open.astype(float)
    closes = tail["close"].astype(float).tolist()
    highs = tail["high"].astype(float).tolist()
    lows = tail["low"].astype(float).tolist()
    volumes = tail["volume"].astype(float).tolist()
    if len(closes) < 12:
        return None

    candidates: list[PatternEvidence] = []
    for detector in (detect_double_bottom, detect_double_top):
        evidence = detector(
            symbol=symbol,
            timeframe=timeframe,
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )
        if evidence is not None:
            candidates.append(evidence)
    if not candidates:
        return None
    best = max(candidates, key=lambda item: float(item.confidence_raw))
    return build_pattern_setup_bundle(
        symbol=symbol,
        timeframe=timeframe,
        frame=tail,
        structure=best,
    )


def _body_size(open_price: float, close_price: float) -> float:
    return abs(close_price - open_price)


def _candle_range(high_price: float, low_price: float) -> float:
    return max(high_price - low_price, 1e-9)


def _detect_recent_candle_patterns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if len(frame) < 2 or "open" not in frame.columns:
        return []
    rows = frame.reset_index(drop=True)
    last = rows.iloc[-1]
    prev = rows.iloc[-2]
    recent: list[dict[str, Any]] = []

    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_close = float(last["close"])
    prev_open = float(prev["open"])
    prev_close = float(prev["close"])

    body = _body_size(last_open, last_close)
    rng = _candle_range(last_high, last_low)
    upper_wick = last_high - max(last_open, last_close)
    lower_wick = min(last_open, last_close) - last_low

    if prev_close < prev_open and last_close > last_open and last_close >= prev_open and last_open <= prev_close:
        recent.append({"name": "bullish_engulfing", "bias": "bullish", "strength": round(min(body / rng + 0.35, 1.0), 4), "confirmed": True})
    if prev_close > prev_open and last_close < last_open and last_open >= prev_close and last_close <= prev_open:
        recent.append({"name": "bearish_engulfing", "bias": "bearish", "strength": round(min(body / rng + 0.35, 1.0), 4), "confirmed": True})
    if lower_wick >= body * 2.0 and upper_wick <= body and last_close > last_open:
        recent.append({"name": "hammer", "bias": "bullish", "strength": round(min(lower_wick / rng + 0.2, 1.0), 4), "confirmed": True})
    if upper_wick >= body * 2.0 and lower_wick <= body and last_close < last_open:
        recent.append({"name": "shooting_star", "bias": "bearish", "strength": round(min(upper_wick / rng + 0.2, 1.0), 4), "confirmed": True})
    if body / rng <= 0.15:
        recent.append({"name": "doji", "bias": "neutral", "strength": round(1.0 - min(body / rng, 1.0), 4), "confirmed": True})
    if max(last_open, last_close) <= float(prev["high"]) and min(last_open, last_close) >= float(prev["low"]):
        recent.append({"name": "inside_bar", "bias": "neutral", "strength": 0.55, "confirmed": True})

    midpoint = (last_high + last_low) / 2.0
    if lower_wick >= body * 1.8 and last_close >= midpoint:
        recent.append({"name": "pin_bar", "bias": "bullish", "strength": round(min(lower_wick / rng + 0.15, 1.0), 4), "confirmed": True})
    elif upper_wick >= body * 1.8 and last_close <= midpoint:
        recent.append({"name": "pin_bar", "bias": "bearish", "strength": round(min(upper_wick / rng + 0.15, 1.0), 4), "confirmed": True})

    return recent[:4]


def _dominant_candle_bias(candles: list[dict[str, Any]]) -> str:
    bullish = sum(float(item.get("strength", 0.0) or 0.0) for item in candles if item.get("bias") == "bullish")
    bearish = sum(float(item.get("strength", 0.0) or 0.0) for item in candles if item.get("bias") == "bearish")
    if bullish > bearish and bullish > 0.0:
        return "bullish"
    if bearish > bullish and bearish > 0.0:
        return "bearish"
    return "neutral"


def _build_location_context(frame: pd.DataFrame, structure: PatternEvidence) -> dict[str, Any]:
    close = float(frame["close"].iloc[-1])
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    full_high = float(highs.max())
    full_low = float(lows.min())
    range_span = max(full_high - full_low, 1e-9)
    range_position = (close - full_low) / range_span
    near_neckline = False
    if structure.neckline_level is not None:
        near_neckline = abs(close - float(structure.neckline_level)) / max(abs(float(structure.neckline_level)), 1e-9) <= 0.01
    at_support = structure.support_level is not None and abs(close - float(structure.support_level)) / max(abs(float(structure.support_level)), 1e-9) <= 0.012
    at_resistance = structure.resistance_level is not None and abs(close - float(structure.resistance_level)) / max(abs(float(structure.resistance_level)), 1e-9) <= 0.012
    return {
        "at_support": bool(at_support),
        "at_resistance": bool(at_resistance),
        "near_neckline": bool(near_neckline),
        "near_range_edge": bool(range_position <= 0.15 or range_position >= 0.85),
        "higher_timeframe_alignment": True,
        "overextended_from_entry_zone": bool(range_position >= 0.9 if structure.bias == "bullish" else range_position <= 0.1),
    }


def _build_conflict_checks(
    *,
    structure: PatternEvidence,
    candle_context: dict[str, Any],
    location_context: dict[str, Any],
) -> dict[str, Any]:
    dominant_bias = str(candle_context.get("dominant_candle_bias", "neutral"))
    structure_vs_candles_conflict = dominant_bias not in {"neutral", structure.bias}
    volume_missing = float(structure.volume_confirmation_score) < 0.35
    messy_structure = float(structure.symmetry_score) < 0.45
    late_entry = bool(location_context.get("overextended_from_entry_zone", False))
    return {
        "structure_vs_candles_conflict": structure_vs_candles_conflict,
        "trend_vs_pattern_conflict": False,
        "volume_confirmation_missing": volume_missing,
        "late_entry_risk": late_entry,
        "messy_structure_flag": messy_structure,
    }


def _compute_extension_risk_score(
    *,
    structure: PatternEvidence,
    distance_to_breakout_pct: float,
    location_context: dict[str, Any],
) -> float:
    breakout_confirmed = bool(structure.breakout_confirmed)
    overextended = bool(location_context.get("overextended_from_entry_zone", False))
    if not breakout_confirmed:
        return 0.0
    distance_risk = min(max(distance_to_breakout_pct, 0.0) / 1.5, 1.0)
    extension_risk = 0.70 * distance_risk
    if overextended:
        extension_risk += 0.30
    return round(min(max(extension_risk, 0.0), 1.0), 4)


def build_pattern_setup_bundle(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
    structure: PatternEvidence,
) -> dict[str, Any]:
    candles = _detect_recent_candle_patterns(frame)
    structure_location = "mid_range"
    if structure.neckline_level is not None:
        if structure.breakout_confirmed and structure.retest_holding:
            structure_location = "near_neckline_retest"
        elif structure.breakout_confirmed:
            structure_location = "post_breakout"
        else:
            structure_location = "pre_breakout"
    for candle in candles:
        candle["location"] = structure_location
    dominant_bias = _dominant_candle_bias(candles)
    candle_confidence = 0.0
    if candles:
        candle_confidence = sum(float(item.get("strength", 0.0) or 0.0) for item in candles) / len(candles)
    location_context = _build_location_context(frame, structure)
    conflict_checks = _build_conflict_checks(
        structure=structure,
        candle_context={
            "recent_candles": candles,
            "dominant_candle_bias": dominant_bias,
            "candle_confidence_score": round(candle_confidence, 4),
        },
        location_context=location_context,
    )
    close = float(frame["close"].iloc[-1])
    distance_to_breakout_pct = 0.0
    if structure.breakout_level is not None:
        distance_to_breakout_pct = abs(close - float(structure.breakout_level)) / max(abs(float(structure.breakout_level)), 1e-9) * 100.0
    extension_risk_score = _compute_extension_risk_score(
        structure=structure,
        distance_to_breakout_pct=distance_to_breakout_pct,
        location_context=location_context,
    )
    structure_dict = structure.to_dict()
    structure_dict["pattern_quality_score"] = round(
        (float(structure.confidence_raw) + float(structure.symmetry_score) + float(structure.breakout_score)) / 3.0,
        4,
    )
    structure_dict["breakout_strength_score"] = structure_dict.pop("breakout_score", 0.0)
    structure_dict["retest_quality_score"] = round(
        (float(structure.retest_score) + (1.0 if structure.retest_holding else 0.0)) / 2.0,
        4,
    )
    structure_dict["bars_in_pattern"] = len(frame)
    structure_dict["distance_to_breakout_pct"] = round(distance_to_breakout_pct, 4)
    structure_dict["extension_risk_score"] = extension_risk_score
    notes = list(structure.notes)
    if extension_risk_score >= 0.6:
        notes.append("Late breakout risk: entry is extended from the ideal breakout zone.")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "structure_candidate": structure_dict,
        "candle_context": {
            "recent_candles": candles,
            "dominant_candle_bias": dominant_bias,
            "candle_confidence_score": round(candle_confidence, 4),
        },
        "location_context": location_context,
        "conflict_checks": conflict_checks,
        "notes": notes,
        "pattern": structure.pattern,
        "bias": structure.bias,
        "confidence_raw": float(structure.confidence_raw),
    }
