from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.llm.client import phi3_advisory_chat, nemotron_chat, parse_json_response, sanitize_for_json
from core.llm.contracts import (
    normalize_candidate_review,
    normalize_market_state_review,
    normalize_outcome_review,
    normalize_posture_review,
)
from core.llm.prompts import (
    PHI3_EXIT_POSTURE_SYSTEM_PROMPT,
    PHI3_REVIEW_MARKET_STATE_SYSTEM_PROMPT,
    PHI3_SUPERVISE_LANE_SYSTEM_PROMPT,
    get_nemotron_review_candidate_system_prompt,
    get_nemotron_review_outcome_system_prompt,
    get_nemotron_set_posture_system_prompt,
)


PUBLIC_TO_INTERNAL_LANE = {
    "main": "L3",
    "meme": "L4",
    "reversion": "L2",
    "breakout": "L1",
}
INTERNAL_TO_PUBLIC_LANE = {value: key for key, value in PUBLIC_TO_INTERNAL_LANE.items()}


@dataclass
class Phi3LaneAdvice:
    lane_candidate: str
    lane_confidence: float
    lane_conflict: bool
    narrative_tag: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_candidate": self.lane_candidate,
            "lane_confidence": self.lane_confidence,
            "lane_conflict": self.lane_conflict,
            "narrative_tag": self.narrative_tag,
            "reason": self.reason,
        }


@dataclass
class NemotronCandidateReview:
    promotion_decision: str
    priority: float
    action_bias: str
    reason: str
    market_posture: str = "mixed"
    hold_bias_nemo: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_decision": self.promotion_decision,
            "priority": self.priority,
            "action_bias": self.action_bias,
            "reason": self.reason,
            "market_posture": self.market_posture,
            "hold_bias_nemo": self.hold_bias_nemo,
        }


@dataclass
class NemotronOutcomeReview:
    outcome_class: str
    lesson: str
    suggested_adjustment: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_class": self.outcome_class,
            "lesson": self.lesson,
            "suggested_adjustment": self.suggested_adjustment,
            "confidence": self.confidence,
        }


@dataclass
class Phi3MarketStateReview:
    market_state: str
    confidence: float
    lane_bias: str
    reason: str
    breakout_state: str = "unclear"
    trend_stage: str = "unclear"
    volume_confirmation: str = "neutral"
    pullback_quality: str = "unclear"
    late_move_risk: str = "moderate"
    pattern_explanation: dict[str, Any] | None = None
    candle_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_state": self.market_state,
            "confidence": self.confidence,
            "lane_bias": self.lane_bias,
            "reason": self.reason,
            "breakout_state": self.breakout_state,
            "trend_stage": self.trend_stage,
            "volume_confirmation": self.volume_confirmation,
            "pullback_quality": self.pullback_quality,
            "late_move_risk": self.late_move_risk,
            "pattern_explanation": self.pattern_explanation or {},
            "candle_evidence": self.candle_evidence or {},
        }


@dataclass
class Phi3ExitPostureReview:
    posture: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "posture": self.posture,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class NemotronPostureReview:
    posture: str
    promotion_bias: str
    exit_bias: str
    size_bias: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "posture": self.posture,
            "promotion_bias": self.promotion_bias,
            "exit_bias": self.exit_bias,
            "size_bias": self.size_bias,
            "reason": self.reason,
        }


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _safe_public_lane(value: Any, default: str = "main") -> str:
    lane = str(value or default).strip().lower()
    if lane not in PUBLIC_TO_INTERNAL_LANE:
        return default
    return lane


def _safe_market_state(value: Any, default: str = "transition") -> str:
    state = str(value or default).strip().lower()
    if state not in {"trending", "ranging", "transition"}:
        return default
    return state


def _safe_lane_bias(value: Any, default: str = "favor_selective") -> str:
    lane_bias = str(value or default).strip().lower()
    if lane_bias not in {"favor_trend", "favor_selective", "reduce_trend_entries"}:
        return default
    return lane_bias


def _safe_posture(value: Any, default: str = "neutral") -> str:
    posture = str(value or default).strip().lower()
    if posture not in {"aggressive", "neutral", "defensive"}:
        return default
    return posture


def _safe_posture_bias(value: Any, allowed: set[str], default: str) -> str:
    bias = str(value or default).strip().lower()
    if bias not in allowed:
        return default
    return bias


def _safe_exit_posture(value: Any, default: str = "RUN") -> str:
    posture = str(value or default).strip().upper()
    if posture not in {"RUN", "TIGHTEN", "EXIT", "STALE"}:
        return default
    return posture


def _heuristic_lane_advice(candidate: dict[str, Any]) -> Phi3LaneAdvice:
    current_lane = INTERNAL_TO_PUBLIC_LANE.get(str(candidate.get("lane", "L3")), "main")
    momentum_5 = float(candidate.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(candidate.get("momentum_14", 0.0) or 0.0)
    rotation_score = float(candidate.get("rotation_score", 0.0) or 0.0)
    volume = float(candidate.get("volume", 0.0) or 0.0)
    rsi = float(candidate.get("rsi", 50.0) or 50.0)
    lane_candidate = current_lane
    reason = "heuristic_lane_supervision"
    tag = "balanced_setup"
    if current_lane == "meme" or (momentum_5 > 0.01 and rotation_score > 0 and rsi >= 65):
        lane_candidate = "meme"
        tag = "early_acceleration"
    elif momentum_14 > 0 and rotation_score > 0.05:
        lane_candidate = "breakout"
        tag = "trend_continuation"
    elif momentum_5 < 0 and momentum_14 > 0:
        lane_candidate = "reversion"
        tag = "bounce_candidate"
    confidence = 0.55
    if rotation_score > 0.1:
        confidence += 0.15
    if volume > 0 and momentum_5 > 0:
        confidence += 0.1
    confidence = min(max(confidence, 0.0), 1.0)
    return Phi3LaneAdvice(
        lane_candidate=lane_candidate,
        lane_confidence=confidence,
        lane_conflict=lane_candidate != current_lane,
        narrative_tag=tag,
        reason=reason,
    )


def _load_candles_for_phi3(symbol: str, timeframe: str, n_bars: int) -> list[list[float]]:
    """Load recent OHLCV bars as compact [o,h,l,c,v] arrays for Phi-3 chart reading."""
    base = symbol.replace("/", "")
    live_dir = Path(os.getenv("LIVE_SNAPSHOT_DIR", "logs/live"))
    path = live_dir / f"candles_{base}_{timeframe}.csv"
    if not path.exists():
        return []
    try:
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([
                    round(float(row["open"]), 5),
                    round(float(row["high"]), 5),
                    round(float(row["low"]), 5),
                    round(float(row["close"]), 5),
                    round(float(row["volume"]), 1),
                ])
        return rows[-n_bars:]
    except Exception:
        return []


# Only the fields Phi-3 needs for lane supervision — keeps payload within NPU context limit.
_PHI3_LANE_CANDIDATE_FIELDS = (
    "symbol", "lane", "entry_score", "rsi", "momentum_5",
    "volume_ratio", "recommendation", "risk",
)


def phi3_supervise_lane(
    candidate: dict[str, Any],
    *,
    news_context: dict[str, Any] | None = None,
    dex_context: dict[str, Any] | None = None,
) -> Phi3LaneAdvice:
    symbol = str(candidate.get("symbol", ""))
    # Slim candidate to essential fields only.
    # Phi-3 mini NPU has ~2560 char total context (system prompt + payload).
    # Full candidate dict is ~550 chars; slimmed is ~180 chars.
    slim_candidate = {k: candidate[k] for k in _PHI3_LANE_CANDIDATE_FIELDS if k in candidate}
    # 15m is the primary trading timeframe; 8 bars = last 2h of price action.
    payload = {
        "candidate": sanitize_for_json(slim_candidate),
        "chart_15m": _load_candles_for_phi3(symbol, "15m", 8),
    }
    try:
        parsed = parse_json_response(
            phi3_advisory_chat(payload, system=PHI3_SUPERVISE_LANE_SYSTEM_PROMPT, max_tokens=400)
        )
        return Phi3LaneAdvice(
            lane_candidate=_safe_public_lane(parsed.get("lane_candidate")),
            lane_confidence=_clamp_confidence(parsed.get("lane_confidence"), 0.5),
            lane_conflict=bool(parsed.get("lane_conflict", False)),
            narrative_tag=str(parsed.get("narrative_tag", "balanced_setup")),
            reason=str(parsed.get("reason", "phi3_lane_supervision")),
        )
    except Exception:
        return _heuristic_lane_advice(candidate)


def _heuristic_market_state_review(features: dict[str, Any]) -> Phi3MarketStateReview:
    ranging_market = bool(features.get("ranging_market", False))
    trend_confirmed = bool(features.get("trend_confirmed", False))
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    macd_hist = float(features.get("macd_hist", 0.0) or 0.0)
    ema9_above_ema20 = bool(features.get("ema9_above_ema20", False))
    ema9_above_ema26 = bool(features.get("ema9_above_ema26", False))
    range_breakout = bool(features.get("range_breakout_1h", False))
    pullback_hold = bool(features.get("pullback_hold", False))
    higher_low_count = int(features.get("higher_low_count", 0) or 0)
    symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    ema_stack_bullish = ema9_above_ema20 and ema9_above_ema26
    mover_present = (
        momentum_5 > 0.0
        or rotation_score > 0.0
        or volume_surge >= 0.35
        or volume_ratio >= 1.2
        or symbol_trending
    )
    if range_breakout and volume_ratio >= 1.2 and momentum_5 > 0.0:
        breakout_state = "fresh_breakout"
    elif pullback_hold and ema_stack_bullish:
        breakout_state = "retest_holding"
    elif range_breakout or momentum_5 > 0.0:
        breakout_state = "breakout_attempt"
    else:
        breakout_state = "inside_range"
    if trend_confirmed and ema_stack_bullish and macd_hist >= 0.0:
        if higher_low_count >= 3 and momentum_14 > 0.0:
            trend_stage = "confirmed"
        else:
            trend_stage = "emerging"
    elif ema_stack_bullish and momentum_5 > 0.0:
        trend_stage = "early"
    elif macd_hist < 0.0 and momentum_5 < 0.0:
        trend_stage = "stalling"
    else:
        trend_stage = "mixed"
    if volume_ratio >= 1.25 or volume_surge >= 0.35:
        volume_confirmation = "supportive"
    elif volume_ratio <= 0.7:
        volume_confirmation = "weak"
    else:
        volume_confirmation = "neutral"
    if pullback_hold and ema_stack_bullish:
        pullback_quality = "clean_retest"
    elif pullback_hold:
        pullback_quality = "loose_retest"
    elif higher_low_count >= 2 and ema_stack_bullish:
        pullback_quality = "higher_low_support"
    else:
        pullback_quality = "none"
    if entry_score >= 80.0 and momentum_5 > 0.015 and volume_surge >= 0.35:
        late_move_risk = "extended"
    elif trend_confirmed and higher_low_count >= 2:
        late_move_risk = "contained"
    else:
        late_move_risk = "moderate"
    primary_candle = "none"
    candle_bias = "neutral"
    location_context = "mid_range"
    if pullback_hold:
        location_context = "at_support"
    elif range_breakout:
        location_context = "at_breakout_level"
    elif trend_confirmed:
        location_context = "aligned_with_trend"
    bullish_engulfing = momentum_5 > 0.0 and macd_hist > 0.0 and ema_stack_bullish and volume_ratio >= 1.1
    bearish_engulfing = momentum_5 < 0.0 and macd_hist < 0.0 and not ema_stack_bullish
    hammer_like = pullback_hold and momentum_5 > 0.0 and macd_hist >= 0.0
    shooting_star_like = late_move_risk == "extended" and macd_hist < 0.0
    inside_bar_like = abs(momentum_5) <= 0.002 and volume_ratio <= 1.05
    outside_bar_like = volume_surge >= 0.35 and abs(momentum_5) > 0.008
    doji_like = abs(momentum_5) <= 0.001 and abs(macd_hist) <= 0.00005
    if bullish_engulfing:
        primary_candle = "bullish_engulfing"
        candle_bias = "bullish"
    elif hammer_like:
        primary_candle = "hammer"
        candle_bias = "bullish"
    elif shooting_star_like:
        primary_candle = "shooting_star"
        candle_bias = "bearish"
    elif bearish_engulfing:
        primary_candle = "bearish_engulfing"
        candle_bias = "bearish"
    elif outside_bar_like:
        primary_candle = "outside_bar"
        candle_bias = "bullish" if momentum_5 > 0.0 else "bearish"
    elif inside_bar_like:
        primary_candle = "inside_bar"
        candle_bias = "neutral"
    elif doji_like:
        primary_candle = "doji"
        candle_bias = "neutral"
    candle_strength = round(min(max(
        0.3
        + (0.18 if candle_bias == "bullish" and momentum_5 > 0.0 else 0.0)
        + (0.18 if candle_bias == "bearish" and momentum_5 < 0.0 else 0.0)
        + (0.12 if macd_hist >= 0.0 and candle_bias != "bearish" else 0.0)
        + (0.10 if volume_confirmation == "supportive" else -0.05 if volume_confirmation == "weak" else 0.0),
        0.0,
    ), 0.95), 2)
    confirmation_score = round(min(max(
        (
            candle_strength
            + (0.12 if location_context in {"at_support", "at_breakout_level", "aligned_with_trend"} else 0.0)
            + (0.12 if volume_confirmation == "supportive" else 0.0)
            + (0.10 if trend_confirmed else 0.0)
        ),
        0.0,
    ), 0.95), 2)
    candle_warnings: list[str] = []
    if primary_candle in {"inside_bar", "doji"}:
        candle_warnings.append("Primary candle is more indecisive than directional.")
    if volume_confirmation == "weak":
        candle_warnings.append("Candle confirmation lacks volume support.")
    if late_move_risk == "extended" and candle_bias == "bullish":
        candle_warnings.append("Bullish candle appears late in the move.")
    candle_summary = (
        "Bullish candle confirmation supports the structure."
        if candle_bias == "bullish" and confirmation_score >= 0.65
        else "Bearish candle warning weakens the structure."
        if candle_bias == "bearish" and confirmation_score >= 0.55
        else "Candle evidence is present but not decisive."
    )
    structure_phase = "unclear"
    if range_breakout and trend_confirmed and volume_confirmation == "supportive":
        structure_phase = "confirmed_breakout"
    elif pullback_hold and ema_stack_bullish:
        structure_phase = "retest_holding"
    elif trend_confirmed and ema_stack_bullish and momentum_5 > 0.0:
        structure_phase = "trend_expansion"
    elif macd_hist < 0.0 and momentum_5 < 0.0:
        structure_phase = "breakout_failure"
    elif momentum_5 > 0.0 or rotation_score > 0.0:
        structure_phase = "attempting_breakout"
    symmetry_score = min(max(0.35 + (0.08 * min(higher_low_count, 4)), 0.0), 0.95)
    breakout_quality_score = min(max(
        0.35
        + (0.18 if range_breakout else 0.0)
        + (0.14 if trend_confirmed else 0.0)
        + (0.08 if macd_hist >= 0.0 else -0.06),
        0.0,
    ), 0.95)
    retest_quality_score = min(max(
        0.3
        + (0.28 if pullback_hold else 0.0)
        + (0.12 if ema_stack_bullish else 0.0),
        0.0,
    ), 0.95)
    location_quality_score = min(max(
        0.35
        + (0.16 if trend_confirmed else 0.0)
        + (0.12 if ranging_market and mover_present else 0.0)
        + (0.08 if higher_low_count >= 2 else 0.0),
        0.0,
    ), 0.95)
    candle_confirmation_score = min(max(
        0.35
        + (0.18 if momentum_5 > 0.0 else -0.05)
        + (0.12 if macd_hist >= 0.0 else -0.08)
        + (0.10 if ema_stack_bullish else 0.0),
        0.0,
    ), 0.95)
    structure_confidence = round(min(max(
        (
            breakout_quality_score
            + retest_quality_score
            + location_quality_score
            + candle_confirmation_score
        ) / 4.0,
        0.0,
    ), 0.95), 2)
    warnings: list[str] = []
    missing_confirmation: list[str] = []
    if volume_confirmation != "supportive":
        warnings.append("Volume confirmation is not strong.")
    if late_move_risk == "extended":
        warnings.append("Entry is getting extended from base structure.")
    if macd_hist < 0.0 and momentum_5 < 0.0:
        warnings.append("Momentum and structure are weakening together.")
    if breakout_state in {"fresh_breakout", "breakout_attempt"} and not range_breakout:
        missing_confirmation.append("No clear breakout close yet.")
    if breakout_state in {"retest_holding", "breakout_attempt"} and not pullback_hold:
        missing_confirmation.append("No clean retest confirmation yet.")
    summary = (
        "Breakout is confirmed with supportive follow-through."
        if structure_phase == "confirmed_breakout"
        else "Price is holding the retest with improving structure."
        if structure_phase == "retest_holding"
        else "Momentum is improving, but the structure is still developing."
        if structure_phase in {"attempting_breakout", "trend_expansion"}
        else "Structure is mixed and currently weakening."
    )
    ema_stack_state = "bullish_stack" if ema_stack_bullish else "mixed_stack"
    ema_slope_state = "rising" if higher_low_count >= 2 and ema_stack_bullish else "flat_to_mixed" if momentum_5 >= 0.0 else "falling"
    price_vs_ema_state = (
        "holding_above_short_ema"
        if pullback_hold or range_breakout or trend_confirmed
        else "testing_short_ema"
        if momentum_5 >= 0.0
        else "below_short_ema_context"
    )
    macd_state = "bullish_expansion" if macd_hist > 0.0 and trend_confirmed else "bullish_improving" if macd_hist > 0.0 else "bearish_fading"
    momentum_state = "positive" if momentum_5 > 0.0 else "flat" if abs(momentum_5) <= 0.001 else "negative"
    acceleration_state = "building" if momentum_5 > 0.0 and momentum_14 > 0.0 else "stalling" if momentum_5 <= 0.0 < momentum_14 else "fading"
    support_context = "higher_low_support" if higher_low_count >= 2 else "ema_support" if pullback_hold else "weak_support"
    resistance_context = "breaking_range_high" if range_breakout else "below_range_high" if breakout_state == "breakout_attempt" else "unclear"
    breakout_level_state = "cleared" if range_breakout else "pressing" if breakout_state == "breakout_attempt" else "not_cleared"
    pattern_explanation = {
        "structure_phase": structure_phase,
        "structure_confidence": structure_confidence,
        "pattern_quality": {
            "symmetry_score": round(symmetry_score, 2),
            "breakout_quality_score": round(breakout_quality_score, 2),
            "retest_quality_score": round(retest_quality_score, 2),
            "location_quality_score": round(location_quality_score, 2),
            "candle_confirmation_score": round(candle_confirmation_score, 2),
        },
        "moving_average_context": {
            "ema_stack_state": ema_stack_state,
            "ema_slope_state": ema_slope_state,
            "price_vs_ema_state": price_vs_ema_state,
        },
        "momentum_context": {
            "macd_state": macd_state,
            "momentum_state": momentum_state,
            "acceleration_state": acceleration_state,
        },
        "key_levels": {
            "support": None,
            "neckline": None,
            "breakout_level": None,
            "stop_hint": None,
            "target_hint": None,
        },
        "context": {
            "at_support": bool(pullback_hold or higher_low_count >= 2),
            "near_neckline": bool(range_breakout),
            "higher_timeframe_alignment": bool(trend_confirmed),
            "volume_confirmation": volume_confirmation == "supportive",
            "liquidity_grab_detected": False,
            "support_context": support_context,
            "resistance_context": resistance_context,
            "breakout_level_state": breakout_level_state,
        },
        "candle_evidence": {
            "primary_candle": primary_candle,
            "candle_bias": candle_bias,
            "candle_strength": candle_strength,
            "location_context": location_context,
            "volume_confirmation": volume_confirmation == "supportive",
            "confirmation_score": confirmation_score,
            "warning_flags": candle_warnings,
            "summary": candle_summary,
        },
        "warnings": warnings,
        "missing_confirmation": missing_confirmation,
        "summary": summary,
    }
    candle_evidence = dict(pattern_explanation["candle_evidence"])
    if ranging_market:
        if entry_score >= 55.0 and mover_present:
            if ema_stack_bullish and macd_hist >= 0.0 and momentum_5 > 0.0:
                return Phi3MarketStateReview("ranging", 0.62, "favor_selective", "range_breakout_attempt_with_ema_support", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
            if volume_surge >= 0.35 or volume_ratio >= 1.2:
                return Phi3MarketStateReview("ranging", 0.6, "favor_selective", "range_breakout_attempt_with_volume_support", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
            if rotation_score > 0.0 or symbol_trending:
                return Phi3MarketStateReview("ranging", 0.58, "favor_selective", "selective_range_mover_with_relative_strength", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
            return Phi3MarketStateReview("ranging", 0.56, "favor_selective", "range_state_but_mover_present", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
        if not ema_stack_bullish and macd_hist < 0.0:
            return Phi3MarketStateReview("ranging", 0.68, "favor_selective", "range_state_without_trend_support", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
        return Phi3MarketStateReview("ranging", 0.65, "favor_selective", "range_signals_detected", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
    if trend_confirmed and (momentum_5 > 0.0 or rotation_score > 0.0):
        if ema_stack_bullish and macd_hist >= 0.0:
            return Phi3MarketStateReview("trending", 0.78, "favor_trend", "trend_confirmation_with_ema_and_macd", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
        if volume_surge >= 0.35 or volume_ratio >= 1.2:
            return Phi3MarketStateReview("trending", 0.75, "favor_trend", "trend_confirmation_with_volume_support", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
        return Phi3MarketStateReview("trending", 0.72, "favor_trend", "trend_confirmation_present", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
    if ema_stack_bullish and macd_hist >= 0.0 and momentum_5 > 0.0:
        return Phi3MarketStateReview("transition", 0.58, "favor_selective", "breakout_attempt_not_yet_confirmed", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
    if macd_hist < 0.0 and momentum_5 < 0.0:
        return Phi3MarketStateReview("transition", 0.62, "reduce_trend_entries", "momentum_fading_into_transition", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)
    return Phi3MarketStateReview("transition", 0.6, "favor_selective", "mixed_market_structure", breakout_state, trend_stage, volume_confirmation, pullback_quality, late_move_risk, pattern_explanation, candle_evidence)


def deterministic_market_state_review(features: dict[str, Any]) -> Phi3MarketStateReview:
    return _heuristic_market_state_review(features)


def phi3_review_market_state(
    features: dict[str, Any],
    *,
    visual_context: dict[str, Any] | None = None,
) -> Phi3MarketStateReview:
    payload = {
        "features": sanitize_for_json(features),
        "visual_context": sanitize_for_json(visual_context or {}),
    }
    try:
        parsed = normalize_market_state_review(parse_json_response(
            phi3_advisory_chat(payload, system=PHI3_REVIEW_MARKET_STATE_SYSTEM_PROMPT, max_tokens=400)
        ))
        return Phi3MarketStateReview(
            market_state=_safe_market_state(parsed.get("market_state")),
            confidence=_clamp_confidence(parsed.get("confidence"), 0.5),
            lane_bias=_safe_lane_bias(parsed.get("lane_bias")),
            reason=str(parsed.get("reason", "market_state_review")),
            breakout_state=str(parsed.get("breakout_state", "unclear") or "unclear"),
            trend_stage=str(parsed.get("trend_stage", "unclear") or "unclear"),
            volume_confirmation=str(parsed.get("volume_confirmation", "neutral") or "neutral"),
            pullback_quality=str(parsed.get("pullback_quality", "unclear") or "unclear"),
            late_move_risk=str(parsed.get("late_move_risk", "moderate") or "moderate"),
            pattern_explanation=parsed.get("pattern_explanation", {}) if isinstance(parsed.get("pattern_explanation", {}), dict) else {},
            candle_evidence=parsed.get("candle_evidence", {}) if isinstance(parsed.get("candle_evidence", {}), dict) else {},
        )
    except Exception:
        return _heuristic_market_state_review(features)


def _heuristic_exit_posture_review(payload: dict[str, Any]) -> Phi3ExitPostureReview:
    pnl_pct = float(payload.get("pnl_pct", 0.0) or 0.0)
    hold_minutes = float(payload.get("hold_minutes", 0.0) or 0.0)
    momentum = float(payload.get("momentum", payload.get("momentum_5", 0.0)) or 0.0)
    trend_1h = float(payload.get("trend_1h", 0.0) or 0.0)
    rsi = float(payload.get("rsi", 50.0) or 50.0)
    # Never STALE or TIGHTEN on new positions — let them breathe
    if hold_minutes < 60.0:
        return Phi3ExitPostureReview("RUN", 0.85, "new_position_let_run")
    # Only EXIT on severe loss with confirmed trend breakdown
    if pnl_pct <= -4.0 and momentum < 0.0 and trend_1h <= 0.0:
        return Phi3ExitPostureReview("EXIT", 0.84, "loser_with_trend_decay")
    # TIGHTEN only on substantial profit with strong reversal signals
    if pnl_pct >= 12.0 and momentum < -0.003 and rsi >= 78.0:
        return Phi3ExitPostureReview("TIGHTEN", 0.74, "protect_open_profit")
    if pnl_pct > 0.0 and trend_1h > 0.0 and momentum >= 0.0:
        return Phi3ExitPostureReview("RUN", 0.80, "trend_intact_let_run")
    return Phi3ExitPostureReview("RUN", 0.70, "default_hold_state")


def phi3_review_exit_posture(position_payload: dict[str, Any]) -> Phi3ExitPostureReview:
    payload = {"position": sanitize_for_json(position_payload)}
    try:
        parsed = parse_json_response(
            phi3_advisory_chat(payload, system=PHI3_EXIT_POSTURE_SYSTEM_PROMPT, max_tokens=400)
        )
        return Phi3ExitPostureReview(
            posture=_safe_exit_posture(parsed.get("posture")),
            confidence=_clamp_confidence(parsed.get("confidence"), 0.6),
            reason=str(parsed.get("reason", "phi3_exit_posture")),
        )
    except Exception:
        return _heuristic_exit_posture_review(position_payload)


def _heuristic_candidate_review(
    features: dict[str, Any],
    symbol_performance: dict[str, Any] | None = None,
) -> NemotronCandidateReview:
    entry_recommendation = str(features.get("entry_recommendation", "WATCH")).upper()
    reversal_risk = str(features.get("reversal_risk", "MEDIUM")).upper()
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    xgb_score = float(features.get("xgb_score", 0.0) or 0.0)
    short_tf_ready_5m = bool(features.get("short_tf_ready_5m"))
    short_tf_ready_15m = bool(features.get("short_tf_ready_15m"))
    symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    leader_urgency = float(features.get("leader_urgency", 0.0) or 0.0)
    leader_takeover = bool(features.get("leader_takeover"))
    # Apply symbol performance overrides before normal heuristics
    if symbol_performance:
        verdict = str(symbol_performance.get("verdict", "neutral"))
        if verdict == "avoid":
            return NemotronCandidateReview("demote", 0.1, "hold_preferred", "poor_track_record")
        if verdict == "weak":
            return NemotronCandidateReview("neutral", 0.4, "hold_preferred", "weak_track_record")
    leader_override = leader_takeover or leader_urgency >= 4.5
    short_tf_ready = short_tf_ready_5m or short_tf_ready_15m
    mover_signal = (
        momentum_5 > 0.0
        or rotation_score > 0.1
        or volume_surge >= 0.35
        or xgb_score >= 60.0
        or symbol_trending
    )
    strong_score = entry_score >= 50.0 and reversal_risk != "HIGH"
    if leader_override:
        return NemotronCandidateReview("promote", 0.85, "reduce_size", "leader_urgency_override")
    if entry_recommendation in {"BUY", "STRONG_BUY"} and strong_score and (mover_signal or short_tf_ready):
        return NemotronCandidateReview("promote", 0.72, "reduce_size", "buy_mover_probe")
    if entry_recommendation == "WATCH" and reversal_risk != "HIGH" and (short_tf_ready or mover_signal):
        return NemotronCandidateReview("promote", 0.62, "reduce_size", "watch_mover_probe")
    if entry_score >= 62.0 and rotation_score > 0.05 and momentum_5 > 0.0:
        return NemotronCandidateReview("neutral", 0.6, "reduce_size", "watch_medium_qualified")
    return NemotronCandidateReview("neutral", 0.45, "hold_preferred", "soft_or_waiting")


def _build_nemo_observation_buckets(features: dict[str, Any]) -> dict[str, Any]:
    """Build clean 4-bucket observation payload for Nemo candidate review."""
    coin_profile = features.get("coin_profile", {}) if isinstance(features.get("coin_profile", {}), dict) else {}
    # --- Setup state ---
    ema_ok = bool(features.get("ema9_above_ema20"))
    ema_cross_ok = bool(features.get("ema9_above_ema26"))
    ema_cross_distance_pct = float(features.get("ema_cross_distance_pct", 0.0) or 0.0)
    higher_low = int(features.get("higher_low_count", 0) or 0)
    if ema_ok and higher_low >= 3:
        structure_quality = "strong"
    elif ema_ok or higher_low >= 1:
        structure_quality = "medium"
    else:
        structure_quality = "weak"

    rec = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
    reversal = str(features.get("reversal_risk", "MEDIUM") or "MEDIUM").upper()
    if rec in {"STRONG_BUY", "BUY"} and reversal == "LOW":
        trigger_quality = "confirmed"
    elif rec in {"STRONG_BUY", "BUY"} or (rec == "WATCH" and reversal != "HIGH"):
        trigger_quality = "borderline"
    else:
        trigger_quality = "unconfirmed"

    # --- Market state ---
    raw_regime_7d = features.get("regime_7d", 0.0)
    try:
        regime_7d = float(raw_regime_7d or 0.0)
    except Exception:
        regime_label = str(raw_regime_7d or "").strip().lower()
        if regime_label in {"bull", "bullish", "trending"}:
            regime_7d = 1.0
        elif regime_label in {"bear", "bearish"}:
            regime_7d = -1.0
        else:
            regime_7d = 0.0
    trend_1h = float(features.get("trend_1h", 0.0) or 0.0)
    if regime_7d > 0.5 and trend_1h > 0:
        market_regime = "bullish"
    elif regime_7d < -0.5 or trend_1h < -0.5:
        market_regime = "bearish"
    else:
        market_regime = "sideways"

    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    if volume_ratio >= 1.5 and momentum_14 > 0:
        breadth = "strong"
    elif volume_ratio < 0.7 or momentum_14 < -0.005:
        breadth = "weak"
    else:
        breadth = "mixed"

    book_imbalance = float(features.get("book_imbalance", 0.0) or 0.0)
    book_wall = str(features.get("book_wall_pressure", "") or "")
    if book_imbalance > 0.1:
        book_pressure = "supportive"
    elif book_imbalance < -0.1 or "ask" in book_wall.lower():
        book_pressure = "hostile"
    else:
        book_pressure = "neutral"

    # --- Volatility context ---
    hurst = float(features.get("hurst", 0.5) or 0.5)
    bb_bandwidth = float(features.get("bb_bandwidth", 0.0) or 0.0)
    # Hurst > 0.55 → trending (momentum reliable), < 0.45 → mean-reverting (fade moves)
    if hurst >= 0.60:
        hurst_regime = "trending"
    elif hurst <= 0.42:
        hurst_regime = "mean_reverting"
    else:
        hurst_regime = "random"
    # BB bandwidth: low = squeeze/consolidation (breakout potential), high = expanding volatility
    if bb_bandwidth < 0.02:
        squeeze_state = "tight_squeeze"
    elif bb_bandwidth < 0.06:
        squeeze_state = "normal"
    else:
        squeeze_state = "expanding"

    # --- Execution state ---
    spread_pct = float(features.get("spread_pct", 0.0) or 0.0)
    if spread_pct < 0.15:
        spread_quality = "clean"
    elif spread_pct < 0.5:
        spread_quality = "thin"
    else:
        spread_quality = "bad"

    return sanitize_for_json({
        "setup_state": {
            "lane": features.get("lane"),
            "setup_type": features.get("promotion_reason") or "balanced_entry",
            "structure_quality": structure_quality,
            "trigger_quality": trigger_quality,
            "ema_cross_bullish": ema_cross_ok,
            "ema_cross_distance_pct": round(ema_cross_distance_pct, 4),
            "pullback_hold": features.get("pullback_hold"),
            "range_breakout": features.get("range_breakout_1h"),
            "leader_urgency": features.get("leader_urgency"),
        },
        "market_state": {
            "market_regime": market_regime,
            "breadth": breadth,
            "book_pressure": book_pressure,
            "ranging_market": features.get("ranging_market"),
            "trend_confirmed": features.get("trend_confirmed"),
            "fng_value": features.get("sentiment_fng_value"),
            "fng_label": features.get("sentiment_fng_label"),
        },
        "volatility_context": {
            "hurst": round(hurst, 3),
            "hurst_regime": hurst_regime,      # trending / random / mean_reverting
            "bb_bandwidth": round(bb_bandwidth, 4),
            "squeeze_state": squeeze_state,    # tight_squeeze / normal / expanding
        },
        "execution_state": {
            "spread_quality": spread_quality,
            "spread_pct": spread_pct,
            "volume_surge": features.get("volume_surge_flag"),
        },
        "ema_cross": {
            "ema_9": features.get("ema_9"),
            "ema_20": features.get("ema_20"),
            "ema_26": features.get("ema_26"),
            "ema9_above_ema20": features.get("ema9_above_ema20"),
            "ema9_above_ema26": features.get("ema9_above_ema26"),
            "ema_cross_distance_pct": features.get("ema_cross_distance_pct"),
        },
        "symbol_context": {
            "symbol": features.get("symbol"),
            "entry_score": features.get("entry_score"),
            "momentum_5": features.get("momentum_5"),
            "rsi": features.get("rsi"),
        },
        "coin_profile": {
            "structure_quality": coin_profile.get("structure_quality", features.get("structure_quality")),
            "momentum_quality": coin_profile.get("momentum_quality", features.get("momentum_quality")),
            "volume_quality": coin_profile.get("volume_quality", features.get("volume_quality")),
            "trade_quality": coin_profile.get("trade_quality", features.get("trade_quality")),
            "market_support": coin_profile.get("market_support", features.get("market_support")),
            "continuation_quality": coin_profile.get("continuation_quality", features.get("continuation_quality")),
            "risk_quality": coin_profile.get("risk_quality", features.get("risk_quality")),
        },
    })


def _compact_universe_context(universe_context: dict[str, Any]) -> dict[str, Any]:
    context = universe_context or {}
    return sanitize_for_json(
        {
            "current_symbol_is_top_candidate": context.get("current_symbol_is_top_candidate"),
            "top_ranked_symbols": [
                item if isinstance(item, str) else item.get("symbol")
                for item in (context.get("top_ranked") or [])[:8]
            ],
            "hot_candidate_symbols": [
                item.get("symbol")
                for item in (context.get("hot_candidates") or [])[:8]
                if isinstance(item, dict)
            ],
        }
    )


def nemotron_review_candidate(
    *,
    symbol: str,
    features: dict[str, Any],
    phi3_advisory: dict[str, Any] | None = None,
    universe_context: dict[str, Any] | None = None,
    visual_context: dict[str, Any] | None = None,
    symbol_performance: dict[str, Any] | None = None,
) -> NemotronCandidateReview:
    # Apply symbol performance pre-filter before LLM call
    if symbol_performance:
        verdict = str(symbol_performance.get("verdict", "neutral"))
        if verdict == "avoid":
            return NemotronCandidateReview("demote", 0.1, "hold_preferred", "poor_track_record")
        if verdict == "weak":
            return NemotronCandidateReview("neutral", 0.4, "hold_preferred", "weak_track_record")
    payload = {
        "symbol": symbol,
        "observation": _build_nemo_observation_buckets(features),
        "phi3_advisory": sanitize_for_json(phi3_advisory or {}),
        "universe_context": _compact_universe_context(universe_context or {}),
    }
    try:
        parsed = normalize_candidate_review(parse_json_response(
            nemotron_chat(payload, system=get_nemotron_review_candidate_system_prompt(), max_tokens=400)
        ))
        return NemotronCandidateReview(
            promotion_decision=str(parsed.get("promotion_decision", "neutral")),
            priority=_clamp_confidence(parsed.get("priority"), 0.5),
            action_bias=str(parsed.get("action_bias", "hold_preferred")),
            reason=str(parsed.get("reason", "candidate_review")),
            market_posture=str(parsed.get("market_posture", "mixed")),
            hold_bias_nemo=str(parsed.get("hold_bias_nemo", "normal")),
        )
    except Exception:
        return _heuristic_candidate_review(features, symbol_performance)


def _heuristic_posture_review(
    features: dict[str, Any],
    market_state: dict[str, Any] | None,
    candidate_review: dict[str, Any] | None,
) -> NemotronPostureReview:
    cr = candidate_review or {}
    # Use new schema fields if available (from redesigned Nemo observer)
    market_posture = str(cr.get("market_posture", "mixed")).strip().lower()
    hold_bias = str(cr.get("hold_bias_nemo", "normal")).strip().lower()
    promotion_decision = str(cr.get("promotion_decision", "neutral")).strip().lower()

    _POSTURE_MAP = {"supportive": "aggressive", "mixed": "neutral", "hostile": "defensive"}
    _EXIT_MAP = {"patient": "let_run", "normal": "standard", "fast": "tighten"}

    posture = _POSTURE_MAP.get(market_posture, "neutral")
    exit_bias = _EXIT_MAP.get(hold_bias, "standard")

    if promotion_decision == "promote":
        return NemotronPostureReview(posture, "wider" if posture == "aggressive" else "normal", exit_bias, "normal" if posture != "aggressive" else "increase", "heuristic_from_candidate_review")
    if promotion_decision == "demote":
        return NemotronPostureReview("defensive", "tighter", "tighten", "reduce", "heuristic_demote")
    return NemotronPostureReview(posture, "normal", exit_bias, "normal", "heuristic_balanced")


def nemotron_set_posture(
    *,
    symbol: str,
    features: dict[str, Any],
    market_state: dict[str, Any] | None = None,
    candidate_review: dict[str, Any] | None = None,
    visual_context: dict[str, Any] | None = None,
) -> NemotronPostureReview:
    payload = {
        "symbol": symbol,
        "lane": str(features.get("lane", "L3") or "L3").upper(),
        "structure_quality": _build_nemo_observation_buckets(features).get("setup_state", {}).get("structure_quality", "medium"),
        "market_state": sanitize_for_json(market_state or {}),
        "candidate_review": sanitize_for_json(candidate_review or {}),
    }
    try:
        parsed = normalize_posture_review(parse_json_response(
            nemotron_chat(payload, system=get_nemotron_set_posture_system_prompt(), max_tokens=500)
        ))
        return NemotronPostureReview(
            posture=_safe_posture(parsed.get("posture")),
            promotion_bias=_safe_posture_bias(parsed.get("promotion_bias"), {"wider", "normal", "tighter"}, "normal"),
            exit_bias=_safe_posture_bias(parsed.get("exit_bias"), {"let_run", "standard", "tighten"}, "standard"),
            size_bias=_safe_posture_bias(parsed.get("size_bias"), {"increase", "normal", "reduce"}, "normal"),
            reason=str(parsed.get("reason", "posture_review")),
        )
    except Exception:
        return _heuristic_posture_review(features, market_state, candidate_review)


def _heuristic_outcome_review(outcome: dict[str, Any]) -> NemotronOutcomeReview:
    pnl_pct = float(outcome.get("pnl_pct", 0.0) or 0.0)
    exit_reason = str(outcome.get("exit_reason", "unknown"))
    if pnl_pct > 0.02:
        return NemotronOutcomeReview("good_breakout", "winner_held_well", "keep_current_posture", 0.75)
    if "stop" in exit_reason:
        return NemotronOutcomeReview("chop_fakeout", "stopped_out_early", "review_entry_timing", 0.7)
    if pnl_pct < 0:
        return NemotronOutcomeReview("weak_follow_through", "entry_lacked_follow_through", "tighten_promotion", 0.65)
    return NemotronOutcomeReview("normal_exit", "neutral_outcome", "no_change", 0.5)


def nemotron_review_outcome(outcome: dict[str, Any]) -> NemotronOutcomeReview:
    payload = {"outcome": sanitize_for_json(outcome)}
    try:
        parsed = normalize_outcome_review(parse_json_response(
            nemotron_chat(payload, system=get_nemotron_review_outcome_system_prompt(), max_tokens=400)
        ))
        return NemotronOutcomeReview(
            outcome_class=str(parsed.get("outcome_class", "normal_exit")),
            lesson=str(parsed.get("lesson", "no_lesson")),
            suggested_adjustment=str(parsed.get("suggested_adjustment", "no_change")),
            confidence=_clamp_confidence(parsed.get("confidence"), 0.5),
        )
    except Exception:
        return _heuristic_outcome_review(outcome)
