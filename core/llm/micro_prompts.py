from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.llm.client import nemotron_chat, parse_json_response, phi3_chat, sanitize_for_json
from core.llm.prompts import (
    NEMOTRON_SET_POSTURE_SYSTEM_PROMPT,
    NEMOTRON_REVIEW_CANDIDATE_SYSTEM_PROMPT,
    NEMOTRON_REVIEW_OUTCOME_SYSTEM_PROMPT,
    PHI3_EXIT_POSTURE_SYSTEM_PROMPT,
    PHI3_REVIEW_MARKET_STATE_SYSTEM_PROMPT,
    PHI3_SUPERVISE_LANE_SYSTEM_PROMPT,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_decision": self.promotion_decision,
            "priority": self.priority,
            "action_bias": self.action_bias,
            "reason": self.reason,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_state": self.market_state,
            "confidence": self.confidence,
            "lane_bias": self.lane_bias,
            "reason": self.reason,
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


def phi3_supervise_lane(
    candidate: dict[str, Any],
    *,
    news_context: dict[str, Any] | None = None,
    dex_context: dict[str, Any] | None = None,
) -> Phi3LaneAdvice:
    payload = {
        "candidate": sanitize_for_json(candidate),
        "news_context": sanitize_for_json(news_context or {}),
        "dex_context": sanitize_for_json(dex_context or {}),
    }
    try:
        parsed = parse_json_response(
            phi3_chat(payload, system=PHI3_SUPERVISE_LANE_SYSTEM_PROMPT, max_tokens=220)
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
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    if ranging_market:
        return Phi3MarketStateReview("ranging", 0.8, "reduce_trend_entries", "range_signals_detected")
    if trend_confirmed and (momentum_5 > 0.0 or rotation_score > 0.0):
        return Phi3MarketStateReview("trending", 0.75, "favor_trend", "trend_confirmation_present")
    return Phi3MarketStateReview("transition", 0.6, "favor_selective", "mixed_market_structure")


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
        parsed = parse_json_response(
            phi3_chat(payload, system=PHI3_REVIEW_MARKET_STATE_SYSTEM_PROMPT, max_tokens=180)
        )
        return Phi3MarketStateReview(
            market_state=_safe_market_state(parsed.get("market_state")),
            confidence=_clamp_confidence(parsed.get("confidence"), 0.5),
            lane_bias=_safe_lane_bias(parsed.get("lane_bias")),
            reason=str(parsed.get("reason", "market_state_review")),
        )
    except Exception:
        return _heuristic_market_state_review(features)


def _heuristic_exit_posture_review(payload: dict[str, Any]) -> Phi3ExitPostureReview:
    pnl_pct = float(payload.get("pnl_pct", 0.0) or 0.0)
    hold_minutes = float(payload.get("hold_minutes", 0.0) or 0.0)
    momentum = float(payload.get("momentum", payload.get("momentum_5", 0.0)) or 0.0)
    trend_1h = float(payload.get("trend_1h", 0.0) or 0.0)
    rsi = float(payload.get("rsi", 50.0) or 50.0)
    if hold_minutes >= 180.0 and abs(pnl_pct) <= 1.5 and abs(momentum) < 0.0015:
        return Phi3ExitPostureReview("STALE", 0.82, "time_stop_no_progress")
    if pnl_pct <= -2.0 and momentum < 0.0 and trend_1h <= 0.0:
        return Phi3ExitPostureReview("EXIT", 0.84, "loser_with_trend_decay")
    if pnl_pct >= 1.0 and (momentum < 0.0 or rsi >= 72.0):
        return Phi3ExitPostureReview("TIGHTEN", 0.74, "protect_open_profit")
    if pnl_pct > 0.0 and trend_1h > 0.0 and momentum >= 0.0:
        return Phi3ExitPostureReview("RUN", 0.72, "trend_intact_let_run")
    return Phi3ExitPostureReview("RUN", 0.58, "default_hold_state")


def phi3_review_exit_posture(position_payload: dict[str, Any]) -> Phi3ExitPostureReview:
    payload = {"position": sanitize_for_json(position_payload)}
    try:
        parsed = parse_json_response(
            phi3_chat(payload, system=PHI3_EXIT_POSTURE_SYSTEM_PROMPT, max_tokens=180)
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
    # Apply symbol performance overrides before normal heuristics
    if symbol_performance:
        verdict = str(symbol_performance.get("verdict", "neutral"))
        if verdict == "avoid":
            return NemotronCandidateReview("demote", 0.1, "hold_preferred", "poor_track_record")
        if verdict == "weak":
            return NemotronCandidateReview("neutral", 0.4, "hold_preferred", "weak_track_record")
    if entry_recommendation == "STRONG_BUY" and reversal_risk == "LOW":
        return NemotronCandidateReview("promote", 0.9, "open_allowed", "strong_buy_low_risk")
    if entry_recommendation == "BUY" and reversal_risk in {"LOW", "MEDIUM"}:
        return NemotronCandidateReview("promote", 0.75, "open_allowed", "buy_signal_supported")
    if entry_recommendation == "WATCH" and reversal_risk == "LOW" and (rotation_score > 0 or momentum_5 > 0):
        return NemotronCandidateReview("neutral", 0.6, "reduce_size", "watch_low_risk_supported")
    if (
        entry_recommendation == "WATCH"
        and reversal_risk == "MEDIUM"
        and entry_score >= 60.0
        and rotation_score > 0.0
        and momentum_5 > 0.0
    ):
        return NemotronCandidateReview("neutral", 0.55, "reduce_size", "watch_medium_qualified")
    return NemotronCandidateReview("demote", 0.35, "hold_preferred", "weak_or_risky_setup")


def _compact_advisory_features(features: dict[str, Any]) -> dict[str, Any]:
    return sanitize_for_json(
        {
            "symbol": features.get("symbol"),
            "lane": features.get("lane"),
            "entry_score": features.get("entry_score"),
            "entry_recommendation": features.get("entry_recommendation"),
            "reversal_risk": features.get("reversal_risk"),
            "entry_reasons": features.get("entry_reasons"),
            "rotation_score": features.get("rotation_score"),
            "momentum_5": features.get("momentum_5"),
            "momentum_14": features.get("momentum_14"),
            "momentum_30": features.get("momentum_30"),
            "trend_confirmed": features.get("trend_confirmed"),
            "ranging_market": features.get("ranging_market"),
            "trend_1h": features.get("trend_1h"),
            "regime_7d": features.get("regime_7d"),
            "macro_30d": features.get("macro_30d"),
            "regime_state": features.get("regime_state"),
            "volume_ratio": features.get("volume_ratio"),
            "volume_surge": features.get("volume_surge"),
            "volume_surge_flag": features.get("volume_surge_flag"),
            "rsi": features.get("rsi"),
            "price_zscore": features.get("price_zscore"),
            "atr": features.get("atr"),
            "price": features.get("price"),
            "book_imbalance": features.get("book_imbalance"),
            "book_wall_pressure": features.get("book_wall_pressure"),
            "sentiment_fng_value": features.get("sentiment_fng_value"),
            "sentiment_fng_label": features.get("sentiment_fng_label"),
            "sentiment_btc_dominance": features.get("sentiment_btc_dominance"),
            "sentiment_market_cap_change_24h": features.get("sentiment_market_cap_change_24h"),
            "sentiment_symbol_trending": features.get("sentiment_symbol_trending"),
            "lane_filter_pass": features.get("lane_filter_pass"),
            "lane_filter_reason": features.get("lane_filter_reason"),
            "lane_filter_severity": features.get("lane_filter_severity"),
            "lane_conflict": features.get("lane_conflict"),
        }
    )


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
        "features": _compact_advisory_features(features),
        "phi3_advisory": sanitize_for_json(phi3_advisory or {}),
        "universe_context": _compact_universe_context(universe_context or {}),
        "visual_context": sanitize_for_json(visual_context or {}),
    }
    try:
        parsed = parse_json_response(
            nemotron_chat(payload, system=NEMOTRON_REVIEW_CANDIDATE_SYSTEM_PROMPT, max_tokens=220)
        )
        decision = str(parsed.get("promotion_decision", "neutral")).strip().lower()
        if decision not in {"promote", "neutral", "demote"}:
            decision = "neutral"
        action_bias = str(parsed.get("action_bias", "hold_preferred")).strip().lower()
        if action_bias not in {"open_allowed", "hold_preferred", "reduce_size"}:
            action_bias = "hold_preferred"
        return NemotronCandidateReview(
            promotion_decision=decision,
            priority=_clamp_confidence(parsed.get("priority"), 0.5),
            action_bias=action_bias,
            reason=str(parsed.get("reason", "candidate_review")),
        )
    except Exception:
        return _heuristic_candidate_review(features, symbol_performance)


def _heuristic_posture_review(
    features: dict[str, Any],
    market_state: dict[str, Any] | None,
    candidate_review: dict[str, Any] | None,
) -> NemotronPostureReview:
    state = _safe_market_state((market_state or {}).get("market_state"))
    promotion_decision = str((candidate_review or {}).get("promotion_decision", "neutral")).strip().lower()
    if state == "ranging":
        return NemotronPostureReview("defensive", "tighter", "tighten", "reduce", "range_state_defensive_posture")
    if state == "trending" and promotion_decision == "promote":
        return NemotronPostureReview("aggressive", "wider", "let_run", "increase", "trend_state_promoted_candidate")
    return NemotronPostureReview("neutral", "normal", "standard", "normal", "balanced_posture")


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
        "features": _compact_advisory_features(features),
        "market_state": sanitize_for_json(market_state or {}),
        "candidate_review": sanitize_for_json(candidate_review or {}),
        "visual_context": sanitize_for_json(visual_context or {}),
    }
    try:
        parsed = parse_json_response(
            nemotron_chat(payload, system=NEMOTRON_SET_POSTURE_SYSTEM_PROMPT, max_tokens=180)
        )
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
        parsed = parse_json_response(
            nemotron_chat(payload, system=NEMOTRON_REVIEW_OUTCOME_SYSTEM_PROMPT, max_tokens=220)
        )
        return NemotronOutcomeReview(
            outcome_class=str(parsed.get("outcome_class", "normal_exit")),
            lesson=str(parsed.get("lesson", "no_lesson")),
            suggested_adjustment=str(parsed.get("suggested_adjustment", "no_change")),
            confidence=_clamp_confidence(parsed.get("confidence"), 0.5),
        )
    except Exception:
        return _heuristic_outcome_review(outcome)
