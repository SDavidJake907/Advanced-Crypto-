from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.llm.client import nemotron_provider_model, nemotron_provider_name


ROLE_TRADE_REVIEWER = "trade_reviewer"
ROLE_RUNTIME_ADVISOR = "runtime_advisor"
ROLE_OUTCOME_REVIEWER = "outcome_reviewer"
ROLE_DRIFT_DETECTOR = "drift_detector"
ROLE_CANDIDATE_REVIEWER = "candidate_reviewer"
ROLE_POSTURE_REVIEWER = "posture_reviewer"
ROLE_MARKET_REVIEWER = "market_reviewer"

CONTRACT_VERSION = "1.0"
PROMPT_VERSION_TRADE_REVIEWER = "trade_reviewer/v1"
PROMPT_VERSION_RUNTIME_ADVISOR = "runtime_advisor/v1"
PROMPT_VERSION_OUTCOME_REVIEWER = "outcome_reviewer/v1"
PROMPT_VERSION_CANDIDATE_REVIEWER = "candidate_reviewer/v2"
PROMPT_VERSION_POSTURE_REVIEWER = "posture_reviewer/v2"
PROMPT_VERSION_MARKET_REVIEWER = "market_reviewer/v1"

DEFER_REASON_UNKNOWN = "unknown"
DEFER_REASON_STALE_STATE = "stale_state"
DEFER_REASON_WEAK_EVIDENCE = "weak_evidence"
DEFER_REASON_CONFLICTING_SIGNALS = "conflicting_signals"
DEFER_REASON_REGIME_MISMATCH = "regime_mismatch"
DEFER_REASON_PORTFOLIO_CONSTRAINT = "portfolio_constraint"
DEFER_REASON_EXECUTION_UNSUITABLE = "execution_unsuitable"
DEFER_REASON_INVALID_OUTPUT = "invalid_output"

DEFER_REASON_TAXONOMY = {
    DEFER_REASON_UNKNOWN,
    DEFER_REASON_STALE_STATE,
    DEFER_REASON_WEAK_EVIDENCE,
    DEFER_REASON_CONFLICTING_SIGNALS,
    DEFER_REASON_REGIME_MISMATCH,
    DEFER_REASON_PORTFOLIO_CONSTRAINT,
    DEFER_REASON_EXECUTION_UNSUITABLE,
    DEFER_REASON_INVALID_OUTPUT,
}


def _clamp_float(value: Any, *, default: float = 0.0, low: float = 0.0, high: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, low), high)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    return items


def _llm_identity() -> dict[str, str]:
    return {"provider": nemotron_provider_name(), "model": nemotron_provider_model()}


def classify_defer_reason(reason: Any) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return DEFER_REASON_UNKNOWN
    if any(token in text for token in {"stale", "warmup", "missing", "data_integrity", "feature_warmup"}):
        return DEFER_REASON_STALE_STATE
    if any(token in text for token in {"conflict", "contradict", "mixed"}):
        return DEFER_REASON_CONFLICTING_SIGNALS
    if any(token in text for token in {"regime", "range", "ranging", "transition"}):
        return DEFER_REASON_REGIME_MISMATCH
    if any(token in text for token in {"portfolio", "exposure", "cash", "position", "funds"}):
        return DEFER_REASON_PORTFOLIO_CONSTRAINT
    if any(token in text for token in {"spread", "execution", "liquidity", "slippage", "order"}):
        return DEFER_REASON_EXECUTION_UNSUITABLE
    if any(token in text for token in {"invalid", "schema", "contract", "malformed"}):
        return DEFER_REASON_INVALID_OUTPUT
    if any(token in text for token in {"weak", "uncertain", "unclear", "hold", "risk", "setup"}):
        return DEFER_REASON_WEAK_EVIDENCE
    return DEFER_REASON_UNKNOWN


@dataclass
class LLMContractEnvelope:
    role: str
    version: str
    confidence: float
    reasons: list[str]
    contradictions: list[str]
    risks: list[str]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_envelope(
    *,
    role: str,
    confidence: Any = 0.0,
    reasons: Any = None,
    contradictions: Any = None,
    risks: Any = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = _llm_identity()
    payload = LLMContractEnvelope(
        role=role,
        version=CONTRACT_VERSION,
        confidence=_clamp_float(confidence, default=0.0),
        reasons=_string_list(reasons),
        contradictions=_string_list(contradictions),
        risks=_string_list(risks),
        meta={
            "provider": identity["provider"],
            "model": identity["model"],
            "schema_valid": True,
            **(meta or {}),
        },
    )
    return payload.to_dict()


def normalize_trade_reviewer_output(
    parsed: dict[str, Any],
    *,
    symbol: str,
) -> dict[str, Any]:
    final_decision = parsed.get("final_decision", {}) if isinstance(parsed.get("final_decision", {}), dict) else {}
    action = str(final_decision.get("action", "HOLD")).upper()
    if action not in {"OPEN", "CLOSE", "HOLD"}:
        action = "HOLD"
    side = final_decision.get("side")
    normalized_fields: list[str] = []
    if side is not None:
        side = str(side).upper()
    if side == "BUY":
        side = "LONG"
        normalized_fields.append("side")
    elif side == "SELL":
        side = "SHORT"
        normalized_fields.append("side")
    raw_symbol = str(final_decision.get("symbol", symbol) or symbol)
    decision_symbol = raw_symbol
    if raw_symbol in {"", "SYMBOL", "<current_symbol>", "CURRENT_SYMBOL", "LINK/USD", "BTC/USD", "ETH/USD"}:
        decision_symbol = symbol
        normalized_fields.append("symbol")
    confidence = _clamp_float(
        final_decision.get("confidence", final_decision.get("size", 0.0 if action == "HOLD" else 0.6)),
        default=0.0 if action == "HOLD" else 0.6,
    )
    reasons = _string_list(final_decision.get("reasons"))
    if not reasons:
        reason = str(final_decision.get("reason", "")).strip()
        if reason:
            reasons = [reason]
            normalized_fields.append("reasons")
    contract = build_envelope(
        role=ROLE_TRADE_REVIEWER,
        confidence=confidence,
        reasons=reasons,
        contradictions=final_decision.get("contradictions"),
        risks=final_decision.get("risks"),
        meta={
            "prompt_version": PROMPT_VERSION_TRADE_REVIEWER,
            "normalized_fields": normalized_fields,
            "normalized_field_count": len(normalized_fields),
        },
    )
    decision = {
        "symbol": decision_symbol,
        "action": action,
        "side": side,
        "size": final_decision.get("size", 0),
        "reason": reasons[0] if reasons else str(final_decision.get("reason", "hold_unspecified")),
        "debug": final_decision.get("debug", {}) if isinstance(final_decision.get("debug", {}), dict) else {},
        "contract": contract,
    }
    return {"final_decision": decision}


def normalize_candidate_review(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized_fields: list[str] = []

    # Handle new 4-bucket schema (market_posture, promotion_bias, size_bias, hold_bias)
    if "market_posture" in parsed:
        market_posture = str(parsed.get("market_posture", "mixed")).strip().lower()
        if market_posture not in {"supportive", "mixed", "hostile"}:
            market_posture = "mixed"
            normalized_fields.append("market_posture")

        promo_bias = str(parsed.get("promotion_bias", "normal")).strip().lower()
        if promo_bias not in {"favor", "normal", "reduce", "block"}:
            promo_bias = "normal"
            normalized_fields.append("promotion_bias")

        size_bias_raw = str(parsed.get("size_bias", "reduced")).strip().lower()
        hold_bias_raw = str(parsed.get("hold_bias", "normal")).strip().lower()

        # Map new schema → downstream-compatible fields
        if market_posture == "supportive":
            decision = "promote"
            priority = 0.80
        elif market_posture == "hostile":
            decision = "demote"
            priority = 0.25
        else:
            decision = "neutral"
            priority = 0.55

        if promo_bias == "favor":
            action_bias = "open_allowed"
            priority = min(priority + 0.12, 1.0)
        elif promo_bias == "block":
            action_bias = "hold_preferred"
            decision = "demote"
        elif promo_bias == "reduce":
            action_bias = "reduce_size"
        else:  # normal
            action_bias = "open_allowed"

        reasons = _string_list(parsed.get("reasons"))
        if not reasons:
            reasons = [str(parsed.get("reason", "candidate_review")).strip() or "candidate_review"]
        return {
            "promotion_decision": decision,
            "priority": _clamp_float(priority, default=0.5),
            "action_bias": action_bias,
            "market_posture": market_posture,
            "hold_bias_nemo": hold_bias_raw if hold_bias_raw in {"patient", "normal", "fast"} else "normal",
            "reason": reasons[0],
            "contract": build_envelope(
                role=ROLE_CANDIDATE_REVIEWER,
                confidence=priority,
                reasons=reasons,
                meta={
                    "prompt_version": PROMPT_VERSION_CANDIDATE_REVIEWER,
                    "normalized_fields": normalized_fields,
                    "normalized_field_count": len(normalized_fields),
                },
            ),
        }

    # Handle old schema (backward compat / heuristic path)
    decision = str(parsed.get("promotion_decision", "neutral")).strip().lower()
    if decision not in {"promote", "neutral", "demote"}:
        decision = "neutral"
        normalized_fields.append("promotion_decision")
    action_bias = str(parsed.get("action_bias", "hold_preferred")).strip().lower()
    if action_bias not in {"open_allowed", "hold_preferred", "reduce_size"}:
        action_bias = "hold_preferred"
        normalized_fields.append("action_bias")
    reasons = _string_list(parsed.get("reasons"))
    if not reasons:
        reasons = [str(parsed.get("reason", "candidate_review")).strip() or "candidate_review"]
        normalized_fields.append("reasons")
    return {
        "promotion_decision": decision,
        "priority": _clamp_float(parsed.get("priority"), default=0.5),
        "action_bias": action_bias,
        "market_posture": "mixed",
        "hold_bias_nemo": "normal",
        "reason": reasons[0],
        "contract": build_envelope(
            role=ROLE_CANDIDATE_REVIEWER,
            confidence=parsed.get("priority", 0.5),
            reasons=reasons,
            contradictions=parsed.get("contradictions"),
            risks=parsed.get("risks"),
            meta={
                "prompt_version": PROMPT_VERSION_CANDIDATE_REVIEWER,
                "normalized_fields": normalized_fields,
                "normalized_field_count": len(normalized_fields),
            },
        ),
    }


def normalize_posture_review(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized_fields: list[str] = []
    posture = str(parsed.get("posture", "neutral")).strip().lower()
    if posture not in {"aggressive", "neutral", "defensive"}:
        posture = "neutral"
        normalized_fields.append("posture")
    promotion_bias = str(parsed.get("promotion_bias", "normal")).strip().lower()
    if promotion_bias not in {"wider", "normal", "tighter"}:
        promotion_bias = "normal"
        normalized_fields.append("promotion_bias")
    exit_bias = str(parsed.get("exit_bias", "standard")).strip().lower()
    if exit_bias not in {"let_run", "standard", "tighten"}:
        exit_bias = "standard"
        normalized_fields.append("exit_bias")
    size_bias = str(parsed.get("size_bias", "normal")).strip().lower()
    if size_bias not in {"increase", "normal", "reduce"}:
        size_bias = "normal"
        normalized_fields.append("size_bias")
    reasons = _string_list(parsed.get("reasons"))
    if not reasons:
        reasons = [str(parsed.get("reason", "posture_review")).strip() or "posture_review"]
        normalized_fields.append("reasons")
    return {
        "posture": posture,
        "promotion_bias": promotion_bias,
        "exit_bias": exit_bias,
        "size_bias": size_bias,
        "reason": reasons[0],
        "contract": build_envelope(
            role=ROLE_POSTURE_REVIEWER,
            confidence=parsed.get("confidence", 0.5),
            reasons=reasons,
            contradictions=parsed.get("contradictions"),
            risks=parsed.get("risks"),
            meta={
                "prompt_version": PROMPT_VERSION_POSTURE_REVIEWER,
                "normalized_fields": normalized_fields,
                "normalized_field_count": len(normalized_fields),
            },
        ),
    }


def normalize_market_state_review(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized_fields: list[str] = []
    market_state = str(parsed.get("market_state", "transition")).strip().lower()
    if market_state not in {"trending", "ranging", "transition"}:
        market_state = "transition"
        normalized_fields.append("market_state")
    lane_bias = str(parsed.get("lane_bias", "favor_selective")).strip().lower()
    if lane_bias not in {"favor_trend", "favor_selective", "reduce_trend_entries"}:
        lane_bias = "favor_selective"
        normalized_fields.append("lane_bias")
    reasons = _string_list(parsed.get("reasons"))
    if not reasons:
        reasons = [str(parsed.get("reason", "market_state_review")).strip() or "market_state_review"]
        normalized_fields.append("reasons")
    breakout_state = str(parsed.get("breakout_state", "unclear")).strip().lower()
    if breakout_state not in {"fresh_breakout", "retest_holding", "breakout_attempt", "inside_range", "unclear"}:
        breakout_state = "unclear"
        normalized_fields.append("breakout_state")
    trend_stage = str(parsed.get("trend_stage", "unclear")).strip().lower()
    if trend_stage not in {"early", "emerging", "confirmed", "stalling", "mixed", "unclear"}:
        trend_stage = "unclear"
        normalized_fields.append("trend_stage")
    volume_confirmation = str(parsed.get("volume_confirmation", "neutral")).strip().lower()
    if volume_confirmation not in {"supportive", "neutral", "weak"}:
        volume_confirmation = "neutral"
        normalized_fields.append("volume_confirmation")
    pullback_quality = str(parsed.get("pullback_quality", "unclear")).strip().lower()
    if pullback_quality not in {"clean_retest", "loose_retest", "higher_low_support", "none", "unclear"}:
        pullback_quality = "unclear"
        normalized_fields.append("pullback_quality")
    late_move_risk = str(parsed.get("late_move_risk", "moderate")).strip().lower()
    if late_move_risk not in {"contained", "moderate", "extended"}:
        late_move_risk = "moderate"
        normalized_fields.append("late_move_risk")
    pattern_explanation = parsed.get("pattern_explanation", {}) if isinstance(parsed.get("pattern_explanation", {}), dict) else {}
    return {
        "market_state": market_state,
        "confidence": _clamp_float(parsed.get("confidence"), default=0.5),
        "lane_bias": lane_bias,
        "reason": reasons[0],
        "breakout_state": breakout_state,
        "trend_stage": trend_stage,
        "volume_confirmation": volume_confirmation,
        "pullback_quality": pullback_quality,
        "late_move_risk": late_move_risk,
        "pattern_explanation": pattern_explanation,
        "contract": build_envelope(
            role=ROLE_MARKET_REVIEWER,
            confidence=parsed.get("confidence", 0.5),
            reasons=reasons,
            contradictions=parsed.get("contradictions"),
            risks=parsed.get("risks"),
            meta={
                "prompt_version": PROMPT_VERSION_MARKET_REVIEWER,
                "normalized_fields": normalized_fields,
                "normalized_field_count": len(normalized_fields),
            },
        ),
    }


def normalize_outcome_review(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized_fields: list[str] = []
    reasons = _string_list(parsed.get("reasons"))
    lesson = str(parsed.get("lesson", "no_lesson")).strip() or "no_lesson"
    if not reasons:
        reasons = [lesson]
        normalized_fields.append("reasons")
    return {
        "outcome_class": str(parsed.get("outcome_class", "normal_exit")).strip() or "normal_exit",
        "lesson": lesson,
        "suggested_adjustment": str(parsed.get("suggested_adjustment", "no_change")).strip() or "no_change",
        "confidence": _clamp_float(parsed.get("confidence"), default=0.5),
        "contract": build_envelope(
            role=ROLE_OUTCOME_REVIEWER,
            confidence=parsed.get("confidence", 0.5),
            reasons=reasons,
            contradictions=parsed.get("contradictions"),
            risks=parsed.get("risks"),
            meta={
                "prompt_version": PROMPT_VERSION_OUTCOME_REVIEWER,
                "normalized_fields": normalized_fields,
                "normalized_field_count": len(normalized_fields),
            },
        ),
    }


def normalize_runtime_advice(parsed: dict[str, Any], *, allowed_keys: set[str]) -> dict[str, Any]:
    raw_updates = parsed.get("recommended_overrides", parsed.get("updates", {}))
    updates = raw_updates if isinstance(raw_updates, dict) else {}
    filtered_updates = {key: value for key, value in updates.items() if key in allowed_keys}
    normalized_fields: list[str] = []
    if filtered_updates != updates:
        normalized_fields.append("recommended_overrides")
    issues = _string_list(parsed.get("issues"))
    cautions = _string_list(parsed.get("cautions"))
    reasons = issues or cautions or [str(parsed.get("summary", "runtime_review")).strip() or "runtime_review"]
    return {
        "summary": str(parsed.get("summary", "")).strip(),
        "recommended_overrides": filtered_updates,
        "issues": issues,
        "cautions": cautions,
        "confidence": _clamp_float(parsed.get("confidence"), default=0.0),
        "contract": build_envelope(
            role=ROLE_RUNTIME_ADVISOR,
            confidence=parsed.get("confidence", 0.0),
            reasons=reasons,
            contradictions=parsed.get("contradictions"),
            risks=parsed.get("risks"),
            meta={
                "prompt_version": PROMPT_VERSION_RUNTIME_ADVISOR,
                "normalized_fields": normalized_fields,
                "normalized_field_count": len(normalized_fields),
            },
        ),
    }
