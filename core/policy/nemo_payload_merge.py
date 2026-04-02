from __future__ import annotations

from typing import Any


def merge_candidate_with_phi3(candidate: dict[str, Any], phi3_result: dict[str, Any]) -> dict[str, Any]:
    merged = dict(candidate)
    structure_confidence = float(phi3_result.get("structure_confidence", phi3_result.get("confidence", 0.0)) or 0.0)
    candle_confirmation_score = float(phi3_result.get("candle_confirmation_score", 0.0) or 0.0)
    location_quality_score = float(phi3_result.get("location_quality_score", 0.0) or 0.0)
    breakout_quality_score = float(phi3_result.get("breakout_quality_score", 0.0) or 0.0)
    retest_quality_score = float(phi3_result.get("retest_quality_score", 0.0) or 0.0)
    extension_risk_score = float(phi3_result.get("extension_risk_score", 0.0) or 0.0)
    pattern_tradeability_score = float(phi3_result.get("pattern_tradeability_score", phi3_result.get("tradeability_score", 0.0)) or 0.0)
    pattern_quality_score = float(phi3_result.get("pattern_quality_score", 0.0) or 0.0)
    if pattern_quality_score <= 0.0:
        pattern_quality_score = max(
            0.0,
            min(
                1.0,
                (0.55 * structure_confidence)
                + (0.20 * location_quality_score)
                + (0.15 * breakout_quality_score)
                + (0.10 * candle_confirmation_score),
            ),
        )
    verification = {
        "validity": phi3_result.get("validity", "unclear"),
        "confidence": float(phi3_result.get("confidence", 0.0) or 0.0),
        "structure_validity": str(phi3_result.get("structure_validity", phi3_result.get("validity", "unclear")) or "unclear"),
        "structure_confidence": structure_confidence,
        "candle_confirmation_validity": str(phi3_result.get("candle_confirmation_validity", "unclear") or "unclear"),
        "candle_confirmation_score": candle_confirmation_score,
        "pattern_tradeability_score": pattern_tradeability_score,
        "location_quality_score": location_quality_score,
        "breakout_quality_score": breakout_quality_score,
        "retest_quality_score": retest_quality_score,
        "extension_risk_score": extension_risk_score,
        "overall_bias": str(phi3_result.get("overall_bias", "neutral") or "neutral"),
        "tradeability_score": pattern_tradeability_score,
        "pattern_quality_score": pattern_quality_score,
        "warnings": list(phi3_result.get("warnings", []) or []),
        "reasons_for_validity": list(phi3_result.get("reasons_for_validity", []) or []),
        "reasons_against_validity": list(phi3_result.get("reasons_against_validity", []) or []),
        "missing_confirmation": list(phi3_result.get("missing_confirmation", []) or []),
        "conflicts": list(phi3_result.get("conflicts", []) or []),
        "recommended_nemo_interpretation": dict(phi3_result.get("recommended_nemo_interpretation", {}) or {}),
        "summary": str(phi3_result.get("summary", "") or ""),
    }
    merged["pattern_verification"] = verification

    validity = str(verification.get("validity", "unclear")).lower()
    quality = float(verification.get("pattern_quality_score", 0.0) or 0.0)
    entry_score = float(merged.get("entry_score", 0.0) or 0.0)
    if validity == "valid":
        extension_penalty = min(max(extension_risk_score, 0.0), 1.0) * 6.0
        merged["entry_score"] = min(100.0, max(0.0, entry_score + (quality * 10.0) - extension_penalty))
        merged["phi3_veto_flag"] = False
    elif validity == "invalid":
        merged["entry_score"] = max(0.0, entry_score - 8.0)
        merged["phi3_veto_flag"] = True
    else:
        merged["phi3_veto_flag"] = False
    return merged
