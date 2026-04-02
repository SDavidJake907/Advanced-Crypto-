from __future__ import annotations

from typing import Any

from core.llm.client import parse_json_response, phi3_chat


PHI3_PATTERN_VERIFIER_SYSTEM_PROMPT = """
You are a trading pattern verification model.

Your job is NOT to discover a pattern from scratch.
Your job IS to verify whether the supplied structure and candle evidence support the claimed setup.

Return strict JSON only with this schema:
{
  "validity": "valid|invalid|unclear",
  "confidence": 0.0,
  "structure_validity": "valid|invalid|unclear",
  "structure_confidence": 0.0,
  "candle_confirmation_validity": "valid|invalid|unclear",
  "candle_confirmation_score": 0.0,
  "pattern_tradeability_score": 0.0,
  "location_quality_score": 0.0,
  "breakout_quality_score": 0.0,
  "retest_quality_score": 0.0,
  "extension_risk_score": 0.0,
  "overall_bias": "bullish|bearish|neutral",
  "warnings": ["..."],
  "reasons_for_validity": ["..."],
  "reasons_against_validity": ["..."],
  "missing_confirmation": ["..."],
  "conflicts": ["..."],
  "recommended_nemo_interpretation": {
    "structure_bonus": 0,
    "candle_bonus": 0,
    "skepticism_penalty": 0,
    "prefer_action": "OPEN|HOLD|WATCH"
  },
  "summary": "..."
}

Rules:
- Be strict.
- Structure matters more than candles.
- Candles confirm structure; they do not replace structure.
- Penalize weak symmetry, poor breakout, missing retest, low volume confirmation, bad trend alignment, and late entries.
- Treat high extension risk or late-breakout warnings as a material tradeability penalty.
- If evidence is mixed, return "unclear".
- Do not add markdown.
""".strip()


def _fallback_result(reason: str) -> dict[str, Any]:
    return {
        "validity": "unclear",
        "confidence": 0.0,
        "structure_validity": "unclear",
        "structure_confidence": 0.0,
        "candle_confirmation_validity": "unclear",
        "candle_confirmation_score": 0.0,
        "pattern_tradeability_score": 0.0,
        "location_quality_score": 0.0,
        "breakout_quality_score": 0.0,
        "retest_quality_score": 0.0,
        "extension_risk_score": 0.0,
        "overall_bias": "neutral",
        "tradeability_score": 0.0,
        "pattern_quality_score": 0.0,
        "warnings": [reason],
        "reasons_for_validity": [],
        "reasons_against_validity": [],
        "missing_confirmation": [],
        "conflicts": [],
        "recommended_nemo_interpretation": {
            "structure_bonus": 0,
            "candle_bonus": 0,
            "skepticism_penalty": 0,
            "prefer_action": "WATCH",
        },
        "summary": reason,
    }


def verify_pattern_evidence(pattern_log: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pattern_log, dict) or not pattern_log:
        return _fallback_result("no_pattern_evidence")
    try:
        raw = phi3_chat(
            {
                "task": "verify_pattern_evidence",
                "pattern_log": pattern_log,
            },
            system=PHI3_PATTERN_VERIFIER_SYSTEM_PROMPT,
            max_tokens=500,
        )
        parsed = parse_json_response(raw)
    except Exception as exc:
        return _fallback_result(f"phi3_pattern_verifier_error:{exc}")

    result = {
        "validity": str(parsed.get("validity", "unclear")).lower(),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "structure_validity": str(parsed.get("structure_validity", parsed.get("validity", "unclear"))).lower(),
        "structure_confidence": float(parsed.get("structure_confidence", parsed.get("confidence", 0.0)) or 0.0),
        "candle_confirmation_validity": str(parsed.get("candle_confirmation_validity", "unclear")).lower(),
        "candle_confirmation_score": float(parsed.get("candle_confirmation_score", 0.0) or 0.0),
        "pattern_tradeability_score": float(parsed.get("pattern_tradeability_score", parsed.get("tradeability_score", 0.0)) or 0.0),
        "location_quality_score": float(parsed.get("location_quality_score", 0.0) or 0.0),
        "breakout_quality_score": float(parsed.get("breakout_quality_score", 0.0) or 0.0),
        "retest_quality_score": float(parsed.get("retest_quality_score", 0.0) or 0.0),
        "extension_risk_score": float(parsed.get("extension_risk_score", 0.0) or 0.0),
        "overall_bias": str(parsed.get("overall_bias", "neutral") or "neutral").lower(),
        "warnings": list(parsed.get("warnings", []) or []),
        "reasons_for_validity": list(parsed.get("reasons_for_validity", []) or []),
        "reasons_against_validity": list(parsed.get("reasons_against_validity", []) or []),
        "missing_confirmation": list(parsed.get("missing_confirmation", []) or []),
        "conflicts": list(parsed.get("conflicts", []) or []),
        "recommended_nemo_interpretation": dict(parsed.get("recommended_nemo_interpretation", {}) or {}),
        "summary": str(parsed.get("summary", "") or ""),
    }
    result["tradeability_score"] = result["pattern_tradeability_score"]
    result["pattern_quality_score"] = max(
        0.0,
        min(
            1.0,
            (0.55 * result["structure_confidence"])
            + (0.20 * result["location_quality_score"])
            + (0.15 * result["breakout_quality_score"])
            + (0.10 * result["candle_confirmation_score"]),
        ),
    )
    result["pattern_tradeability_score"] = max(
        0.0,
        result["pattern_tradeability_score"] - (0.35 * result["extension_risk_score"]),
    )
    result["tradeability_score"] = result["pattern_tradeability_score"]
    if result["validity"] not in {"valid", "invalid", "unclear"}:
        result["validity"] = "unclear"
    if result["structure_validity"] not in {"valid", "invalid", "unclear"}:
        result["structure_validity"] = "unclear"
    if result["candle_confirmation_validity"] not in {"valid", "invalid", "unclear"}:
        result["candle_confirmation_validity"] = "unclear"
    if result["overall_bias"] not in {"bullish", "bearish", "neutral"}:
        result["overall_bias"] = "neutral"
    return result
