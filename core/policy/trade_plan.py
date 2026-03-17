from __future__ import annotations

from typing import Any


def infer_entry_thesis(
    signal: str,
    features: dict[str, object],
    exec_result: dict[str, object] | None = None,
) -> str:
    nemotron_reason = ""
    if exec_result:
        nemotron_payload = exec_result.get("nemotron", {})
        if isinstance(nemotron_payload, dict):
            nemotron_reason = str(nemotron_payload.get("reason", "")).strip()
    if nemotron_reason:
        lowered = nemotron_reason.lower()
        if "meme" in lowered:
            return "meme_acceleration"
        if "reversion" in lowered or "bounce" in lowered:
            return "reversion_bounce"
        if "breakout" in lowered:
            return "selective_breakout"
        if "trend" in lowered or "continuation" in lowered:
            return "trend_continuation"
        return "thesis_from_nemo"

    lane = str(features.get("lane", "L3"))
    trend_1h = int(features.get("trend_1h", 0) or 0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    recommendation = str(features.get("entry_recommendation", "")).upper()
    if lane == "L4":
        return "meme_acceleration"
    if recommendation == "WATCH":
        return "selective_breakout"
    if signal == "LONG" and trend_1h > 0 and momentum_14 > 0.0:
        return "trend_continuation"
    if signal == "SHORT" and trend_1h < 0 and momentum_14 < 0.0:
        return "trend_continuation"
    if str(features.get("regime_7d", "")).lower() in {"ranging", "mean_revert", "reversion"}:
        return "reversion_bounce"
    return "momentum_entry"


def infer_expected_hold_style(features: dict[str, object]) -> str:
    lane = str(features.get("lane", "L3"))
    regime_7d = str(features.get("regime_7d", "unknown")).lower()
    if lane == "L4":
        return "fast"
    if regime_7d in {"bull", "bullish", "trend", "trending"}:
        return "runner"
    return "standard"


def infer_invalidation(features: dict[str, object]) -> str:
    lane = str(features.get("lane", "L3"))
    regime_7d = str(features.get("regime_7d", "unknown")).lower()
    trend_1h = int(features.get("trend_1h", 0) or 0)
    if lane == "L4":
        return "momentum_fade_or_trailing_break"
    if regime_7d in {"ranging", "mean_revert", "reversion"}:
        return "failed_bounce_or_time_stop"
    if trend_1h != 0:
        return "trend_loss_or_time_stop"
    return "trail_break_or_no_followthrough"


def build_trade_plan_metadata(
    signal: str,
    features: dict[str, object],
    exec_result: dict[str, object] | None = None,
) -> dict[str, str]:
    return {
        "entry_thesis": infer_entry_thesis(signal, features, exec_result),
        "expected_hold_style": infer_expected_hold_style(features),
        "invalidate_on": infer_invalidation(features),
    }
