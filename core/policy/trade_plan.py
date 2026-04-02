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
    structure_quality = float(features.get("structure_quality", 0.0) or 0.0)
    continuation_quality = float(features.get("continuation_quality", 0.0) or 0.0)
    breakout = bool(features.get("range_breakout_1h", False))
    breakout_confirmed = bool(features.get("breakout_confirmed", False))
    pullback_hold = bool(features.get("pullback_hold", False))
    retest_confirmed = bool(features.get("retest_confirmed", False))
    structure_build = bool(features.get("structure_build", False))
    overextended = bool(features.get("overextended", False))
    ema_aligned = bool(features.get("ema9_above_ema20", False)) and bool(features.get("price_above_ema20", False))
    ema_cross_bullish = bool(features.get("ema9_above_ema26", False))
    ema_cross_distance_pct = float(features.get("ema_cross_distance_pct", 0.0) or 0.0)
    if lane == "L4":
        return "fast_breakout"
    if lane == "L1" and (
        breakout_confirmed
        or retest_confirmed
        or (structure_build and ema_cross_bullish and continuation_quality >= 68.0)
        or (structure_quality >= 74.0 and continuation_quality >= 72.0 and not overextended)
    ):
        return "leader_runner"
    if lane in {"L2", "L3"} and (
        (breakout or breakout_confirmed or pullback_hold or retest_confirmed)
        or (ema_aligned and ema_cross_bullish and continuation_quality >= 68.0)
        or (ema_cross_bullish and ema_cross_distance_pct >= 0.002 and continuation_quality >= 66.0)
        or (structure_quality >= 72.0 and continuation_quality >= 70.0 and structure_build)
    ):
        return "rotation_runner" if lane == "L2" else "structured_runner"
    if regime_7d in {"bull", "bullish", "trend", "trending"}:
        return "runner"
    return "standard"


def infer_invalidation(features: dict[str, object]) -> str:
    lane = str(features.get("lane", "L3"))
    regime_7d = str(features.get("regime_7d", "unknown")).lower()
    trend_1h = int(features.get("trend_1h", 0) or 0)
    if lane == "L4":
        return "15m_breakout_failure_or_fast_trail_break"
    if lane == "L1":
        return "1h_structure_break_or_4h_macro_failure"
    if lane == "L2":
        return "1h_rotation_failure_or_failed_retest"
    if regime_7d in {"ranging", "mean_revert", "reversion"}:
        return "failed_bounce_or_time_stop"
    if trend_1h != 0:
        return "governing_structure_break_or_time_stop"
    return "15m_structure_break_or_no_followthrough"


def build_trade_plan_metadata(
    signal: str,
    features: dict[str, object],
    exec_result: dict[str, object] | None = None,
) -> dict[str, str | float | bool]:
    return {
        "entry_thesis": infer_entry_thesis(signal, features, exec_result),
        "expected_hold_style": infer_expected_hold_style(features),
        "invalidate_on": infer_invalidation(features),
        "governing_timeframe": str(features.get("governing_timeframe", "15m")),
        "macro_timeframe": str(features.get("macro_timeframe", "4h")),
        "trigger_timeframe": str(features.get("trigger_timeframe", "15m")),
        "expected_move_pct": float(features.get("expected_move_pct", 0.0) or 0.0),
        "total_cost_pct": float(features.get("total_cost_pct", 0.0) or 0.0),
        "required_edge_pct": float(features.get("required_edge_pct", 0.0) or 0.0),
        "net_edge_pct": float(features.get("net_edge_pct", 0.0) or 0.0),
        "tp_after_cost_valid": bool(features.get("tp_after_cost_valid", False)),
    }
