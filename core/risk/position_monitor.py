from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.config.runtime import get_runtime_setting
from core.llm.phi3_exit_posture import ExitPostureDecision
from core.risk.exits import (
    _lane_min_hold_minutes,
    _structure_intact,
    _lane_tighten_min_pnl_pct,
    compute_structure_state,
    maybe_apply_exit_posture,
    maybe_arm_break_even,
    maybe_update_trailing,
    review_live_exit_state,
    update_position_excursions,
)
from core.risk.portfolio import Position


@dataclass
class PositionStateDecision:
    state: str
    reason: str
    confidence: float
    posture: ExitPostureDecision


@dataclass
class PositionMonitorResult:
    position: Position
    phi3_posture: ExitPostureDecision
    live_posture: ExitPostureDecision
    lane_state: PositionStateDecision
    live_state: PositionStateDecision
    final_state: PositionStateDecision


def _build_state_decision(state: str, reason: str, confidence: float) -> PositionStateDecision:
    posture_map = {
        "RUN": "RUN",
        "STALL": "STALE",
        "WEAKEN": "TIGHTEN",
        "FAIL": "EXIT",
        "ROTATE": "EXIT",
    }
    posture = ExitPostureDecision(posture_map.get(state, "RUN"), reason, confidence)
    return PositionStateDecision(state=state, reason=reason, confidence=confidence, posture=posture)


def _state_from_posture(posture: ExitPostureDecision) -> PositionStateDecision:
    mapped = {
        "RUN": "RUN",
        "STALE": "STALL",
        "TIGHTEN": "WEAKEN",
        "EXIT": "ROTATE" if str(posture.reason).startswith("rotate_exit:") else "FAIL",
    }
    state = mapped.get(posture.posture, "RUN")
    return PositionStateDecision(state=state, reason=posture.reason, confidence=posture.confidence, posture=posture)


def _merge_states(*states: PositionStateDecision) -> PositionStateDecision:
    priority = {"RUN": 1, "STALL": 2, "WEAKEN": 3, "FAIL": 4, "ROTATE": 5}
    chosen = _build_state_decision("RUN", "default_hold_state", 0.0)
    for state in states:
        if priority.get(state.state, 0) > priority.get(chosen.state, 0):
            chosen = state
        elif state.state == chosen.state and state.confidence > chosen.confidence:
            chosen = state
    return chosen


def _lane_monitor_state(
    position: Position,
    *,
    price: float,
    hold_minutes: float,
    features: dict[str, Any],
    universe_context: dict[str, Any] | None = None,
) -> PositionStateDecision:
    if position.entry_price is None or position.entry_price <= 0.0 or price <= 0.0:
        return _build_state_decision("RUN", "lane_monitor_data_incomplete", 0.0)

    lane = str(position.lane or "L3").upper()
    pnl_pct = (
        ((price / position.entry_price) - 1.0) * 100.0
        if position.side == "LONG"
        else ((position.entry_price / price) - 1.0) * 100.0
    )
    momentum = float(features.get("momentum", features.get("momentum_5", 0.0)) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    trend_1h = int(features.get("trend_1h", 0) or 0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    rsi = float(features.get("rsi", 50.0) or 50.0)
    context = universe_context or {}
    still_top = bool(context.get("current_symbol_is_top_candidate")) or bool(
        context.get("current_symbol_is_top_lane_candidate")
    )
    rotation_margin_score = float(get_runtime_setting("ROTATION_MARGIN_SCORE"))
    rotation_persist_rank_delta = int(get_runtime_setting("ROTATION_PERSIST_MIN_RANK_DELTA"))
    rotation_persist_lane_rank_delta = int(get_runtime_setting("ROTATION_PERSIST_MIN_LANE_RANK_DELTA"))
    rotation_persist_urgency = float(get_runtime_setting("ROTATION_PERSIST_MIN_LEADER_URGENCY"))
    weak_hold_rotation_min_hold = float(get_runtime_setting("ROTATION_WEAK_HOLD_MIN_HOLD_MIN"))
    current_candidate_score = float(features.get("entry_score", 0.0) or 0.0)
    top_scored = context.get("top_scored") or []
    replacement_advantage = 0.0
    replacement_persistent = False
    top_score = 0.0
    top_urgency = 0.0
    if top_scored and isinstance(top_scored[0], dict):
        top_score = float(top_scored[0].get("candidate_score", 0.0) or 0.0)
        top_urgency = float(top_scored[0].get("leader_urgency", 0.0) or 0.0)
        replacement_advantage = top_score - current_candidate_score
        replacement_persistent = (
            bool(top_scored[0].get("leader_takeover", False))
            or int(top_scored[0].get("rank_delta", 0) or 0) >= rotation_persist_rank_delta
            or int(top_scored[0].get("lane_rank_delta", 0) or 0) >= rotation_persist_lane_rank_delta
            or top_urgency >= rotation_persist_urgency
        )
    hold_style = str(position.expected_hold_style or "").lower()
    structure_quality = float(features.get("structure_quality", 0.0) or 0.0)
    continuation_quality = float(features.get("continuation_quality", 0.0) or 0.0)
    ema_cross_bullish = bool(features.get("ema9_above_ema26", False))
    ema_cross_distance_pct = float(features.get("ema_cross_distance_pct", 0.0) or 0.0)
    structure_build = bool(features.get("structure_build", False))
    breakout_failure_risk = bool(features.get("breakout_failure_risk", False))
    structure_break_risk = bool(features.get("structure_break_risk", False))
    structure_state = str(features.get("structure_state") or compute_structure_state(position, features))
    overextended = bool(features.get("overextended", False))
    structured_hold = hold_style in {"structured_runner", "rotation_runner", "leader_runner"} or (
        lane in {"L2", "L3"}
        and (
            (structure_quality >= 72.0 and continuation_quality >= 70.0)
            or (ema_cross_bullish and ema_cross_distance_pct >= 0.002 and continuation_quality >= 66.0)
        )
    )
    structure_intact = lane in {"L1", "L2", "L3"} and _structure_intact(position, features)
    if structured_hold:
        if hold_style == "leader_runner" and lane == "L1":
            structured_min_hold = 720.0
        elif hold_style == "rotation_runner" and lane == "L2":
            structured_min_hold = 300.0
        else:
            structured_min_hold = 240.0 if lane == "L2" else 360.0 if lane == "L3" else 0.0
    else:
        structured_min_hold = 0.0
    lane_min_hold = _lane_min_hold_minutes(position)
    tighten_min_pnl = _lane_tighten_min_pnl_pct(lane)
    l3_broken_weaken_min_hold = float(get_runtime_setting("EXIT_L3_BROKEN_WEAKEN_MIN_HOLD_MIN"))
    l3_broken_weaken_neg_pnl = float(get_runtime_setting("EXIT_L3_BROKEN_WEAKEN_NEG_PNL_PCT"))
    l3_broken_fail_min_hold = float(get_runtime_setting("EXIT_L3_BROKEN_FAIL_MIN_HOLD_MIN"))
    l3_broken_fail_neg_pnl = float(get_runtime_setting("EXIT_L3_BROKEN_FAIL_NEG_PNL_PCT"))
    l3_fragile_weaken_min_hold = float(get_runtime_setting("EXIT_L3_FRAGILE_WEAKEN_MIN_HOLD_MIN"))
    l3_fragile_weaken_neg_pnl = float(get_runtime_setting("EXIT_L3_FRAGILE_WEAKEN_NEG_PNL_PCT"))
    mae_r = float(getattr(position, "mae_r", 0.0) or 0.0)

    if (
        not structured_hold
        and not structure_intact
        and not still_top
        and pnl_pct <= 0.0
        and momentum_5 <= 0.0
        and hold_minutes >= max(10.0, lane_min_hold)
        and replacement_advantage >= rotation_margin_score
        and replacement_persistent
    ):
        if top_score >= 80.0 and top_urgency >= 3.0:
            return _build_state_decision("ROTATE", "rotation_to_stronger_candidate", 0.78)

    weak_score_for_lane = current_candidate_score < max(64.0, top_score - max(rotation_margin_score * 0.5, 4.0))
    weak_continuation = continuation_quality < 62.0 or structure_quality < 60.0
    weak_momentum = momentum_5 <= 0.0 and momentum_14 <= 0.001
    rotation_candidate_ready = (
        not structured_hold
        and not still_top
        and replacement_advantage >= rotation_margin_score
        and replacement_persistent
        and top_score >= 80.0
        and top_urgency >= 3.0
        and hold_minutes >= max(weak_hold_rotation_min_hold, lane_min_hold)
    )
    if (
        lane in {"L2", "L3"}
        and rotation_candidate_ready
        and weak_score_for_lane
        and (structure_state in {"fragile", "broken"} or weak_continuation or weak_momentum)
        and pnl_pct <= 0.75
    ):
        return _build_state_decision("ROTATE", "rotation_to_stronger_candidate", 0.74)

    if lane == "L1":
        if pnl_pct > 0.0 and hold_minutes < max(12.0, lane_min_hold) and trend_1h > 0 and momentum_14 >= 0.0:
            return _build_state_decision("RUN", "l1_early_breakout_room", 0.72)
        if not structure_intact and pnl_pct >= tighten_min_pnl and momentum_5 < -0.0015 and volume_ratio < 0.9:
            return _build_state_decision("WEAKEN", "l1_breakout_momentum_fade", 0.76)
        if hold_minutes >= 12.0 and (structure_break_risk or breakout_failure_risk) and trend_1h <= 0:
            return _build_state_decision("FAIL", "l1_breakout_failed", 0.88)
        if hold_minutes >= 12.0 and pnl_pct < -1.0 and trend_1h <= 0 and momentum_5 < -0.002:
            return _build_state_decision("FAIL", "l1_breakout_failed", 0.82)
        return _build_state_decision("RUN", "l1_trend_room", 0.64)

    if lane == "L2":
        if hold_minutes >= max(15.0, structured_min_hold) and (structure_break_risk or breakout_failure_risk):
            return _build_state_decision("FAIL", "l2_bounce_failed", 0.88)
        if hold_minutes >= max(15.0, structured_min_hold) and pnl_pct < -1.0 and momentum_5 <= 0.0:
            return _build_state_decision("FAIL", "l2_bounce_failed", 0.84)
        if not structured_hold and not structure_intact and pnl_pct >= tighten_min_pnl and momentum_5 <= 0.0:
            return _build_state_decision("WEAKEN", "l2_rebound_flattening", 0.74)
        if (
            not structured_hold
            and not structure_intact
            and hold_minutes >= max(30.0, structured_min_hold, lane_min_hold)
            and abs(pnl_pct) <= 0.60
            and abs(momentum) < 0.0015
        ):
            return _build_state_decision("STALL", "l2_no_follow_through", 0.8)
        return _build_state_decision("RUN", "l2_rebound_alive", 0.62)

    if lane == "L4":
        if pnl_pct >= tighten_min_pnl and (momentum_5 < 0.0 or rsi >= 68.0 or overextended):
            return _build_state_decision("WEAKEN", "l4_protect_fast_move", 0.78)
        if hold_minutes >= 5.0 and (breakout_failure_risk or (momentum_5 < -0.002 and not still_top)):
            return _build_state_decision("FAIL", "l4_momentum_collapse", 0.86)
        if hold_minutes >= max(10.0, lane_min_hold) and abs(pnl_pct) <= 0.4 and abs(momentum) < 0.002:
            return _build_state_decision("STALL", "l4_no_progress", 0.82)
        return _build_state_decision("RUN", "l4_move_alive", 0.60)

    if (
        not structured_hold
        and not structure_intact
        and
        hold_minutes >= max(float(get_runtime_setting("EXIT_STALE_MIN_HOLD_MIN")), structured_min_hold, lane_min_hold)
        and abs(pnl_pct) <= max(float(get_runtime_setting("EXIT_STALE_MAX_ABS_PNL_PCT")), 0.6)
        and abs(momentum) < 0.0015
        and not structure_build
    ):
        return _build_state_decision("STALL", "l3_no_progress", 0.78)
    if (
        not structured_hold
        and not structure_intact
        and pnl_pct >= max(tighten_min_pnl * 0.5, 1.0)
        and momentum_5 < 0.0
    ):
        return _build_state_decision("WEAKEN", "l3_trend_softening", 0.72)
    if structure_state == "broken":
        if not structured_hold and hold_minutes >= max(30.0, lane_min_hold):
            if pnl_pct <= -2.0 or momentum_5 < -0.0015:
                return _build_state_decision("FAIL", "l3_structure_broken_stale", 0.88)
            if pnl_pct <= -0.5 or mae_r <= -0.75:
                return _build_state_decision("WEAKEN", "l3_structure_broken_stale", 0.8)
        if mae_r <= -2.5:
            return _build_state_decision("FAIL", "l3_structure_broken_deep_adverse", 0.9)
        if mae_r <= -1.5:
            return _build_state_decision("WEAKEN", "l3_structure_broken_adverse", 0.82)
        if hold_minutes >= max(l3_broken_fail_min_hold, lane_min_hold) and pnl_pct <= l3_broken_fail_neg_pnl:
            return _build_state_decision("FAIL", "l3_structure_broken_neg", 0.84)
        if hold_minutes >= max(l3_broken_weaken_min_hold, lane_min_hold) and pnl_pct <= l3_broken_weaken_neg_pnl:
            return _build_state_decision("WEAKEN", "l3_structure_broken_watch", 0.76)
    if structure_state == "fragile":
        if mae_r <= -3.0:
            return _build_state_decision("FAIL", "l3_fragile_deep_adverse", 0.86)
        if mae_r <= -1.75:
            return _build_state_decision("WEAKEN", "l3_fragile_adverse", 0.76)
        if hold_minutes >= max(l3_fragile_weaken_min_hold, lane_min_hold) and pnl_pct <= l3_fragile_weaken_neg_pnl:
            return _build_state_decision("WEAKEN", "l3_fragile_negative", 0.7)
    return _build_state_decision("RUN", "l3_balanced_hold", 0.62)


def monitor_open_position(
    position: Position,
    *,
    price: float,
    atr: float,
    hold_minutes: float,
    features: dict[str, Any],
    phi3_posture: ExitPostureDecision,
    universe_context: dict[str, Any] | None = None,
) -> PositionMonitorResult:
    updated_position = maybe_arm_break_even(position, price)
    updated_position = update_position_excursions(updated_position, price=price, bar_ts=str(features.get("bar_ts") or ""))
    structure_state = compute_structure_state(updated_position, features)
    features = dict(features)
    features["structure_state"] = structure_state
    updated_position = maybe_update_trailing(updated_position, price, atr)
    lane_state = _lane_monitor_state(
        updated_position,
        price=price,
        hold_minutes=hold_minutes,
        features=features,
        universe_context=universe_context,
    )
    live_posture = review_live_exit_state(
        updated_position,
        price=price,
        hold_minutes=hold_minutes,
        features=features,
        universe_context=universe_context,
    )
    live_state = _state_from_posture(live_posture)
    phi3_state = _state_from_posture(phi3_posture)
    final_state = _merge_states(lane_state, phi3_state, live_state)
    updated_position = replace(
        updated_position,
        monitor_state=final_state.state,
        monitor_reason=final_state.reason,
        monitor_confidence=final_state.confidence,
        structure_state=structure_state,
    )
    updated_position = maybe_apply_exit_posture(
        updated_position,
        price=price,
        atr=atr,
        posture=final_state.posture,
    )
    return PositionMonitorResult(
        position=updated_position,
        phi3_posture=phi3_posture,
        live_posture=final_state.posture,
        lane_state=lane_state,
        live_state=live_state,
        final_state=final_state,
    )
