from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from core.config.runtime import get_runtime_setting, get_symbol_lane
from core.llm.phi3_exit_posture import ExitPostureDecision
from core.risk.portfolio import Position


def _lane_min_hold_minutes(position: Position) -> float:
    lane = str(position.lane or get_symbol_lane(position.symbol)).upper()
    if lane == "L1":
        return float(get_runtime_setting("L1_MIN_HOLD_MIN"))
    if lane == "L2":
        return float(get_runtime_setting("L2_MIN_HOLD_MIN"))
    if lane == "L4":
        return float(get_runtime_setting("L4_MIN_HOLD_MIN"))
    return float(get_runtime_setting("L3_MIN_HOLD_MIN"))


def _lane_tighten_min_pnl_pct(lane: str) -> float:
    lane = str(lane or "L3").upper()
    if lane == "L1":
        return float(get_runtime_setting("L1_EXIT_TIGHTEN_MIN_PNL_PCT"))
    if lane == "L2":
        return float(get_runtime_setting("L2_EXIT_TIGHTEN_MIN_PNL_PCT"))
    if lane == "L4":
        return float(get_runtime_setting("L4_EXIT_TIGHTEN_MIN_PNL_PCT"))
    return float(get_runtime_setting("L3_EXIT_TIGHTEN_MIN_PNL_PCT"))


def _lane_primary_tp_atr_mult(lane: str) -> float:
    lane = str(lane or "L3").upper()
    if lane == "L1":
        return float(get_runtime_setting("L1_EXIT_PRIMARY_TP_ATR_MULT")) * 1.4  # Data-driven bump to 2.5+
    if lane == "L2":
        return float(get_runtime_setting("L2_EXIT_PRIMARY_TP_ATR_MULT")) * 1.6  # Data-driven bump to 2.2+
    if lane == "L4":
        return float(get_runtime_setting("MEME_EXIT_PRIMARY_TP_ATR_MULT"))
    return float(get_runtime_setting("EXIT_PRIMARY_TP_ATR_MULT"))


def _min_viable_take_profit_pct(lane: str) -> float:
    cost_floor_pct = _exit_profit_floor_pct()
    safety_buffer_pct = float(get_runtime_setting("TRADE_COST_SAFETY_BUFFER_PCT"))
    tighten_floor_pct = _lane_tighten_min_pnl_pct(lane) * 0.6
    return max(cost_floor_pct + safety_buffer_pct, tighten_floor_pct)


def _structured_hold_min_minutes(position: Position, features: dict[str, Any]) -> float:
    hold_style = str(position.expected_hold_style or "").lower()
    lane = str(position.lane or get_symbol_lane(position.symbol)).upper()
    structure_quality = float(features.get("structure_quality", 0.0) or 0.0)
    continuation_quality = float(features.get("continuation_quality", 0.0) or 0.0)
    ema_cross_bullish = bool(features.get("ema9_above_ema26", False))
    ema_cross_distance_pct = float(features.get("ema_cross_distance_pct", 0.0) or 0.0)
    breakout = bool(features.get("range_breakout_1h", False))
    pullback_hold = bool(features.get("pullback_hold", False))

    if hold_style == "leader_runner":
        return 720.0 if lane == "L1" else 360.0
    if hold_style == "rotation_runner":
        return 300.0 if lane == "L2" else 240.0
    if hold_style == "structured_runner":
        if lane == "L2":
            return 240.0
        if lane == "L3":
            return 360.0
        return 180.0
    if lane in {"L2", "L3"} and (
        breakout
        or pullback_hold
        or (ema_cross_bullish and ema_cross_distance_pct >= 0.002 and continuation_quality >= 66.0)
        or (structure_quality >= 72.0 and continuation_quality >= 70.0)
    ):
        return 180.0 if lane == "L2" else 240.0
    return 0.0


def _is_hard_failure_reason(reason: str) -> bool:
    lowered = str(reason or "").lower()
    return lowered.startswith("hard_fail:") or lowered.startswith("l2_bounce_failed") or lowered.startswith("l4_momentum_collapse") or lowered.startswith("l1_breakout_failed")


def _exit_profit_floor_pct() -> float:
    maker_fee_pct = float(get_runtime_setting("EXEC_MAKER_FEE_PCT"))
    taker_fee_pct = float(get_runtime_setting("EXEC_TAKER_FEE_PCT"))
    # Conservative exit floor: entry may be maker or taker, exits often end up less favorable than ideal maker math.
    modeled_round_trip_pct = maker_fee_pct + max(maker_fee_pct, taker_fee_pct * 0.75)
    configured_floor_pct = float(get_runtime_setting("EXIT_MIN_PROFIT_AFTER_COST_PCT"))
    return max(modeled_round_trip_pct, configured_floor_pct)


def _green_strength_hold_active(position: Position, *, price: float, features: dict[str, Any] | None) -> bool:
    if not bool(get_runtime_setting("EXIT_GREEN_EMA_RSI_ATR_HOLD")):
        return False
    if position.entry_price is None or position.entry_price <= 0.0 or price <= 0.0:
        return False
    if position.side != "LONG":
        return False
    if features is None:
        return False
    pnl_pct = ((price / position.entry_price) - 1.0) * 100.0
    if pnl_pct <= 0.0:
        return False

    ema_bullish = bool(features.get("ema9_above_ema20", features.get("ema9_above_ema26", False)))
    rsi = float(features.get("rsi", 50.0) or 50.0)
    min_rsi = float(get_runtime_setting("EXIT_GREEN_HOLD_MIN_RSI"))
    if not ema_bullish or rsi < min_rsi:
        return False

    stop_level = position.trail_stop if position.trail_stop is not None else position.stop_loss
    atr = float(features.get("atr", 0.0) or 0.0)
    min_atr_buffer = float(get_runtime_setting("EXIT_GREEN_HOLD_MIN_STOP_ATR_BUFFER"))
    if stop_level is None:
        return True
    if atr <= 0.0:
        return price > stop_level
    return (price - stop_level) >= (atr * min_atr_buffer)


def _structure_intact(position: Position, features: dict[str, Any]) -> bool:
    lane = str(position.lane or get_symbol_lane(position.symbol)).upper()
    mae_r = float(position.mae_r or 0.0)
    structure_quality = float(features.get("structure_quality", 0.0) or 0.0)
    continuation_quality = float(features.get("continuation_quality", 0.0) or 0.0)
    ema_cross_bullish = bool(features.get("ema9_above_ema26", False))
    ema_cross_distance_pct = float(features.get("ema_cross_distance_pct", 0.0) or 0.0)
    trend_confirmed = bool(features.get("trend_confirmed", False))
    breakout = bool(features.get("range_breakout_1h", False))
    breakout_confirmed = bool(features.get("breakout_confirmed", False))
    pullback_hold = bool(features.get("pullback_hold", False))
    retest_confirmed = bool(features.get("retest_confirmed", False))
    structure_build = bool(features.get("structure_build", False))
    structure_break_risk = bool(features.get("structure_break_risk", False))
    breakout_failure_risk = bool(features.get("breakout_failure_risk", False))
    higher_low_count = int(features.get("higher_low_count", 0) or 0)
    if mae_r <= -1.5:
        return False
    if structure_break_risk:
        return False
    if lane == "L4":
        return bool(
            trend_confirmed
            and ema_cross_bullish
            and ema_cross_distance_pct >= 0.001
            and not breakout_failure_risk
        )
    return bool(
        (ema_cross_bullish and ema_cross_distance_pct >= 0.001)
        or breakout
        or breakout_confirmed
        or pullback_hold
        or retest_confirmed
        or structure_build
        or higher_low_count >= 3
        or (trend_confirmed and structure_quality >= 60.0 and continuation_quality >= 62.0)
    )


def compute_structure_state(position: Position, features: dict[str, Any]) -> str:
    if bool(features.get("structure_break_risk", False)) or bool(features.get("breakout_failure_risk", False)):
        return "broken"
    if _structure_intact(position, features):
        return "intact"
    structure_quality = float(features.get("structure_quality", 0.0) or 0.0)
    continuation_quality = float(features.get("continuation_quality", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    ema_cross_bullish = bool(features.get("ema9_above_ema26", False))
    if ema_cross_bullish or structure_quality >= 55.0 or continuation_quality >= 55.0 or momentum_5 > -0.002:
        return "fragile"
    return "broken"


def update_position_excursions(position: Position, *, price: float, bar_ts: str | None) -> Position:
    if position.entry_price is None or position.entry_price <= 0.0 or price <= 0.0:
        return position
    if position.risk_r is None or float(position.risk_r or 0.0) <= 0.0:
        return position
    max_price_seen = float(position.max_price_seen or position.entry_price)
    min_price_seen = float(position.min_price_seen or position.entry_price)
    mfe_pct = float(position.mfe_pct or 0.0)
    mae_pct = float(position.mae_pct or 0.0)
    mfe_r = float(position.mfe_r or 0.0)
    mae_r = float(position.mae_r or 0.0)
    etd_pct = float(position.etd_pct or 0.0)
    etd_r = float(position.etd_r or 0.0)
    mfe_ts = str(position.mfe_ts or "")
    mae_ts = str(position.mae_ts or "")
    risk_r = float(position.risk_r)

    if position.side == "LONG":
        if price > max_price_seen:
            max_price_seen = price
            mfe_pct = ((price / position.entry_price) - 1.0) * 100.0
            mfe_r = ((price - position.entry_price) / risk_r)
            mfe_ts = str(bar_ts or datetime.now(timezone.utc).isoformat())
        if price < min_price_seen:
            min_price_seen = price
            mae_pct = min(mae_pct, ((price / position.entry_price) - 1.0) * 100.0)
            mae_r = min(mae_r, ((price - position.entry_price) / risk_r))
            mae_ts = str(bar_ts or datetime.now(timezone.utc).isoformat())
        current_pnl_pct = ((price / position.entry_price) - 1.0) * 100.0
        current_r = (price - position.entry_price) / risk_r
    else:
        if price < min_price_seen:
            min_price_seen = price
            mfe_pct = max(mfe_pct, ((position.entry_price / price) - 1.0) * 100.0)
            mfe_r = max(mfe_r, ((position.entry_price - price) / risk_r))
            mfe_ts = str(bar_ts or datetime.now(timezone.utc).isoformat())
        if price > max_price_seen:
            max_price_seen = price
            mae_pct = min(mae_pct, ((position.entry_price / price) - 1.0) * 100.0)
            mae_r = min(mae_r, ((position.entry_price - price) / risk_r))
            mae_ts = str(bar_ts or datetime.now(timezone.utc).isoformat())
        current_pnl_pct = ((position.entry_price / price) - 1.0) * 100.0
        current_r = (position.entry_price - price) / risk_r

    etd_pct = max(mfe_pct - current_pnl_pct, 0.0)
    etd_r = max(mfe_r - current_r, 0.0)

    return replace(
        position,
        max_price_seen=max_price_seen,
        min_price_seen=min_price_seen,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        mfe_r=mfe_r,
        mae_r=mae_r,
        etd_pct=etd_pct,
        etd_r=etd_r,
        mfe_ts=mfe_ts,
        mae_ts=mae_ts,
    )


def build_exit_plan(
    *,
    symbol: str,
    side: str,
    weight: float,
    entry_price: float,
    atr: float,
    entry_bar_ts: str | None,
    entry_bar_idx: int | None,
    entry_reasons: list[str] | None = None,
    lane: str | None = None,
    entry_thesis: str | None = None,
    expected_hold_style: str | None = None,
    invalidate_on: str | None = None,
    expected_edge_pct: float = 0.0,
) -> Position:
    lane = lane or get_symbol_lane(symbol)
    hold_style = str(expected_hold_style or "").lower()
    if lane == "L1":
        stop_mult = float(get_runtime_setting("L1_EXIT_ATR_STOP_MULT")) * 1.25  # Bump to 2.2+
        min_stop_pct = float(get_runtime_setting("L1_EXIT_MIN_STOP_PCT")) / 100.0
    elif lane == "L2":
        stop_mult = float(get_runtime_setting("L2_EXIT_ATR_STOP_MULT")) * 1.15  # Bump to 2.0+
        min_stop_pct = float(get_runtime_setting("L2_EXIT_MIN_STOP_PCT")) / 100.0
    elif lane == "L4":
        stop_mult = float(get_runtime_setting("MEME_EXIT_ATR_STOP_MULT"))
        min_stop_pct = float(get_runtime_setting("MEME_EXIT_MIN_STOP_PCT")) / 100.0
    else:
        stop_mult = float(get_runtime_setting("EXIT_ATR_STOP_MULT"))
        min_stop_pct = float(get_runtime_setting("EXIT_MIN_STOP_PCT")) / 100.0
    atr_risk_r = atr * stop_mult
    min_risk_r = entry_price * min_stop_pct if entry_price > 0.0 else 0.0
    risk_r = max(atr_risk_r, min_risk_r, 0.0)
    tp_mult = _lane_primary_tp_atr_mult(lane)
    min_tp_pct = _min_viable_take_profit_pct(lane) / 100.0
    tp_distance = max(atr * tp_mult, entry_price * min_tp_pct) if entry_price > 0.0 else 0.0
    risk_reward_ratio = (tp_distance / risk_r) if (tp_distance > 0.0 and risk_r > 0.0) else 0.0
    runner_style = hold_style in {"leader_runner", "rotation_runner", "structured_runner"}
    if side == "LONG":
        stop_loss = entry_price - risk_r if risk_r > 0.0 else None
        take_profit = None if runner_style else (entry_price + tp_distance if tp_distance > 0.0 else None)
    else:
        stop_loss = entry_price + risk_r if risk_r > 0.0 else None
        take_profit = None if runner_style else (entry_price - tp_distance if tp_distance > 0.0 else None)
    return Position(
        symbol=symbol,
        side=side,
        weight=weight,
        lane=lane,
        entry_price=entry_price,
        entry_bar_ts=entry_bar_ts,
        entry_bar_idx=entry_bar_idx,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_r=risk_r if risk_r > 0.0 else None,
        entry_reasons=list(entry_reasons or []),
        entry_thesis=str(entry_thesis or ""),
        expected_hold_style=hold_style,
        invalidate_on=str(invalidate_on or ""),
        structure_state="intact",
        max_price_seen=entry_price if entry_price > 0.0 else None,
        min_price_seen=entry_price if entry_price > 0.0 else None,
        mfe_r=0.0,
        mae_r=0.0,
        etd_pct=0.0,
        etd_r=0.0,
        expected_edge_pct=expected_edge_pct,
        risk_reward_ratio=round(risk_reward_ratio, 4),
    )


def maybe_arm_break_even(position: Position, price: float) -> Position:
    # Break-even disabled — coins need room to breathe through normal 15m pullbacks.
    # Trail stop handles profit protection once the move is well established.
    return position


def maybe_update_trailing(position: Position, price: float, atr: float) -> Position:
    if position.entry_price is None or position.risk_r is None or position.risk_r <= 0.0:
        return position
    if atr <= 0.0:
        return position

    if position.lane == "L1":
        arm_key, trail_mult_key = "L1_TRAIL_ARM_R", "L1_TRAIL_ATR_MULT"
    elif position.lane == "L2":
        arm_key, trail_mult_key = "L2_TRAIL_ARM_R", "L2_TRAIL_ATR_MULT"
    elif position.lane == "L4":
        arm_key, trail_mult_key = "MEME_TRAIL_ARM_R", "MEME_TRAIL_ATR_MULT"
    else:
        arm_key, trail_mult_key = "EXIT_TRAIL_ARM_R", "EXIT_TRAIL_ATR_MULT"
    arm_r = float(get_runtime_setting(arm_key))
    trail_mult = float(get_runtime_setting(trail_mult_key))
    hold_style = str(position.expected_hold_style or "").lower()
    structured_features = {
        "structure_quality": 80.0 if hold_style in {"structured_runner", "rotation_runner", "leader_runner"} else 0.0,
        "continuation_quality": 80.0 if hold_style in {"structured_runner", "rotation_runner", "leader_runner"} else 0.0,
        "ema9_above_ema26": hold_style in {"structured_runner", "rotation_runner", "leader_runner"},
        "ema_cross_distance_pct": 0.003 if hold_style in {"structured_runner", "rotation_runner", "leader_runner"} else 0.0,
        "structure_build": hold_style in {"structured_runner", "rotation_runner", "leader_runner"},
    }
    structured_hold = _structured_hold_min_minutes(position, structured_features) > 0.0
    if position.lane in {"L2", "L3"} and structured_hold:
        arm_r *= 1.15 if hold_style == "structured_runner" else 1.1
        trail_mult *= 1.35 if hold_style == "structured_runner" else 1.25
        if float(position.mfe_pct or 0.0) < (position.risk_r * arm_r / max(position.entry_price, 1e-10) * 100.0):
            return position
    if position.lane == "L1" and structured_hold:
        arm_r *= 1.35
        trail_mult *= 1.6
        if float(position.mfe_pct or 0.0) < (position.risk_r * arm_r / max(position.entry_price, 1e-10) * 100.0):
            return position

    if position.side == "LONG":
        if price - position.entry_price < position.risk_r * arm_r:
            return position
        new_stop = price - (atr * trail_mult)
        base_stop = position.trail_stop if position.trail_stop is not None else position.stop_loss
        if base_stop is None:
            next_stop = new_stop
        else:
            next_stop = max(base_stop, new_stop)
    else:
        if position.entry_price - price < position.risk_r * arm_r:
            return position
        new_stop = price + (atr * trail_mult)
        base_stop = position.trail_stop if position.trail_stop is not None else position.stop_loss
        if base_stop is None:
            next_stop = new_stop
        else:
            next_stop = min(base_stop, new_stop)

    return replace(position, trailing_armed=True, trail_stop=next_stop)


def maybe_apply_exit_posture(
    position: Position,
    *,
    price: float,
    atr: float,
    posture: ExitPostureDecision,
) -> Position:
    updated = replace(
        position,
        exit_posture=posture.posture,
        exit_posture_reason=posture.reason,
        exit_posture_confidence=posture.confidence,
    )
    if posture.posture != "TIGHTEN" or price <= 0.0:
        return updated

    tighten_key = "MEME_EXIT_POSTURE_TIGHTEN_STOP_PCT" if updated.lane == "L4" else "EXIT_POSTURE_TIGHTEN_STOP_PCT"
    tighten_pct = float(get_runtime_setting(tighten_key))
    hold_style = str(updated.expected_hold_style or "").lower()
    if updated.lane in {"L2", "L3"} and hold_style in {"structured_runner", "rotation_runner"}:
        tighten_pct *= 2.0
    if updated.lane == "L1" and hold_style == "leader_runner":
        tighten_pct *= 2.5
    if updated.side == "LONG":
        tightened_stop = price * (1.0 - tighten_pct)
        if atr > 0.0:
            atr_floor = 1.0
            if updated.lane in {"L2", "L3"} and hold_style in {"structured_runner", "rotation_runner"}:
                atr_floor = 1.8
            if updated.lane == "L1" and hold_style == "leader_runner":
                atr_floor = 2.2
            tightened_stop = max(tightened_stop, price - (atr * atr_floor))
        base_stop = updated.trail_stop if updated.trail_stop is not None else updated.stop_loss
        next_stop = tightened_stop if base_stop is None else max(base_stop, tightened_stop)
    else:
        tightened_stop = price * (1.0 + tighten_pct)
        if atr > 0.0:
            atr_floor = 1.0
            if updated.lane in {"L2", "L3"} and hold_style in {"structured_runner", "rotation_runner"}:
                atr_floor = 1.8
            if updated.lane == "L1" and hold_style == "leader_runner":
                atr_floor = 2.2
            tightened_stop = min(tightened_stop, price + (atr * atr_floor))
        base_stop = updated.trail_stop if updated.trail_stop is not None else updated.stop_loss
        next_stop = tightened_stop if base_stop is None else min(base_stop, tightened_stop)

    return replace(updated, trailing_armed=True, trail_stop=next_stop)


def review_live_exit_state(
    position: Position,
    *,
    price: float,
    hold_minutes: float,
    features: dict[str, Any],
    universe_context: dict[str, Any] | None = None,
) -> ExitPostureDecision:
    if not bool(get_runtime_setting("EXIT_LIVE_STATE_ENABLED")):
        return ExitPostureDecision("RUN", "live_state_disabled", 0.0)
    if position.entry_price is None or position.entry_price <= 0.0 or price <= 0.0:
        return ExitPostureDecision("RUN", "live_state_data_incomplete", 0.0)

    lane = str(position.lane or "L3").upper()
    pnl_pct = ((price / position.entry_price) - 1.0) * 100.0 if position.side == "LONG" else ((position.entry_price / price) - 1.0) * 100.0
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    momentum = float(features.get("momentum", momentum_5) or 0.0)
    spread_pct = float(features.get("spread_pct", 0.0) or 0.0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    stall_min_hold = float(get_runtime_setting("EXIT_LIVE_STALL_MIN_HOLD_MIN"))
    stall_max_pnl = float(get_runtime_setting("EXIT_LIVE_STALL_MAX_PNL_PCT"))
    spread_tighten_pct = float(get_runtime_setting("EXIT_LIVE_SPREAD_TIGHTEN_PCT"))
    spread_exit_pct = float(get_runtime_setting("EXIT_LIVE_SPREAD_EXIT_PCT"))
    rank_decay_min_pnl = float(get_runtime_setting("EXIT_LIVE_RANK_DECAY_MIN_PNL_PCT"))
    tighten_min_pnl = _lane_tighten_min_pnl_pct(lane)
    structured_hold_min = _structured_hold_min_minutes(position, features)
    lane_min_hold = _lane_min_hold_minutes(position)
    structure_state = str(features.get("structure_state") or compute_structure_state(position, features))
    structure_intact = structure_state == "intact"
    mfe_pct = float(position.mfe_pct or 0.0)
    etd_pct = float(position.etd_pct or 0.0)

    if lane == "L4":
        stall_min_hold = max(stall_min_hold * 0.5, 5.0)
        stall_max_pnl *= 1.3
        spread_tighten_pct *= 1.5
        spread_exit_pct *= 1.5

    context = universe_context or {}
    still_top = bool(context.get("current_symbol_is_top_candidate")) or bool(context.get("current_symbol_is_top_lane_candidate"))
    rotation_margin_score = float(get_runtime_setting("ROTATION_MARGIN_SCORE"))
    rotation_persist_rank_delta = int(get_runtime_setting("ROTATION_PERSIST_MIN_RANK_DELTA"))
    rotation_persist_lane_rank_delta = int(get_runtime_setting("ROTATION_PERSIST_MIN_LANE_RANK_DELTA"))
    rotation_persist_urgency = float(get_runtime_setting("ROTATION_PERSIST_MIN_LEADER_URGENCY"))
    current_candidate_score = float(features.get("entry_score", 0.0) or 0.0)
    top_scored = context.get("top_scored") or []
    replacement_advantage = 0.0
    replacement_persistent = False
    if top_scored and isinstance(top_scored[0], dict):
        replacement_advantage = float(top_scored[0].get("candidate_score", 0.0) or 0.0) - current_candidate_score
        replacement_persistent = (
            bool(top_scored[0].get("leader_takeover", False))
            or int(top_scored[0].get("rank_delta", 0) or 0) >= rotation_persist_rank_delta
            or int(top_scored[0].get("lane_rank_delta", 0) or 0) >= rotation_persist_lane_rank_delta
            or float(top_scored[0].get("leader_urgency", 0.0) or 0.0) >= rotation_persist_urgency
        )
    better_replacement_available = (not still_top) and replacement_advantage >= rotation_margin_score and replacement_persistent
    structured_hold_active = hold_minutes < structured_hold_min

    soft_reasons: list[str] = []
    hard_reasons: list[str] = []

    if pnl_pct >= tighten_min_pnl and momentum_5 < -0.003 and momentum_14 <= 0.0:
        soft_reasons.append("momentum_decay")
    if hold_minutes >= max(stall_min_hold, structured_hold_min, lane_min_hold) and pnl_pct <= stall_max_pnl and abs(momentum) < 0.0015 and volume_ratio < 1.05:
        soft_reasons.append("move_stall")
    if not structured_hold_active and not structure_intact and pnl_pct >= rank_decay_min_pnl and better_replacement_available:
        soft_reasons.append("rank_decay")
    if pnl_pct >= tighten_min_pnl and spread_pct >= spread_tighten_pct:
        soft_reasons.append("spread_widening")
    if (
        not structured_hold_active
        and structure_state == "broken"
        and hold_minutes >= max(20.0, lane_min_hold)
        and mfe_pct >= max(tighten_min_pnl, 1.0)
        and etd_pct >= max(tighten_min_pnl * 0.5, 1.0)
        and pnl_pct <= max(tighten_min_pnl * 0.25, 0.5)
    ):
        hard_reasons.append("failed_follow_through")
    if (
        lane == "L4"
        and hold_minutes >= 5.0
        and momentum_5 < -0.002
        and better_replacement_available
    ):
        hard_reasons.append("follow_through_lost")
    elif not structured_hold_active and not structure_intact and pnl_pct >= tighten_min_pnl and momentum_5 < -0.002 and momentum_14 <= 0.0 and better_replacement_available:
        hard_reasons.append("follow_through_lost")
    if not structured_hold_active and not structure_intact and spread_pct >= spread_exit_pct and pnl_pct >= (tighten_min_pnl + 1.0) and better_replacement_available:
        soft_reasons.append("spread_exit")

    if structure_intact and not _is_hard_failure_reason("+".join(hard_reasons)):
        hard_reasons = []
        soft_reasons = [reason for reason in soft_reasons if reason == "momentum_decay"]

    if hard_reasons or (len(soft_reasons) >= 3 and pnl_pct >= tighten_min_pnl and better_replacement_available and not structure_intact):
        reasons = hard_reasons or soft_reasons
        return ExitPostureDecision("EXIT", f"rotate_exit:{'+'.join(reasons[:3])}", 0.83)
    if len(soft_reasons) >= 2 and pnl_pct >= tighten_min_pnl:
        return ExitPostureDecision("TIGHTEN", f"live_tighten:{'+'.join(soft_reasons[:3])}", 0.74)
    return ExitPostureDecision("RUN", "live_state_ok", 0.55)


def evaluate_exit(
    position: Position,
    price: float,
    hold_minutes: float = 0.0,
    features: dict[str, Any] | None = None,
) -> str | None:
    from core.config.runtime import get_runtime_setting
    posture_reason = str(position.exit_posture_reason or "")
    stale_green_block = bool(get_runtime_setting("EXIT_STALE_GREEN_BLOCK"))
    fee_aware_green_block = bool(get_runtime_setting("EXIT_FEE_AWARE_GREEN_BLOCK"))
    green_only_stop_or_trail = bool(get_runtime_setting("EXIT_GREEN_ONLY_STOP_OR_TRAIL"))
    min_profit_after_cost_pct = _exit_profit_floor_pct()
    pnl_pct = 0.0
    if position.entry_price is not None and position.entry_price > 0.0 and price > 0.0:
        if position.side == "LONG":
            pnl_pct = ((price / position.entry_price) - 1.0) * 100.0
        else:
            pnl_pct = ((position.entry_price / price) - 1.0) * 100.0
    if position.exit_posture == "EXIT" and _is_hard_failure_reason(posture_reason):
        return f"exit_posture:{position.exit_posture_reason or 'phi3_exit'}"
    if position.exit_posture == "STALE" and _is_hard_failure_reason(posture_reason):
        return f"exit_posture:{position.exit_posture_reason or 'time_stop'}"
    stop_level = position.trail_stop if position.trail_stop is not None else position.stop_loss
    min_hold = max(
        float(get_runtime_setting("STOP_MIN_HOLD_MIN")),
        _lane_min_hold_minutes(position),
        _structured_hold_min_minutes(position, {}),
    )
    stop_armed = hold_minutes >= min_hold
    if (
        stale_green_block
        and position.exit_posture == "STALE"
        and pnl_pct > 0.0
    ):
        return None
    if (
        green_only_stop_or_trail
        and position.exit_posture in {"EXIT", "STALE"}
        and pnl_pct > 0.0
        and not _is_hard_failure_reason(posture_reason)
    ):
        return None
    if (
        position.exit_posture in {"EXIT", "STALE"}
        and pnl_pct > 0.0
        and not _is_hard_failure_reason(posture_reason)
        and _green_strength_hold_active(position, price=price, features=features)
    ):
        return None
    if (
        fee_aware_green_block
        and position.exit_posture in {"EXIT", "STALE"}
        and pnl_pct > 0.0
        and not _is_hard_failure_reason(posture_reason)
        and pnl_pct < min_profit_after_cost_pct
    ):
        return None
    if position.exit_posture in {"EXIT", "STALE"} and hold_minutes >= min_hold:
        return f"exit_posture:{position.exit_posture_reason or 'phi3_exit'}"
    # Safety exit: position never made any profit AND is underwater AND held long enough.
    # Fires regardless of exit_posture (bypasses RUN lock) to prevent indefinite holding of
    # entries that never moved in our direction.
    never_profit_min_hold = float(get_runtime_setting("EXIT_NEVER_PROFITED_MIN_HOLD_MIN"))
    if (
        float(position.mfe_pct or 0.0) <= 0.0
        and pnl_pct < 0.0
        and hold_minutes >= never_profit_min_hold
    ):
        return "stale_never_profited"
    if position.side == "LONG":
        if stop_armed and stop_level is not None and price <= stop_level:
            return "stop_loss"
        if position.take_profit is not None and price >= position.take_profit:
            return "take_profit"
    else:
        if stop_armed and stop_level is not None and price >= stop_level:
            return "stop_loss"
        if position.take_profit is not None and price <= position.take_profit:
            return "take_profit"
    return None


def build_exit_execution(
    *,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    fee_rate: float,
    bar_ts: str | None,
    bar_idx: int | None,
    reason: str,
) -> dict[str, Any]:
    notional = qty * price
    fee = notional * fee_rate
    return {
        "status": "filled",
        "symbol": symbol,
        "side": "SELL" if side == "LONG" else "BUY",
        "qty": float(qty),
        "price": float(price),
        "notional": float(notional),
        "fee": float(fee),
        "mark_price": float(price),
        "bar_ts": bar_ts,
        "bar_idx": bar_idx,
        "exit_reason": reason,
    }
