from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.config.runtime import get_runtime_setting, get_symbol_lane
from core.llm.phi3_exit_posture import ExitPostureDecision
from core.risk.portfolio import Position


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
) -> Position:
    lane = lane or get_symbol_lane(symbol)
    if lane == "L4":
        stop_mult = float(get_runtime_setting("MEME_EXIT_ATR_STOP_MULT"))
        tp_mult = float(get_runtime_setting("MEME_EXIT_ATR_TAKE_PROFIT_MULT"))
    else:
        stop_mult = float(get_runtime_setting("EXIT_ATR_STOP_MULT"))
        tp_mult = float(get_runtime_setting("EXIT_ATR_TAKE_PROFIT_MULT"))
    risk_r = max(atr * stop_mult, 0.0)
    if side == "LONG":
        stop_loss = entry_price - risk_r if risk_r > 0.0 else None
        take_profit = entry_price + (atr * tp_mult) if atr > 0.0 else None
    else:
        stop_loss = entry_price + risk_r if risk_r > 0.0 else None
        take_profit = entry_price - (atr * tp_mult) if atr > 0.0 else None
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
        expected_hold_style=str(expected_hold_style or ""),
        invalidate_on=str(invalidate_on or ""),
    )


def maybe_arm_break_even(position: Position, price: float) -> Position:
    if position.entry_price is None or position.risk_r is None or position.risk_r <= 0.0:
        return position
    if position.break_even_armed:
        return position
    trigger_key = "MEME_BREAK_EVEN_R" if position.lane == "L4" else "EXIT_BREAK_EVEN_R"
    trigger_r = float(get_runtime_setting(trigger_key))
    if position.side == "LONG":
        if price - position.entry_price >= position.risk_r * trigger_r:
            return replace(position, break_even_armed=True, stop_loss=position.entry_price)
    else:
        if position.entry_price - price >= position.risk_r * trigger_r:
            return replace(position, break_even_armed=True, stop_loss=position.entry_price)
    return position


def maybe_update_trailing(position: Position, price: float, atr: float) -> Position:
    if position.entry_price is None or position.risk_r is None or position.risk_r <= 0.0:
        return position
    if atr <= 0.0:
        return position

    arm_key = "MEME_TRAIL_ARM_R" if position.lane == "L4" else "EXIT_TRAIL_ARM_R"
    trail_mult_key = "MEME_TRAIL_ATR_MULT" if position.lane == "L4" else "EXIT_TRAIL_ATR_MULT"
    arm_r = float(get_runtime_setting(arm_key))
    trail_mult = float(get_runtime_setting(trail_mult_key))

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
    if updated.side == "LONG":
        tightened_stop = price * (1.0 - tighten_pct)
        if atr > 0.0:
            tightened_stop = max(tightened_stop, price - atr)
        base_stop = updated.trail_stop if updated.trail_stop is not None else updated.stop_loss
        next_stop = tightened_stop if base_stop is None else max(base_stop, tightened_stop)
    else:
        tightened_stop = price * (1.0 + tighten_pct)
        if atr > 0.0:
            tightened_stop = min(tightened_stop, price + atr)
        base_stop = updated.trail_stop if updated.trail_stop is not None else updated.stop_loss
        next_stop = tightened_stop if base_stop is None else min(base_stop, tightened_stop)

    return replace(updated, trailing_armed=True, trail_stop=next_stop)


def evaluate_exit(position: Position, price: float) -> str | None:
    if position.exit_posture == "EXIT":
        return f"exit_posture:{position.exit_posture_reason or 'phi3_exit'}"
    if position.exit_posture == "STALE":
        return f"exit_posture:{position.exit_posture_reason or 'time_stop'}"
    stop_level = position.trail_stop if position.trail_stop is not None else position.stop_loss
    if position.side == "LONG":
        if stop_level is not None and price <= stop_level:
            return "stop_loss"
        if position.take_profit is not None and price >= position.take_profit:
            return "take_profit"
    else:
        if stop_level is not None and price >= stop_level:
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
