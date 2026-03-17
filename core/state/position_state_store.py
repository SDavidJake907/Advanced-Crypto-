from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.risk.portfolio import Position, PositionState


POSITION_STATE_PATH = Path("logs/position_state.json")


def load_position_state() -> PositionState:
    state = PositionState()
    if not POSITION_STATE_PATH.exists():
        return state
    try:
        payload = json.loads(POSITION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return state
    positions = payload.get("positions", [])
    if not isinstance(positions, list):
        return state
    for item in positions:
        if not isinstance(item, dict):
            continue
        try:
            state.add_or_update(Position(**item))
        except TypeError:
            continue
    return state


def save_position_state(state: PositionState) -> None:
    POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"positions": [asdict(position) for position in state.all()]}
    tmp_path = POSITION_STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(POSITION_STATE_PATH)


def merge_persisted_positions(synced: PositionState, persisted: PositionState) -> PositionState:
    merged = PositionState()
    for synced_position in synced.all():
        persisted_position = persisted.get(synced_position.symbol)
        if persisted_position is None:
            merged.add_or_update(synced_position)
            continue
        merged.add_or_update(
            Position(
                symbol=synced_position.symbol,
                side=synced_position.side,
                weight=synced_position.weight,
                lane=synced_position.lane or persisted_position.lane,
                entry_price=synced_position.entry_price or persisted_position.entry_price,
                entry_bar_ts=synced_position.entry_bar_ts or persisted_position.entry_bar_ts,
                entry_bar_idx=synced_position.entry_bar_idx
                if synced_position.entry_bar_idx is not None
                else persisted_position.entry_bar_idx,
                stop_loss=persisted_position.stop_loss,
                take_profit=persisted_position.take_profit,
                risk_r=persisted_position.risk_r,
                break_even_armed=persisted_position.break_even_armed,
                trailing_armed=persisted_position.trailing_armed,
                trail_stop=persisted_position.trail_stop,
                entry_reasons=synced_position.entry_reasons or persisted_position.entry_reasons,
                entry_thesis=synced_position.entry_thesis or persisted_position.entry_thesis,
                expected_hold_style=synced_position.expected_hold_style or persisted_position.expected_hold_style,
                invalidate_on=synced_position.invalidate_on or persisted_position.invalidate_on,
                exit_posture=persisted_position.exit_posture,
                exit_posture_reason=persisted_position.exit_posture_reason,
                exit_posture_confidence=persisted_position.exit_posture_confidence,
            )
        )
    return merged
