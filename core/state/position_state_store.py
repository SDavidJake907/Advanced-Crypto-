from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.risk.portfolio import Position, PositionState


POSITION_STATE_PATH = Path("logs/position_state.json")


def _sanitize_excursion_state(position: Position) -> Position:
    risk_r = float(position.risk_r or 0.0)
    if risk_r <= 0.0:
        return position
    absurd_r = any(
        abs(float(value or 0.0)) >= 1000.0
        for value in (position.mfe_r, position.mae_r, position.etd_r)
    )
    if not absurd_r:
        return position
    return Position(
        symbol=position.symbol,
        side=position.side,
        weight=position.weight,
        lane=position.lane,
        entry_price=position.entry_price,
        entry_bar_ts=position.entry_bar_ts,
        entry_bar_idx=position.entry_bar_idx,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        risk_r=position.risk_r,
        break_even_armed=position.break_even_armed,
        trailing_armed=position.trailing_armed,
        trail_stop=position.trail_stop,
        entry_reasons=position.entry_reasons,
        entry_thesis=position.entry_thesis,
        expected_hold_style=position.expected_hold_style,
        invalidate_on=position.invalidate_on,
        monitor_state=position.monitor_state,
        monitor_reason=position.monitor_reason,
        monitor_confidence=position.monitor_confidence,
        exit_posture=position.exit_posture,
        exit_posture_reason=position.exit_posture_reason,
        exit_posture_confidence=position.exit_posture_confidence,
        structure_state=position.structure_state,
        max_price_seen=position.entry_price,
        min_price_seen=position.entry_price,
        mfe_pct=0.0,
        mae_pct=0.0,
        mfe_ts="",
        mae_ts="",
        mfe_r=0.0,
        mae_r=0.0,
        etd_pct=0.0,
        etd_r=0.0,
        expected_edge_pct=position.expected_edge_pct,
        risk_reward_ratio=position.risk_reward_ratio,
    )


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
            state.add_or_update(_sanitize_excursion_state(Position(**item)))
        except TypeError:
            continue
    return state


def save_position_state(state: PositionState) -> None:
    POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"positions": [asdict(position) for position in state.all()]}
    tmp_path = POSITION_STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(POSITION_STATE_PATH)


def _prefer_persisted_float(synced_value: float | None, persisted_value: float | None) -> float | None:
    synced = float(synced_value or 0.0)
    persisted = float(persisted_value or 0.0)
    if synced != 0.0:
        return synced_value
    if persisted != 0.0:
        return persisted_value
    return synced_value if synced_value is not None else persisted_value


def _prefer_persisted_ts(synced_value: str | None, persisted_value: str | None) -> str:
    synced = str(synced_value or "").strip()
    if synced:
        return synced
    persisted = str(persisted_value or "").strip()
    return persisted


def _prefer_persisted_price(synced_value: float | None, persisted_value: float | None, fallback_entry: float | None) -> float | None:
    if synced_value is not None and float(synced_value or 0.0) > 0.0:
        return synced_value
    if persisted_value is not None and float(persisted_value or 0.0) > 0.0:
        return persisted_value
    return fallback_entry


def merge_persisted_positions(synced: PositionState, persisted: PositionState) -> PositionState:
    merged = PositionState()
    for synced_position in synced.all():
        persisted_position = persisted.get(synced_position.symbol)
        if persisted_position is None:
            merged.add_or_update(_sanitize_excursion_state(synced_position))
            continue
        synced_risk = synced_position.risk_r
        persisted_risk = persisted_position.risk_r
        risk_profile_changed = False
        if synced_risk is not None:
            if persisted_risk is None or persisted_risk <= 0.0:
                risk_profile_changed = True
            else:
                ratio = abs(float(synced_risk) - float(persisted_risk)) / max(abs(float(persisted_risk)), 1e-12)
                risk_profile_changed = ratio >= 0.25
        merged.add_or_update(
            _sanitize_excursion_state(
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
                stop_loss=synced_position.stop_loss if synced_position.stop_loss is not None else persisted_position.stop_loss,
                take_profit=synced_position.take_profit if synced_position.take_profit is not None else persisted_position.take_profit,
                risk_r=synced_position.risk_r if synced_position.risk_r is not None else persisted_position.risk_r,
                break_even_armed=persisted_position.break_even_armed,
                trailing_armed=persisted_position.trailing_armed,
                trail_stop=synced_position.trail_stop if synced_position.trail_stop is not None else persisted_position.trail_stop,
                entry_reasons=synced_position.entry_reasons or persisted_position.entry_reasons,
                entry_thesis=synced_position.entry_thesis or persisted_position.entry_thesis,
                expected_hold_style=synced_position.expected_hold_style or persisted_position.expected_hold_style,
                invalidate_on=synced_position.invalidate_on or persisted_position.invalidate_on,
                monitor_state=persisted_position.monitor_state,
                monitor_reason=persisted_position.monitor_reason,
                monitor_confidence=persisted_position.monitor_confidence,
                exit_posture=persisted_position.exit_posture,
                exit_posture_reason=persisted_position.exit_posture_reason,
                exit_posture_confidence=persisted_position.exit_posture_confidence,
                structure_state=persisted_position.structure_state,
                max_price_seen=(
                    synced_position.entry_price
                    if risk_profile_changed
                    else _prefer_persisted_price(
                        synced_position.max_price_seen,
                        persisted_position.max_price_seen,
                        synced_position.entry_price or persisted_position.entry_price,
                    )
                ),
                min_price_seen=(
                    synced_position.entry_price
                    if risk_profile_changed
                    else _prefer_persisted_price(
                        synced_position.min_price_seen,
                        persisted_position.min_price_seen,
                        synced_position.entry_price or persisted_position.entry_price,
                    )
                ),
                mfe_pct=0.0 if risk_profile_changed else float(_prefer_persisted_float(synced_position.mfe_pct, persisted_position.mfe_pct) or 0.0),
                mae_pct=0.0 if risk_profile_changed else float(_prefer_persisted_float(synced_position.mae_pct, persisted_position.mae_pct) or 0.0),
                mfe_ts="" if risk_profile_changed else _prefer_persisted_ts(synced_position.mfe_ts, persisted_position.mfe_ts),
                mae_ts="" if risk_profile_changed else _prefer_persisted_ts(synced_position.mae_ts, persisted_position.mae_ts),
                mfe_r=0.0 if risk_profile_changed else float(_prefer_persisted_float(synced_position.mfe_r, persisted_position.mfe_r) or 0.0),
                mae_r=0.0 if risk_profile_changed else float(_prefer_persisted_float(synced_position.mae_r, persisted_position.mae_r) or 0.0),
                etd_pct=0.0 if risk_profile_changed else float(_prefer_persisted_float(synced_position.etd_pct, persisted_position.etd_pct) or 0.0),
                etd_r=0.0 if risk_profile_changed else float(_prefer_persisted_float(synced_position.etd_r, persisted_position.etd_r) or 0.0),
                expected_edge_pct=(
                    synced_position.expected_edge_pct
                    if float(synced_position.expected_edge_pct or 0.0) != 0.0
                    else persisted_position.expected_edge_pct
                ),
                risk_reward_ratio=(
                    synced_position.risk_reward_ratio
                    if float(synced_position.risk_reward_ratio or 0.0) != 0.0
                    else persisted_position.risk_reward_ratio
                ),
                )
            )
        )
    return merged
