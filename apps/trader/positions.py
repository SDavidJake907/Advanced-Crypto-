"""Position management for the trader loop.

Responsibilities:
- Pure helpers: hold-time, PnL, capture-vs-MFE calculations
- Tracking state pruning (stale symbols)
- Open-position monitoring, exit evaluation, outcome recording
- handle_open_position() — the single entry point trader_loop calls per symbol

Returns a PositionAction enum so the loop knows whether to continue
to the entry pipeline or skip it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from apps.trader.logging_sink import (
    log_decision_debug,
    log_outcome_review,
    log_trace,
    now_iso,
)
from core.llm.micro_prompts import phi3_review_exit_posture
from core.memory.trade_memory import TradeMemoryStore, build_outcome_record
from core.policy.nemotron_gate import load_universe_candidate_context
from core.risk.exits import build_exit_plan, evaluate_exit
from core.risk.position_monitor import monitor_open_position
from core.risk.portfolio import Position, PositionState
from core.state.portfolio import PortfolioState
from core.state.position_state_store import save_position_state
from core.state.trace import DecisionTrace


# Approximate round-trip fee cost (entry + exit taker fees on Kraken).
# Used to compute realized_edge_pct = pnl_pct - fees.
_FEE_ROUND_TRIP_PCT = 0.0063   # ~0.26% × 2 sides + slippage buffer


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class PositionAction(Enum):
    PROCEED    = "proceed"    # no open position — fall through to entry pipeline
    HOLDING    = "holding"    # position open, no exit triggered — skip entry
    EXITED     = "exited"     # position closed this bar — skip entry


# ---------------------------------------------------------------------------
# Replacement helper
# ---------------------------------------------------------------------------

def execute_replacement_exit(
    *,
    target_symbol: str,
    replacement_symbol: str,
    executor: Any,
    portfolio: PortfolioState,
    positions_state: PositionState,
    last_exit_ts: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    position = positions_state.get(target_symbol)
    live_qty = float(portfolio.positions.get(target_symbol, 0.0) or 0.0)
    mark_price = float(
        portfolio.position_marks.get(target_symbol, 0.0)
        or (position.entry_price if position is not None else 0.0)
        or 0.0
    )
    if position is None or live_qty <= 0.0 or mark_price <= 0.0:
        return (
            {"status": "blocked", "reason": "replacement_target_unavailable", "target_symbol": target_symbol},
            {},
        )

    exec_result = executor.execute_exit(
        symbol=target_symbol,
        side=position.side,
        qty=abs(live_qty),
        price=mark_price,
        features={"price": mark_price},
        exit_reason=f"portfolio_replace:{replacement_symbol}",
    )
    state_change: dict[str, Any] = {}
    if exec_result.get("status") == "filled":
        state_change = portfolio.apply_execution(exec_result)
        positions_state.remove(target_symbol)
        save_position_state(positions_state)
        last_exit_ts[target_symbol] = time.time()
    return exec_result, state_change


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def compute_hold_minutes(entry_bar_ts: str | None, exit_bar_ts: str | None) -> float:
    if not entry_bar_ts:
        return 0.0
    try:
        start = pd.Timestamp(entry_bar_ts)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        end = pd.Timestamp(exit_bar_ts) if exit_bar_ts else pd.Timestamp.now(tz="UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        return max((end - start).total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0


def compute_pnl_pct(side: str, entry_price: float | None, exit_price: float) -> float:
    if entry_price is None or entry_price <= 0.0 or exit_price <= 0.0:
        return 0.0
    if side == "LONG":
        return (exit_price / entry_price) - 1.0
    return (entry_price / exit_price) - 1.0


def compute_capture_vs_mfe(pnl_pct: float, mfe_pct_raw: float) -> float:
    """How much of the max favorable excursion was captured (0.0–1.0+)."""
    mfe = mfe_pct_raw / 100.0
    if mfe <= 0.0:
        return 0.0
    return pnl_pct / mfe


# ---------------------------------------------------------------------------
# Tracking state helpers
# ---------------------------------------------------------------------------

def prune_symbol_tracking_state(
    current_symbols: list[str],
    positions_state: PositionState,
    *,
    last_prices: dict[str, float],
    last_bar_keys: dict[str, str],
    last_exit_ts: dict[str, float],
    retention_sec: float,
    additional_caches: tuple[dict, ...] | None = None,
) -> None:
    allowed = {str(s).strip().upper() for s in current_symbols if str(s).strip()}
    allowed.update(p.symbol for p in positions_state.all())
    for mapping in (last_prices, last_bar_keys):
        for sym in list(mapping.keys()):
            if sym not in allowed:
                mapping.pop(sym, None)
    if additional_caches:
        for mapping in additional_caches:
            for sym in list(mapping.keys()):
                if sym not in allowed:
                    mapping.pop(sym, None)
    cutoff = time.time() - max(retention_sec, 0.0)
    for sym in list(last_exit_ts.keys()):
        if sym in allowed:
            continue
        if float(last_exit_ts.get(sym, 0.0) or 0.0) < cutoff:
            last_exit_ts.pop(sym, None)


# ---------------------------------------------------------------------------
# Exit payload builders (reduce inline duplication)
# ---------------------------------------------------------------------------

def _build_position_debug_fields(
    position: Position,
    features: dict[str, Any],
    hold_minutes: float,
    live_exit_posture: Any,
) -> dict[str, Any]:
    return {
        "price": features.get("price"),
        "atr": features.get("atr"),
        "entry_price": position.entry_price,
        "hold_minutes": hold_minutes,
        "stop_loss": position.stop_loss,
        "take_profit": position.take_profit,
        "trail_stop": position.trail_stop,
        "exit_posture": position.exit_posture,
        "exit_posture_reason": position.exit_posture_reason,
        "exit_posture_confidence": position.exit_posture_confidence,
        "live_exit_posture": live_exit_posture.posture,
        "live_exit_posture_reason": live_exit_posture.reason,
        "live_exit_posture_confidence": live_exit_posture.confidence,
        "entry_thesis": position.entry_thesis,
        "expected_hold_style": position.expected_hold_style,
        "invalidate_on": position.invalidate_on,
        "structure_state": position.structure_state,
        "mfe_pct": position.mfe_pct,
        "mae_pct": position.mae_pct,
        "mfe_r": position.mfe_r,
        "mae_r": position.mae_r,
        "etd_pct": position.etd_pct,
        "etd_r": position.etd_r,
    }


def _build_outcome_fields(
    position: Position,
    pnl_pct: float,
    hold_minutes: float,
    exit_reason: str,
    features: dict[str, Any],
) -> dict[str, Any]:
    capture = compute_capture_vs_mfe(pnl_pct, float(position.mfe_pct or 0.0))
    return {
        "symbol": position.symbol,
        "lane": features.get("lane"),
        "side": position.side,
        "pnl_pct": pnl_pct,
        "hold_minutes": hold_minutes,
        "exit_reason": exit_reason,
        "exit_posture": position.exit_posture,
        "exit_posture_reason": position.exit_posture_reason,
        "exit_posture_confidence": position.exit_posture_confidence,
        "entry_thesis": position.entry_thesis,
        "expected_hold_style": position.expected_hold_style,
        "invalidate_on": position.invalidate_on,
        "mfe_pct": position.mfe_pct,
        "mae_pct": position.mae_pct,
        "capture_vs_mfe_pct": capture,
        "structure_state": position.structure_state,
        "mfe_r": position.mfe_r,
        "mae_r": position.mae_r,
        "etd_pct": position.etd_pct,
        "etd_r": position.etd_r,
    }


def _deterministic_outcome_review(outcome: dict[str, Any]) -> dict[str, Any]:
    pnl_pct = float(outcome.get("pnl_pct", 0.0) or 0.0)
    exit_reason = str(outcome.get("exit_reason", "unknown") or "unknown")
    capture = float(outcome.get("capture_vs_mfe_pct", 0.0) or 0.0)
    structure_state = str(outcome.get("structure_state", "") or "").lower()

    if pnl_pct > 0.02 and capture >= 0.5:
        return {
            "outcome_class": "good_breakout",
            "lesson": "winner_held_well",
            "suggested_adjustment": "keep_current_posture",
            "confidence": 0.75,
        }
    if "stop" in exit_reason or structure_state == "broken":
        return {
            "outcome_class": "chop_fakeout" if pnl_pct >= -0.01 else "weak_follow_through",
            "lesson": "stopped_out_early" if pnl_pct >= -0.01 else "entry_lacked_follow_through",
            "suggested_adjustment": "review_entry_timing" if pnl_pct >= -0.01 else "tighten_promotion",
            "confidence": 0.7 if pnl_pct >= -0.01 else 0.65,
        }
    if pnl_pct < 0:
        return {
            "outcome_class": "weak_follow_through",
            "lesson": "entry_lacked_follow_through",
            "suggested_adjustment": "tighten_promotion",
            "confidence": 0.65,
        }
    return {
        "outcome_class": "normal_exit",
        "lesson": "neutral_outcome",
        "suggested_adjustment": "no_change",
        "confidence": 0.5,
    }


# ---------------------------------------------------------------------------
# Core open-position handler
# ---------------------------------------------------------------------------

def handle_open_position(
    symbol: str,
    features: dict[str, Any],
    existing_position: Position,
    live_qty: float,
    *,
    executor: Any,
    portfolio: PortfolioState,
    positions_state: PositionState,
    memory_store: TradeMemoryStore,
    last_exit_ts: dict[str, float],
) -> PositionAction:
    """Monitor an open position, execute exits, and record outcomes.

    Returns:
        EXITED  — position closed, loop should skip entry pipeline and update last_prices
        HOLDING — position open, no exit, loop should skip entry pipeline
    """
    features = dict(features)
    features["universe_lane"] = features.get("universe_lane") or features.get("lane")
    features["lane"] = existing_position.lane

    price        = float(features.get("price", 0.0))
    atr          = float(features.get("atr", 0.0))
    hold_minutes = compute_hold_minutes(existing_position.entry_bar_ts, features.get("bar_ts"))

    # Phi-3 exit posture review
    posture = phi3_review_exit_posture({
        "symbol": symbol,
        "lane": existing_position.lane,
        "side": existing_position.side,
        "price": price,
        "entry_price": float(existing_position.entry_price or 0.0),
        "pnl_pct": compute_pnl_pct(existing_position.side, existing_position.entry_price, price) * 100.0,
        "hold_minutes": hold_minutes,
        "atr": atr,
        "rsi": float(features.get("rsi", 50.0)),
        "momentum": float(features.get("momentum", 0.0)),
        "momentum_5": float(features.get("momentum_5", 0.0)),
        "momentum_14": float(features.get("momentum_14", 0.0)),
        "trend_1h": int(features.get("trend_1h", 0)),
        "regime_7d": str(features.get("regime_7d", "unknown")),
        "macro_30d": str(features.get("macro_30d", "unknown")),
        "entry_thesis": existing_position.entry_thesis,
        "expected_hold_style": existing_position.expected_hold_style,
        "invalidate_on": existing_position.invalidate_on,
    })

    # Position monitor (updates MFE/MAE/trail)
    monitor_result   = monitor_open_position(
        existing_position,
        price=price,
        atr=atr,
        hold_minutes=hold_minutes,
        features=features,
        phi3_posture=posture,
        universe_context=load_universe_candidate_context(symbol),
    )
    updated_position  = monitor_result.position
    live_exit_posture = monitor_result.live_posture
    positions_state.add_or_update(updated_position)
    save_position_state(positions_state)

    exit_reason = evaluate_exit(updated_position, price, hold_minutes=hold_minutes, features=features)

    if exit_reason:
        exec_result = executor.execute_exit(
            symbol=symbol,
            side=updated_position.side,
            qty=abs(live_qty),
            price=price,
            features=features,
            exit_reason=exit_reason,
        )
        exec_result["bar_ts"]  = features.get("bar_ts")
        exec_result["bar_idx"] = features.get("bar_idx")
        state_change = portfolio.apply_execution(exec_result)

        if exec_result.get("status") == "filled":
            last_exit_ts[symbol] = time.time()
            positions_state.remove(symbol)
            save_position_state(positions_state)

            pnl_pct      = compute_pnl_pct(updated_position.side, updated_position.entry_price, float(exec_result["price"]))
            pnl_usd      = pnl_pct * abs(live_qty) * float(updated_position.entry_price or 0.0)
            hold_minutes = compute_hold_minutes(updated_position.entry_bar_ts, features.get("bar_ts"))
            capture      = compute_capture_vs_mfe(pnl_pct, float(updated_position.mfe_pct or 0.0))

            _expected_edge = float(updated_position.expected_edge_pct or 0.0)
            _realized_edge = pnl_pct - _FEE_ROUND_TRIP_PCT
            _edge_capture  = (_realized_edge / _expected_edge) if _expected_edge > 0.0 else 0.0

            memory_store.append_outcome(build_outcome_record(
                symbol=symbol,
                side=updated_position.side,
                lane=updated_position.lane,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                hold_minutes=hold_minutes,
                exit_reason=exit_reason,
                entry_reasons=list(updated_position.entry_reasons),
                regime_label=str(features.get("regime_7d", "unknown")),
                entry_score=float(features.get("entry_score", 0.0) or 0.0),
                entry_recommendation=str(features.get("entry_recommendation", "") or ""),
                pattern_name=str(((features.get("pattern_candidate") or {}).get("pattern", "")) or ""),
                mfe_pct=float(updated_position.mfe_pct or 0.0) / 100.0,
                mae_pct=float(updated_position.mae_pct or 0.0) / 100.0,
                capture_vs_mfe_pct=capture,
                structure_state=updated_position.structure_state,
                mfe_r=float(updated_position.mfe_r or 0.0),
                mae_r=float(updated_position.mae_r or 0.0),
                etd_pct=float(updated_position.etd_pct or 0.0) / 100.0,
                etd_r=float(updated_position.etd_r or 0.0),
                expected_edge_pct=_expected_edge,
                realized_edge_pct=_realized_edge,
                edge_capture_ratio=_edge_capture,
            ))

            outcome_fields = _build_outcome_fields(updated_position, pnl_pct, hold_minutes, exit_reason, features)
            outcome_review = _deterministic_outcome_review({
                **outcome_fields,
                "entry_reasons": list(updated_position.entry_reasons),
                "entry_score": features.get("entry_score"),
                "entry_recommendation": features.get("entry_recommendation"),
                "reversal_risk": features.get("reversal_risk"),
            })

            log_outcome_review({"ts": now_iso(), **outcome_fields, "review": outcome_review})

            memory_store.save_lesson(
                symbol=symbol,
                outcome_class=outcome_review.get("outcome_class", "normal_exit"),
                lesson=outcome_review.get("lesson", ""),
                suggested_adjustment=outcome_review.get("suggested_adjustment", ""),
                confidence=float(outcome_review.get("confidence", 0.5)),
            )

        log_trace(DecisionTrace(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            features=features,
            signal="EXIT",
            risk_checks=[exit_reason],
            execution=exec_result,
            state_change=state_change,
        ))
        log_decision_debug({
            "ts": now_iso(),
            "symbol": symbol,
            "lane": features.get("lane"),
            "universe_lane": features.get("universe_lane"),
            "phase": "exit",
            "execution_status": exec_result.get("status"),
            "exit_reason": exit_reason,
            **_build_position_debug_fields(updated_position, features, hold_minutes, live_exit_posture),
        })
        return PositionAction.EXITED

    # No exit — log hold state and skip entry pipeline
    log_decision_debug({
        "ts": now_iso(),
        "symbol": symbol,
        "lane": updated_position.lane,
        "phase": "hold_manager",
        **_build_position_debug_fields(updated_position, features, hold_minutes, live_exit_posture),
    })
    return PositionAction.HOLDING
