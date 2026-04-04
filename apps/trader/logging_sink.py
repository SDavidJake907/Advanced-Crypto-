"""Logging sink for the trader loop.

All JSONL and SQLite writes go through this module. Nothing else in the
trader should open log files directly — route through these functions so
the write path is in one place and easy to make async later (Phase 2).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from core.runtime.alerts import emit_alert
from core.runtime.log_rotation import rotate_jsonl_if_needed
from core.state.system_record import (
    record_decision_debug as _record_debug_db,
    record_decision_trace as _record_trace_db,
    record_fill_event as _record_fill_db,
    record_outcome_review as _record_outcome_db,
    record_shadow_decision as _record_shadow_db,
)
from core.state.trace import DecisionTrace

# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------

TRACE_LOG_PATH         = Path("logs/decision_traces.jsonl")
DEBUG_LOG_PATH         = Path("logs/decision_debug.jsonl")
ACCOUNT_SYNC_LOG_PATH  = Path("logs/account_sync.json")
OUTCOME_REVIEW_LOG_PATH = Path("logs/outcome_reviews.jsonl")
SHADOW_LOG_PATH        = Path("logs/shadow_decisions.jsonl")
NEMO_RECOMMENDATIONS_LOG_PATH = Path("logs/nemo_recommendations.jsonl")

# ---------------------------------------------------------------------------
# Feature flags (read once at import; cheap to re-read if needed)
# ---------------------------------------------------------------------------

TRACE_LOG_ENABLED    = os.getenv("TRACE_LOG_ENABLED", "false").lower() == "true"
DECISION_DEBUG_ENABLED = os.getenv("DECISION_DEBUG_ENABLED", "true").lower() == "true"
CONSOLE_EVENT_LOG    = os.getenv("CONSOLE_EVENT_LOG", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

_log_lock = threading.Lock()

def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        rotate_jsonl_if_needed(path)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")


# ---------------------------------------------------------------------------
# Public write functions
# ---------------------------------------------------------------------------

def log_trace(trace: DecisionTrace) -> None:
    trace_payload = trace.to_dict()
    if TRACE_LOG_ENABLED:
        _append_jsonl(TRACE_LOG_PATH, trace_payload)
    _record_trace_db(trace_payload)
    # Also record fill event when a trade was executed
    execution = trace_payload.get("execution", {})
    if isinstance(execution, dict) and str(execution.get("status", "")).lower() == "filled":
        features = trace_payload.get("features", {})
        lane = str(features.get("lane", "")) if isinstance(features, dict) else ""
        _record_fill_db(execution, lane=lane, ts=str(trace_payload.get("timestamp") or now_iso()))


def log_decision_debug(payload: dict) -> None:
    if DECISION_DEBUG_ENABLED:
        _append_jsonl(DEBUG_LOG_PATH, payload)
    _record_debug_db(payload)
    _maybe_log_nemo_recommendation(payload)
    _maybe_emit_alert_from_decision(payload)


def log_account_sync(payload: dict) -> None:
    ACCOUNT_SYNC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNT_SYNC_LOG_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _maybe_emit_alert_from_account_sync(payload)


def log_outcome_review(payload: dict) -> None:
    _append_jsonl(OUTCOME_REVIEW_LOG_PATH, payload)
    _record_outcome_db(payload)


def log_shadow_decision(payload: dict) -> None:
    _append_jsonl(SHADOW_LOG_PATH, payload)
    _record_shadow_db(payload)


def console_event(payload: dict, *, force: bool = False) -> None:
    if force or CONSOLE_EVENT_LOG:
        print(payload)


def _maybe_log_nemo_recommendation(payload: dict) -> None:
    if not isinstance(payload, dict):
        return
    symbol = str(payload.get("symbol", "") or "").strip()
    nemotron = payload.get("nemotron", {})
    if not symbol or not isinstance(nemotron, dict):
        return
    action = str(nemotron.get("action", "") or "").upper()
    reason = str(nemotron.get("reason", "") or "").strip()
    if action not in {"OPEN", "HOLD", "CLOSE"}:
        return
    recommendation = {
        "ts": str(payload.get("ts") or now_iso()),
        "symbol": symbol,
        "lane": str(payload.get("lane", "") or ""),
        "scope": "entry",
        "action": action,
        "reason": reason,
        "size_factor_hint": float(nemotron.get("size_factor_hint", 1.0) or 1.0),
        "entry_score": float(payload.get("entry_score", 0.0) or 0.0),
        "entry_recommendation": str(payload.get("entry_recommendation", "") or ""),
        "reversal_risk": str(payload.get("reversal_risk", "") or ""),
        "signal": str(payload.get("signal", "") or ""),
        "execution_status": str(payload.get("execution_status", "") or ""),
    }
    debug = nemotron.get("debug", {})
    if isinstance(debug, dict):
        batch_reasoning = str(debug.get("batch_reasoning", "") or "").strip()
        if batch_reasoning:
            recommendation["batch_reasoning"] = batch_reasoning
    _append_jsonl(NEMO_RECOMMENDATIONS_LOG_PATH, recommendation)


def _maybe_emit_alert_from_decision(payload: dict) -> None:
    if not isinstance(payload, dict):
        return
    symbol = str(payload.get("symbol", "") or "")
    loop = str(payload.get("loop", "") or "")
    phase = str(payload.get("phase", "") or "")
    execution_status = str(payload.get("execution_status", "") or "").lower()
    execution_reason = str(payload.get("execution_reason", "") or "").strip()
    runtime_health = payload.get("runtime_health", {}) if isinstance(payload.get("runtime_health", {}), dict) else {}
    nemotron = payload.get("nemotron", {}) if isinstance(payload.get("nemotron", {}), dict) else {}
    nemotron_reason = str(nemotron.get("reason", "") or "").strip()
    batch_reasoning = ""
    debug = nemotron.get("debug", {})
    if isinstance(debug, dict):
        batch_reasoning = str(debug.get("batch_reasoning", "") or "").strip()

    if loop == "missing_ohlc":
        emit_alert(
            level="warning",
            source="trader",
            code="missing_ohlc",
            message="Missing OHLC data for one or more active symbols",
            details={
                "missing_symbols": payload.get("missing_symbols", []),
                "total_eval": payload.get("total_eval"),
                "total_valid": payload.get("total_valid"),
            },
            dedupe_key=f"missing_ohlc:{','.join(sorted(payload.get('missing_symbols', [])))}",
        )

    if phase == "feature_warmup":
        emit_alert(
            level="warning",
            source="trader",
            code="feature_warmup",
            message=f"{symbol or 'symbol'} is missing required features",
            details={
                "symbol": symbol,
                "reason": payload.get("reason"),
                "feature_status": payload.get("feature_status"),
                "history_points": payload.get("history_points"),
            },
            dedupe_key=f"feature_warmup:{symbol}",
        )

    if nemotron_reason.startswith("batch_parse_fallback"):
        emit_alert(
            level="error",
            source="nemotron",
            code="batch_parse_fallback",
            message="Nemotron batch response fell back due to parse failure",
            details={
                "symbol": symbol,
                "reason": nemotron_reason,
                "batch_reasoning": batch_reasoning,
            },
            dedupe_key=f"nemotron_batch_parse:{symbol}:{nemotron_reason}",
        )

    if execution_status in {"blocked", "rejected"}:
        emit_alert(
            level="error" if execution_status == "rejected" else "warning",
            source="execution",
            code=f"execution_{execution_status}",
            message=f"{symbol or 'order'} {execution_status}",
            details={
                "symbol": symbol,
                "reason": execution_reason,
                "order_type": payload.get("order_type"),
                "signal": payload.get("signal"),
            },
            dedupe_key=f"execution:{execution_status}:{symbol}:{execution_reason}",
        )

    if isinstance(runtime_health, dict) and runtime_health and not bool(runtime_health.get("ok", True)):
        emit_alert(
            level="error",
            source="runtime_health",
            code="runtime_unhealthy",
            message="Runtime health gate reports unhealthy dependencies",
            details={
                "symbol": symbol,
                "reasons": runtime_health.get("reasons", []),
                "details": runtime_health.get("details", {}),
            },
            dedupe_key="runtime_unhealthy",
        )


def _maybe_emit_alert_from_account_sync(payload: dict) -> None:
    if not isinstance(payload, dict):
        return
    status = str(payload.get("status", "") or "").lower()
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics", {}), dict) else {}
    if status == "sync_failed":
        emit_alert(
            level="error",
            source="account_sync",
            code="sync_failed",
            message="Account sync failed",
            details={"error": payload.get("error", "")},
            dedupe_key="account_sync_failed",
        )
    elif status == "degraded":
        emit_alert(
            level="warning",
            source="account_sync",
            code="sync_degraded",
            message="Account sync degraded; using last known-good snapshot",
            details={
                "error": payload.get("error", ""),
                "last_success_ts": payload.get("last_success_ts", ""),
            },
            dedupe_key="account_sync_degraded",
        )
    trades_history_source = str(diagnostics.get("trades_history_source", "") or "").strip().lower()
    trades_history_error = str(diagnostics.get("trades_history_error", "") or "").strip()
    if trades_history_error and trades_history_source != "disabled_ledger_only":
        invalid_key = "EAPI:Invalid key" in trades_history_error
        emit_alert(
            level="info" if invalid_key else "warning",
            source="account_sync",
            code="trades_history_invalid_key" if invalid_key else "trades_history_fallback",
            message=(
                "TradesHistory permission absent (expected); securely utilizing ledger fallback for closed trades."
                if invalid_key
                else "TradesHistory unavailable; using ledger fallback"
            ),
            details={
                "error": trades_history_error,
                "likely_cause": "kraken_api_key_permission_or_scope"
                if invalid_key
                else "unknown",
            },
            dedupe_key="trades_history_invalid_key" if invalid_key else "trades_history_fallback",
        )
