from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_RECORD_DB_PATH = ROOT / "logs" / "system_record.sqlite3"


def _connect() -> sqlite3.Connection:
    SYSTEM_RECORD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SYSTEM_RECORD_DB_PATH)
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _json(payload: Any) -> str:
    return json.dumps(payload, default=str, sort_keys=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_system_record_schema() -> None:
    with closing(_connect()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decision_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                lane TEXT,
                signal TEXT,
                execution_status TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decision_traces_symbol_ts
            ON decision_traces(symbol, ts DESC);

            CREATE TABLE IF NOT EXISTS decision_debug (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                lane TEXT,
                phase TEXT,
                action TEXT,
                execution_status TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decision_debug_symbol_ts
            ON decision_debug(symbol, ts DESC);

            CREATE TABLE IF NOT EXISTS fill_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                lane TEXT,
                side TEXT,
                status TEXT,
                price REAL,
                qty REAL,
                notional REAL,
                order_type TEXT,
                exit_reason TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fill_events_symbol_ts
            ON fill_events(symbol, ts DESC);

            CREATE TABLE IF NOT EXISTS outcome_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                lane TEXT,
                side TEXT,
                exit_reason TEXT,
                pnl_pct REAL,
                hold_minutes REAL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_outcome_reviews_symbol_ts
            ON outcome_reviews(symbol, ts DESC);

            CREATE TABLE IF NOT EXISTS optimizer_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                summary TEXT,
                confidence REAL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_override_proposals (
                proposal_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source TEXT,
                status TEXT,
                summary TEXT,
                applied_at TEXT,
                applied_by TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS replay_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                baseline_engine TEXT,
                shadow_engine TEXT,
                symbols_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS shadow_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                baseline_engine TEXT,
                shadow_engine TEXT,
                baseline_signal TEXT,
                shadow_signal TEXT,
                baseline_status TEXT,
                shadow_status TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_decisions_symbol_ts
            ON shadow_decisions(symbol, ts DESC);
            """
        )
        conn.commit()


def record_decision_trace(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO decision_traces (ts, symbol, lane, signal, execution_status, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("timestamp") or payload.get("ts") or _now_iso()),
                str(payload.get("symbol", "")),
                str((payload.get("features") or {}).get("lane", "")),
                str(payload.get("signal", "")),
                str((payload.get("execution") or {}).get("status", "")),
                _json(payload),
            ),
        )
        conn.commit()


def record_decision_debug(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO decision_debug (ts, symbol, lane, phase, action, execution_status, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("ts") or _now_iso()),
                str(payload.get("symbol", "")),
                str(payload.get("lane", "")),
                str(payload.get("phase", "")),
                str(payload.get("action", "")),
                str(payload.get("execution_status", "")),
                _json(payload),
            ),
        )
        conn.commit()


def record_fill_event(payload: dict[str, Any], *, lane: str = "", ts: str | None = None) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO fill_events (ts, symbol, lane, side, status, price, qty, notional, order_type, exit_reason, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(ts or payload.get("bar_ts") or payload.get("ts") or _now_iso()),
                str(payload.get("symbol", "")),
                str(lane),
                str(payload.get("side", "")),
                str(payload.get("status", "")),
                float(payload.get("price", 0.0) or 0.0),
                float(payload.get("qty", 0.0) or 0.0),
                float(payload.get("notional", 0.0) or 0.0),
                str(payload.get("order_type", "")),
                str(payload.get("exit_reason", "")),
                _json(payload),
            ),
        )
        conn.commit()


def record_outcome_review(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO outcome_reviews (ts, symbol, lane, side, exit_reason, pnl_pct, hold_minutes, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("ts") or _now_iso()),
                str(payload.get("symbol", "")),
                str(payload.get("lane", "")),
                str(payload.get("side", "")),
                str(payload.get("exit_reason", "")),
                float(payload.get("pnl_pct", 0.0) or 0.0),
                float(payload.get("hold_minutes", 0.0) or 0.0),
                _json(payload),
            ),
        )
        conn.commit()


def record_optimizer_review(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO optimizer_reviews (ts, summary, confidence, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(payload.get("ts") or _now_iso()),
                str(payload.get("summary", "")),
                float(payload.get("confidence", 0.0) or 0.0),
                _json(payload),
            ),
        )
        conn.commit()


def upsert_runtime_override_proposal(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO runtime_override_proposals (
                proposal_id, created_at, source, status, summary, applied_at, applied_by, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                status=excluded.status,
                summary=excluded.summary,
                applied_at=excluded.applied_at,
                applied_by=excluded.applied_by,
                payload_json=excluded.payload_json
            """,
            (
                str(payload.get("id", "")),
                str(payload.get("created_at") or _now_iso()),
                str(payload.get("source", "")),
                str(payload.get("status", "")),
                str(payload.get("summary", "")),
                str(payload.get("applied_at", "")),
                str(payload.get("applied_by", "")),
                _json(payload),
            ),
        )
        conn.commit()


def record_replay_run(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO replay_runs (run_id, created_at, baseline_engine, shadow_engine, symbols_json, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                baseline_engine=excluded.baseline_engine,
                shadow_engine=excluded.shadow_engine,
                symbols_json=excluded.symbols_json,
                payload_json=excluded.payload_json
            """,
            (
                str(payload.get("run_id", "")),
                str(payload.get("created_at") or _now_iso()),
                str(payload.get("baseline_engine", "")),
                str(payload.get("shadow_engine", "")),
                _json(payload.get("symbols", [])),
                _json(payload),
            ),
        )
        conn.commit()


def record_shadow_decision(payload: dict[str, Any]) -> None:
    ensure_system_record_schema()
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO shadow_decisions (
                ts, symbol, baseline_engine, shadow_engine, baseline_signal, shadow_signal,
                baseline_status, shadow_status, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("ts") or _now_iso()),
                str(payload.get("symbol", "")),
                str(payload.get("baseline_engine", "")),
                str(payload.get("shadow_engine", "")),
                str(payload.get("baseline_signal", "")),
                str(payload.get("shadow_signal", "")),
                str(payload.get("baseline_status", "")),
                str(payload.get("shadow_status", "")),
                _json(payload),
            ),
        )
        conn.commit()
