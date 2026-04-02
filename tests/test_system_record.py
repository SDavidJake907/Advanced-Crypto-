from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config import runtime
from core.state import system_record


class SystemRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db_path = self.root / "system_record.sqlite3"
        self.override_path = self.root / "runtime_overrides.json"
        self.proposal_path = self.root / "runtime_override_proposals.json"
        self.db_patch = patch.object(system_record, "SYSTEM_RECORD_DB_PATH", self.db_path)
        self.override_patch = patch.object(runtime, "RUNTIME_OVERRIDES_PATH", self.override_path)
        self.proposal_patch = patch.object(runtime, "RUNTIME_OVERRIDE_PROPOSALS_PATH", self.proposal_path)
        self.db_patch.start()
        self.override_patch.start()
        self.proposal_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(self.override_patch.stop)
        self.addCleanup(self.proposal_patch.stop)

    def _count(self, table: str) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0] if row else 0)

    def test_system_record_writes_core_event_tables(self) -> None:
        system_record.record_decision_trace(
            {
                "timestamp": "2026-03-19T00:00:00Z",
                "symbol": "BTC/USD",
                "features": {"lane": "L2"},
                "signal": "LONG",
                "execution": {"status": "filled"},
            }
        )
        system_record.record_decision_debug(
            {
                "ts": "2026-03-19T00:00:01Z",
                "symbol": "BTC/USD",
                "lane": "L2",
                "phase": "entry",
                "execution_status": "filled",
            }
        )
        system_record.record_fill_event(
            {
                "symbol": "BTC/USD",
                "side": "BUY",
                "status": "filled",
                "price": 50000.0,
                "qty": 0.01,
                "notional": 500.0,
                "order_type": "limit",
            },
            lane="L2",
            ts="2026-03-19T00:00:02Z",
        )
        system_record.record_outcome_review(
            {
                "ts": "2026-03-19T00:00:03Z",
                "symbol": "BTC/USD",
                "lane": "L2",
                "side": "LONG",
                "exit_reason": "rotate_exit",
                "pnl_pct": 0.04,
                "hold_minutes": 32.0,
            }
        )
        system_record.record_optimizer_review(
            {
                "ts": "2026-03-19T00:00:04Z",
                "summary": "small tune",
                "confidence": 0.7,
            }
        )
        system_record.record_shadow_decision(
            {
                "ts": "2026-03-19T00:00:05Z",
                "symbol": "BTC/USD",
                "baseline_engine": "classic",
                "shadow_engine": "llm",
                "baseline_signal": "LONG",
                "shadow_signal": "FLAT",
                "baseline_status": "filled",
                "shadow_status": "blocked",
            }
        )
        system_record.record_replay_run(
            {
                "run_id": "replay-unit-test",
                "created_at": "2026-03-19T00:00:06Z",
                "baseline_engine": "classic",
                "shadow_engine": "llm",
                "symbols": ["BTC/USD"],
            }
        )

        self.assertEqual(self._count("decision_traces"), 1)
        self.assertEqual(self._count("decision_debug"), 1)
        self.assertEqual(self._count("fill_events"), 1)
        self.assertEqual(self._count("outcome_reviews"), 1)
        self.assertEqual(self._count("optimizer_reviews"), 1)
        self.assertEqual(self._count("shadow_decisions"), 1)
        self.assertEqual(self._count("replay_runs"), 1)

    def test_runtime_override_proposals_are_synced_to_system_record(self) -> None:
        proposal = runtime.stage_runtime_override_proposal(
            {"PORTFOLIO_MAX_OPEN_POSITIONS": 3},
            source="unit_test",
            summary="raise slots",
            validation={
                "replay_passed": True,
                "shadow_passed": True,
                "human_approved": True,
            },
        )

        with closing(sqlite3.connect(self.db_path)) as conn:
            staged = conn.execute(
                "SELECT status, source FROM runtime_override_proposals WHERE proposal_id = ?",
                (proposal["id"],),
            ).fetchone()
        self.assertEqual(staged, ("pending", "unit_test"))

        runtime.apply_runtime_override_proposal(proposal["id"], approved_by="tester")

        with closing(sqlite3.connect(self.db_path)) as conn:
            applied = conn.execute(
                "SELECT status, applied_by FROM runtime_override_proposals WHERE proposal_id = ?",
                (proposal["id"],),
            ).fetchone()
        self.assertEqual(applied, ("applied", "tester"))


if __name__ == "__main__":
    unittest.main()
