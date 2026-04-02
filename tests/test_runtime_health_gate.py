from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from core.risk.basic_risk import BasicRiskEngine
from core.risk.runtime_health import evaluate_runtime_health


class RuntimeHealthGateTests(unittest.TestCase):
    def test_runtime_health_blocks_on_stale_collector_and_account_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            health = evaluate_runtime_health(
                now_ts=time.time(),
                collector_telemetry={"ts": "2026-01-01T00:00:00+00:00"},
                account_sync={"ts": "2026-01-01T00:00:00+00:00", "status": "sync_failed"},
                watchdog_results={"phi3": {"ok": True}, "nemotron": {"ok": True}},
                telemetry_dir=Path(tmpdir),
                require_account_sync=True,
                llm_required=False,
                collector_max_age_sec=1.0,
                account_sync_max_age_sec=1.0,
            )
        self.assertFalse(health.ok)
        self.assertIn("runtime_health_collector_stale", health.reasons)
        self.assertIn("runtime_health_account_sync_failed", health.reasons)

    def test_runtime_health_blocks_llm_path_when_dependencies_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            now_ts = time.time()
            health = evaluate_runtime_health(
                now_ts=now_ts,
                collector_telemetry={"ts": "2026-03-19T00:00:00+00:00"},
                account_sync={"ts": "2026-03-19T00:00:00+00:00", "status": "synced"},
                watchdog_results={"phi3": {"ok": False}, "nemotron": {"ok": False}},
                telemetry_dir=Path(tmpdir),
                require_account_sync=False,
                llm_required=True,
                collector_max_age_sec=10**10,
                account_sync_max_age_sec=10**10,
            )
        self.assertFalse(health.ok)
        self.assertIn("runtime_health_phi3_unhealthy", health.reasons)
        self.assertIn("runtime_health_nemotron_unhealthy", health.reasons)

    def test_runtime_health_allows_degraded_account_sync_when_last_success_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            now_ts = time.time()
            fresh_ts = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now_ts))
            health = evaluate_runtime_health(
                now_ts=now_ts,
                collector_telemetry={"ts": fresh_ts},
                account_sync={
                    "ts": fresh_ts,
                    "status": "degraded",
                    "last_success_ts": fresh_ts,
                    "error": "temporary network error",
                },
                watchdog_results={"phi3": {"ok": True}, "nemotron": {"ok": True}},
                telemetry_dir=Path(tmpdir),
                require_account_sync=True,
                llm_required=False,
                collector_max_age_sec=60.0,
                account_sync_max_age_sec=60.0,
            )
        self.assertTrue(health.ok)
        self.assertEqual(health.reasons, [])

    def test_basic_risk_engine_blocks_on_runtime_health_and_invalid_book(self) -> None:
        engine = BasicRiskEngine(max_position_notional=1_000_000.0, max_leverage=10.0, cooldown_bars=0)
        checks = engine.check(
            "LONG",
            {
                "symbol": "BTC/USD",
                "price": 100.0,
                "book_valid": False,
                "runtime_health": {
                    "ok": False,
                    "reasons": ["runtime_health_collector_stale"],
                },
            },
            {
                "cash": 1000.0,
                "positions": {},
            },
        )
        self.assertIn("runtime_health_collector_stale", checks)
        self.assertIn("runtime_health_book_invalid", checks)
        self.assertIn("block", checks)

    def test_basic_risk_engine_warns_only_for_invalid_book_when_flat(self) -> None:
        engine = BasicRiskEngine(max_position_notional=1_000_000.0, max_leverage=10.0, cooldown_bars=0)
        checks = engine.check(
            "FLAT",
            {
                "symbol": "BTC/USD",
                "price": 100.0,
                "book_valid": False,
                "runtime_health": {"ok": True, "reasons": []},
            },
            {"cash": 1000.0, "positions": {}},
        )
        self.assertIn("no_action", checks)


if __name__ == "__main__":
    unittest.main()
