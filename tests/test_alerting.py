from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.operator_ui import main as operator_ui
from apps.trader import logging_sink
from core.runtime import alerts


class AlertingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.alerts_path = self.root / "logs" / "alerts.jsonl"
        self.nemo_recommendations_path = self.root / "logs" / "nemo_recommendations.jsonl"
        alerts._LAST_ALERT_TS.clear()
        self.alerts_patch = patch.object(alerts, "ALERTS_LOG_PATH", self.alerts_path)
        self.alerts_patch.start()
        self.addCleanup(self.alerts_patch.stop)
        self.nemo_patch = patch.object(logging_sink, "NEMO_RECOMMENDATIONS_LOG_PATH", self.nemo_recommendations_path)
        self.nemo_patch.start()
        self.addCleanup(self.nemo_patch.stop)

    def _read_alerts(self) -> list[dict]:
        if not self.alerts_path.exists():
            return []
        return [json.loads(line) for line in self.alerts_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _read_nemo_recommendations(self) -> list[dict]:
        if not self.nemo_recommendations_path.exists():
            return []
        return [json.loads(line) for line in self.nemo_recommendations_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_decision_debug_batch_parse_fallback_emits_alert(self) -> None:
        logging_sink.log_decision_debug(
            {
                "ts": "2026-03-30T23:00:00Z",
                "symbol": "BTC/USD",
                "nemotron": {
                    "reason": "batch_parse_fallback_hold",
                    "debug": {"batch_reasoning": "batch_error:No JSON object found"},
                },
            }
        )
        alerts_payload = self._read_alerts()
        self.assertEqual(len(alerts_payload), 1)
        self.assertEqual(alerts_payload[0]["code"], "batch_parse_fallback")

    def test_decision_debug_writes_nemo_recommendation_log(self) -> None:
        logging_sink.log_decision_debug(
            {
                "ts": "2026-03-31T05:00:00Z",
                "symbol": "APT/USD",
                "lane": "L2",
                "entry_score": 77.5,
                "entry_recommendation": "BUY",
                "reversal_risk": "LOW",
                "signal": "LONG",
                "execution_status": "submitted",
                "nemotron": {
                    "action": "OPEN",
                    "reason": "breakout",
                    "size_factor_hint": 0.75,
                    "debug": {"batch_reasoning": "APT ranked first"},
                },
            }
        )
        recs = self._read_nemo_recommendations()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["symbol"], "APT/USD")
        self.assertEqual(recs[0]["action"], "OPEN")
        self.assertEqual(recs[0]["reason"], "breakout")
        self.assertEqual(recs[0]["batch_reasoning"], "APT ranked first")

    def test_account_sync_fallback_emits_alert(self) -> None:
        logging_sink.log_account_sync(
            {
                "ts": "2026-03-30T23:00:00Z",
                "status": "synced",
                "diagnostics": {
                    "trades_history_error": "Kraken private API error for /0/private/TradesHistory: ['EAPI:Invalid key']"
                },
            }
        )
        alerts_payload = self._read_alerts()
        self.assertEqual(len(alerts_payload), 1)
        self.assertEqual(alerts_payload[0]["code"], "trades_history_invalid_key")

    def test_account_sync_degraded_emits_warning_alert(self) -> None:
        logging_sink.log_account_sync(
            {
                "ts": "2026-03-30T23:00:00Z",
                "status": "degraded",
                "last_success_ts": "2026-03-30T22:58:00Z",
                "error": "temporary network error",
            }
        )
        alerts_payload = self._read_alerts()
        self.assertEqual(len(alerts_payload), 1)
        self.assertEqual(alerts_payload[0]["code"], "sync_degraded")

    def test_operator_ui_status_includes_alerts(self) -> None:
        logs_dir = self.root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "account_sync.json").write_text(json.dumps({"cash_usd": 1.0, "positions_state": []}), encoding="utf-8")
        (logs_dir / "collector_telemetry.json").write_text(json.dumps({}), encoding="utf-8")
        (logs_dir / "open_orders.json").write_text(json.dumps({}), encoding="utf-8")
        (logs_dir / "nvidia_optimizer_reviews.jsonl").write_text("", encoding="utf-8")
        (logs_dir / "decision_debug.jsonl").write_text("", encoding="utf-8")
        (logs_dir / "alerts.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-03-30T23:00:00Z",
                    "level": "error",
                    "source": "nemotron",
                    "code": "batch_parse_fallback",
                    "message": "Nemotron batch response fell back due to parse failure",
                    "details": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.root / "universe.json").write_text(json.dumps({"active_pairs": ["BTC/USD"]}), encoding="utf-8")
        with patch.object(operator_ui, "ROOT", self.root):
            payload = operator_ui.status()
        self.assertEqual(payload["alert_summary"]["count"], 1)
        self.assertEqual(payload["alerts"][0]["code"], "batch_parse_fallback")


if __name__ == "__main__":
    unittest.main()
