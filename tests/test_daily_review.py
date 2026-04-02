from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from core.memory.daily_review import build_daily_review_report
from core.memory.trade_memory import TradeMemoryStore, build_outcome_record


class DailyReviewTests(unittest.TestCase):
    def test_build_daily_review_summarizes_value_trades_blocks_and_lessons(self) -> None:
        as_of = datetime(2026, 4, 1, 23, 55, tzinfo=timezone.utc)
        outcomes = [
            build_outcome_record(
                symbol="FET/USD",
                side="LONG",
                lane="L2",
                pnl_pct=-0.042,
                pnl_usd=-3.15,
                hold_minutes=32.0,
                exit_reason="stop_loss",
                entry_reasons=["breakout"],
                regime_label="trend",
            ),
            build_outcome_record(
                symbol="RENDER/USD",
                side="LONG",
                lane="L2",
                pnl_pct=0.031,
                pnl_usd=2.35,
                hold_minutes=28.0,
                exit_reason="take_profit",
                entry_reasons=["continuation"],
                regime_label="trend",
            ),
        ]
        for item in outcomes:
            item.ts = as_of.isoformat()

        report = build_daily_review_report(
            account_sync={
                "cash_usd": 668.76,
                "initial_equity_usd": 676.17,
                "positions_state": [],
                "synced_positions_usd": {},
            },
            outcomes=outcomes,
            decision_debug=[
                {
                    "ts": as_of.isoformat(),
                    "execution_status": "blocked",
                    "execution_reason": ["net_edge_non_positive"],
                },
                {
                    "ts": as_of.isoformat(),
                    "execution_status": "blocked",
                    "execution_reason": ["net_edge_non_positive", "spread_high(0.5>0.2)"],
                },
                {
                    "ts": as_of.isoformat(),
                    "execution_status": "no_trade",
                },
            ],
            lessons=[
                {
                    "ts": as_of.isoformat(),
                    "symbol": "FET/USD",
                    "lesson": "avoid weak continuation after cost drag",
                    "suggested_adjustment": "raise cost sensitivity for weak momentum",
                    "confidence": 0.82,
                }
            ],
            as_of=as_of,
        )

        self.assertEqual(report["day"], "2026-04-01")
        self.assertAlmostEqual(report["account_value"]["book_value_usd"], 668.76)
        self.assertAlmostEqual(report["account_value"]["open_pnl_usd"], -7.41, places=2)
        self.assertEqual(report["trade_day"]["closed_trade_count"], 2)
        self.assertEqual(report["trade_day"]["win_count"], 1)
        self.assertEqual(report["trade_day"]["loss_count"], 1)
        self.assertAlmostEqual(report["trade_day"]["realized_pnl_usd"], -0.8, places=2)
        self.assertEqual(report["execution_review"]["blocked_trade_count"], 2)
        self.assertEqual(report["execution_review"]["blocked_reasons"][0]["reason"], "net_edge_non_positive")
        self.assertEqual(report["execution_review"]["blocked_reasons"][0]["count"], 2)
        self.assertEqual(report["learning_review"]["lesson_count"], 1)
        self.assertEqual(report["learning_review"]["recent_lessons"][0]["symbol"], "FET/USD")

    def test_build_daily_review_ignores_other_days(self) -> None:
        as_of = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        current = build_outcome_record(
            symbol="SEI/USD",
            side="LONG",
            lane="L3",
            pnl_pct=0.01,
            pnl_usd=1.0,
            hold_minutes=15.0,
            exit_reason="take_profit",
            entry_reasons=["breakout"],
            regime_label="trend",
        )
        current.ts = as_of.isoformat()
        older = build_outcome_record(
            symbol="SEI/USD",
            side="LONG",
            lane="L3",
            pnl_pct=-0.02,
            pnl_usd=-2.0,
            hold_minutes=20.0,
            exit_reason="stop_loss",
            entry_reasons=["breakout"],
            regime_label="trend",
        )
        older.ts = datetime(2026, 3, 31, 23, 50, tzinfo=timezone.utc).isoformat()

        report = build_daily_review_report(
            account_sync={"cash_usd": 100.0, "initial_equity_usd": 100.0, "positions_state": [], "synced_positions_usd": {}},
            outcomes=[current, older],
            decision_debug=[],
            lessons=[],
            as_of=as_of,
        )

        self.assertEqual(report["trade_day"]["closed_trade_count"], 1)
        self.assertAlmostEqual(report["trade_day"]["realized_pnl_usd"], 1.0)


class DailyReviewDiskTests(unittest.TestCase):
    def test_trade_memory_store_can_feed_daily_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeMemoryStore(base_dir=Path(tmpdir))
            outcome = build_outcome_record(
                symbol="BTC/USD",
                side="LONG",
                lane="L1",
                pnl_pct=0.02,
                pnl_usd=5.0,
                hold_minutes=60.0,
                exit_reason="take_profit",
                entry_reasons=["trend"],
                regime_label="trend",
            )
            store.append_outcome(outcome)
            raw = [item.to_dict() for item in store.load_outcomes()]
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw[0]["symbol"], "BTC/USD")


if __name__ == "__main__":
    unittest.main()
