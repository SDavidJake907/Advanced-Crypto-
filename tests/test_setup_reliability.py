import tempfile
import unittest
from pathlib import Path

from core.memory.setup_reliability import (
    build_setup_reliability_map,
    build_setup_reliability_summary,
    setup_reliability_as_dict,
)
from core.memory.trade_memory import TradeMemoryStore, build_outcome_record


class SetupReliabilityTests(unittest.TestCase):
    def test_empty_store_returns_empty_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeMemoryStore(base_dir=Path(tmpdir))
            grouped = build_setup_reliability_map(store=store, lookback=20, min_trades=2)

        self.assertEqual(grouped["by_lane"], {})
        self.assertEqual(grouped["by_setup_key"], {})

    def test_grouped_records_include_expectancy_and_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeMemoryStore(base_dir=Path(tmpdir))
            for pnl in (0.03, 0.02, 0.01, -0.002):
                store.append_outcome(
                    build_outcome_record(
                        symbol="PENDLE/USD",
                        side="LONG",
                        lane="L2",
                        pnl_pct=pnl,
                        pnl_usd=7.0,
                        hold_minutes=55.0,
                        exit_reason="take_profit" if pnl > 0 else "stale_exit",
                        entry_reasons=["pullback_hold"],
                        regime_label="breakout",
                        entry_score=79.0,
                        entry_recommendation="BUY",
                        pattern_name="bullish_flag",
                        expected_edge_pct=1.4,
                        realized_edge_pct=pnl - 0.0063,
                        edge_capture_ratio=0.55,
                    )
                )

            grouped = build_setup_reliability_map(store=store, lookback=20, min_trades=3)
            as_dict = setup_reliability_as_dict(grouped)

        record = grouped["by_setup_key"]["L2|bullish_flag|BUY|pullback_hold"]
        self.assertEqual(record.dimension, "setup_key")
        self.assertEqual(record.trade_count, 4)
        self.assertGreater(record.win_rate, 0.7)
        self.assertGreater(record.avg_pnl_pct, 0.0)
        self.assertEqual(record.verdict, "strong")
        self.assertIn("L2|bullish_flag|BUY|pullback_hold", as_dict["by_setup_key"])

    def test_summary_orders_strongest_and_weakest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeMemoryStore(base_dir=Path(tmpdir))
            for pnl in (0.03, 0.025, 0.015):
                store.append_outcome(
                    build_outcome_record(
                        symbol="ALGO/USD",
                        side="LONG",
                        lane="L2",
                        pnl_pct=pnl,
                        pnl_usd=5.0,
                        hold_minutes=60.0,
                        exit_reason="take_profit",
                        entry_reasons=["breakout"],
                        regime_label="trend",
                        entry_score=80.0,
                        entry_recommendation="BUY",
                        pattern_name="double_bottom",
                        expected_edge_pct=1.0,
                        realized_edge_pct=pnl - 0.006,
                        edge_capture_ratio=0.7,
                    )
                )
            for pnl in (-0.03, -0.02, -0.01):
                store.append_outcome(
                    build_outcome_record(
                        symbol="ARB/USD",
                        side="LONG",
                        lane="L3",
                        pnl_pct=pnl,
                        pnl_usd=-5.0,
                        hold_minutes=35.0,
                        exit_reason="stop_loss",
                        entry_reasons=["late_breakout"],
                        regime_label="range",
                        entry_score=65.0,
                        entry_recommendation="WATCH",
                        pattern_name="none",
                        expected_edge_pct=0.7,
                        realized_edge_pct=pnl - 0.006,
                        edge_capture_ratio=-0.5,
                    )
                )

            summary = build_setup_reliability_summary(store=store, lookback=20, min_trades=3, limit=1)

        self.assertEqual(summary["by_lane"]["count"], 2)
        self.assertEqual(summary["by_lane"]["strongest"][0]["key"], "L2")
        self.assertEqual(summary["by_lane"]["weakest"][0]["key"], "L3")


if __name__ == "__main__":
    unittest.main()
