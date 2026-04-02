import tempfile
import unittest
from pathlib import Path

from core.memory.trade_memory import TradeMemoryStore, build_outcome_record
from core.memory.setup_reliability import build_setup_reliability_map


class TradeMemoryEdgeTests(unittest.TestCase):
    def test_outcome_store_round_trip_preserves_edge_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeMemoryStore(base_dir=Path(tmpdir))
            outcome = build_outcome_record(
                symbol="ETH/USD",
                side="LONG",
                lane="L2",
                pnl_pct=0.02,
                pnl_usd=10.0,
                hold_minutes=30.0,
                exit_reason="take_profit",
                entry_reasons=["breakout"],
                regime_label="trending",
                entry_score=74.0,
                entry_recommendation="BUY",
                pattern_name="double_bottom",
                expected_edge_pct=1.5,
                realized_edge_pct=0.014,
                edge_capture_ratio=0.0093,
            )
            store.append_outcome(outcome)

            loaded = store.load_outcomes()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].expected_edge_pct, 1.5)
        self.assertEqual(loaded[0].realized_edge_pct, 0.014)
        self.assertEqual(loaded[0].edge_capture_ratio, 0.0093)
        self.assertEqual(loaded[0].lane, "L2")
        self.assertEqual(loaded[0].entry_score, 74.0)
        self.assertEqual(loaded[0].entry_recommendation, "BUY")
        self.assertEqual(loaded[0].pattern_name, "double_bottom")
        self.assertEqual(loaded[0].setup_key, "L2|double_bottom|BUY|breakout")

    def test_setup_reliability_groups_by_lane_pattern_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeMemoryStore(base_dir=Path(tmpdir))
            for pnl in (0.02, 0.015, -0.005):
                store.append_outcome(
                    build_outcome_record(
                        symbol="ALGO/USD",
                        side="LONG",
                        lane="L3",
                        pnl_pct=pnl,
                        pnl_usd=5.0,
                        hold_minutes=40.0,
                        exit_reason="take_profit" if pnl > 0 else "stale_exit",
                        entry_reasons=["breakout"],
                        regime_label="trending",
                        entry_score=72.0,
                        entry_recommendation="BUY",
                        pattern_name="double_bottom",
                        expected_edge_pct=1.2,
                        realized_edge_pct=pnl - 0.0063,
                        edge_capture_ratio=0.4,
                    )
                )

            grouped = build_setup_reliability_map(store=store, lookback=20, min_trades=2)

        self.assertIn("L3", grouped["by_lane"])
        self.assertIn("double_bottom", grouped["by_pattern"])
        self.assertIn("breakout", grouped["by_entry_reason"])
        self.assertIn("ALGO/USD|trending", grouped["by_symbol_regime"])
        self.assertIn("L3|double_bottom|BUY|breakout", grouped["by_setup_key"])
        self.assertEqual(grouped["by_lane"]["L3"].trade_count, 3)
        self.assertEqual(grouped["by_pattern"]["double_bottom"].trade_count, 3)


if __name__ == "__main__":
    unittest.main()
