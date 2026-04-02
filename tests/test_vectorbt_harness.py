import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.research.vectorbt_harness import (
    _build_entry_mask,
    build_policy_feature_frame,
    discover_history_symbols,
    load_history_frame,
    run_batch_threshold_sweep,
    run_threshold_sweep,
    run_walk_forward_validation,
)


def _write_history_csv(path: Path, rows: int = 80) -> None:
    closes = [100.0 + i * 0.5 for i in range(rows)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1000.0 + i * 5.0 for i in range(rows)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


class VectorBTHarnessTests(unittest.TestCase):
    def test_load_history_frame_reads_symbol_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "candles_BTCUSD_1h.csv"
            _write_history_csv(path, rows=12)

            frame = load_history_frame("BTC/USD", history_dir=root)

            self.assertEqual(len(frame), 12)
            self.assertIn("timestamp", frame.columns)

    def test_build_policy_feature_frame_adds_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "candles_BTCUSD_1h.csv"
            _write_history_csv(path, rows=40)
            frame = load_history_frame("BTC/USD", history_dir=root)

            features = build_policy_feature_frame("BTC/USD", frame)

            for column in ("momentum_5", "momentum_14", "momentum_30", "trend_1h", "volume_ratio", "price_zscore", "rsi", "atr_pct"):
                self.assertIn(column, features.columns)

    def test_run_threshold_sweep_returns_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "candles_BTCUSD_1h.csv"
            _write_history_csv(path, rows=120)

            result = run_threshold_sweep(
                "BTC/USD",
                history_dir=root,
                entry_score_thresholds=[55.0, 65.0],
                hold_bars=12,
            )

            self.assertEqual(len(result.summary), 2)
            self.assertIn("variant", result.summary.columns)
            self.assertIn("total_return_pct", result.summary.columns)

    def test_run_threshold_sweep_can_export_summary_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "candles_BTCUSD_1h.csv"
            out_path = root / "results" / "summary.csv"
            _write_history_csv(path, rows=120)

            result = run_threshold_sweep(
                "BTC/USD",
                history_dir=root,
                entry_score_thresholds=[60.0],
                lane_filter="L3",
                min_net_edge_pct=-2.0,
                summary_csv_path=out_path,
            )

            self.assertEqual(len(result.summary), 1)
            self.assertTrue(out_path.exists())
            exported = pd.read_csv(out_path)
            self.assertEqual(len(exported), 1)

    def test_discover_history_symbols_reads_available_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_history_csv(root / "candles_BTCUSD_1h.csv", rows=20)
            _write_history_csv(root / "candles_ETHUSD_1h.csv", rows=20)

            symbols = discover_history_symbols(root)

            self.assertEqual(symbols, ["BTC/USD", "ETH/USD"])

    def test_run_batch_threshold_sweep_builds_aggregate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_history_csv(root / "candles_BTCUSD_1h.csv", rows=120)
            _write_history_csv(root / "candles_ETHUSD_1h.csv", rows=120)
            per_symbol_csv = root / "out" / "per_symbol.csv"
            aggregate_csv = root / "out" / "aggregate.csv"

            result = run_batch_threshold_sweep(
                ["BTC/USD", "ETH/USD"],
                history_dir=root,
                entry_score_thresholds=[60.0],
                hold_bars=12,
                min_net_edge_pct=-2.0,
                per_symbol_csv_path=per_symbol_csv,
                aggregate_csv_path=aggregate_csv,
            )

            self.assertEqual(len(result.per_symbol_summary), 2)
            self.assertEqual(len(result.aggregate_summary), 1)
            self.assertTrue(per_symbol_csv.exists())
            self.assertTrue(aggregate_csv.exists())
            self.assertIn("avg_total_return_pct", result.aggregate_summary.columns)

    def test_build_entry_mask_applies_bullish_divergence_score_bonus(self) -> None:
        features = pd.DataFrame(
            {
                "entry_score": [67.0, 67.0],
                "entry_recommendation": ["BUY", "BUY"],
                "reversal_risk": ["LOW", "LOW"],
                "lane": ["L3", "L3"],
                "bullish_divergence": [True, False],
                "net_edge_pct": [1.0, 1.0],
                "tp_after_cost_valid": [True, True],
            }
        )

        mask = _build_entry_mask(
            features,
            threshold=70.0,
            bullish_divergence_score_bonus=4.0,
        )

        self.assertEqual(mask.tolist(), [True, False])

    def test_build_entry_mask_can_promote_near_threshold_watch(self) -> None:
        features = pd.DataFrame(
            {
                "entry_score": [68.0, 68.0],
                "entry_recommendation": ["WATCH", "WATCH"],
                "reversal_risk": ["LOW", "LOW"],
                "lane": ["L3", "L3"],
                "bullish_divergence": [True, False],
                "net_edge_pct": [1.0, 1.0],
                "tp_after_cost_valid": [True, True],
            }
        )

        mask = _build_entry_mask(
            features,
            threshold=70.0,
            bullish_divergence_promotion_window=2.0,
        )

        self.assertEqual(mask.tolist(), [True, False])

    def test_run_threshold_sweep_variant_names_include_bonus_and_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "candles_BTCUSD_1h.csv"
            _write_history_csv(path, rows=120)

            result = run_threshold_sweep(
                "BTC/USD",
                history_dir=root,
                entry_score_thresholds=[70.0],
                lane_filter="L3",
                bullish_divergence_score_bonus=4.0,
                bullish_divergence_promotion_window=2.0,
                hold_bars=12,
            )

            variant = str(result.summary.loc[0, "variant"])
            self.assertIn("divbonus_4p0", variant)
            self.assertIn("divpromo_2p0", variant)

    def test_run_walk_forward_validation_returns_summary_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "candles_BTCUSD_1h.csv"
            summary_csv = root / "wf" / "summary.csv"
            windows_csv = root / "wf" / "windows.csv"
            _write_history_csv(path, rows=160)

            result = run_walk_forward_validation(
                "BTC/USD",
                history_dir=root,
                entry_score_thresholds=[55.0, 65.0],
                train_bars=72,
                test_bars=24,
                hold_bars=12,
                min_net_edge_pct=-2.0,
                summary_csv_path=summary_csv,
                windows_csv_path=windows_csv,
            )

            self.assertEqual(len(result.summary), 1)
            self.assertGreater(len(result.windows), 0)
            self.assertIn("selected_threshold", result.windows.columns)
            self.assertTrue(summary_csv.exists())
            self.assertTrue(windows_csv.exists())


if __name__ == "__main__":
    unittest.main()
