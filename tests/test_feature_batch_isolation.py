from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.features.batch import compute_features_batch, slice_features_for_asset


def _frame(length: int, *, start: float) -> pd.DataFrame:
    closes = np.linspace(start, start + length - 1, num=length, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=length, freq="min"),
            "open": closes - 0.5,
            "high": closes + 0.5,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(length, 100.0, dtype=np.float64),
        }
    )


def _fake_gpu_features(flat_prices: list[float], n_assets: int, n_points: int, _cfg: object) -> SimpleNamespace:
    prices = np.asarray(flat_prices, dtype=np.float64).reshape(n_assets, n_points)
    base = np.maximum(prices[:, 0], 1.0)
    return SimpleNamespace(
        momentum=(prices[:, -1] / base) - 1.0,
        volatility=np.full(n_assets, 0.02, dtype=np.float64),
    )


class FeatureBatchIsolationTests(unittest.TestCase):
    @patch("core.features.batch.compute_trend_state")
    @patch("core.features.batch.compute_macro_30d_batch")
    @patch("core.features.batch.compute_regime_7d_batch")
    @patch("core.features.batch.compute_trend_1h_batch")
    @patch("core.features.batch.compute_northstar_fingerprint_gpu")
    @patch("core.features.batch.compute_correlation_gpu")
    @patch("core.features.batch.compute_bollinger_gpu")
    @patch("core.features.batch.compute_northstar_batch_features_gpu")
    @patch("core.features.batch.compute_atr_gpu")
    @patch("core.features.batch.compute_rsi_gpu")
    @patch("core.features.batch.cuda_features.compute_features_gpu")
    def test_short_history_symbol_does_not_flatten_batch(
        self,
        mock_compute_features_gpu,
        mock_rsi,
        mock_atr,
        mock_northstar,
        mock_bollinger,
        mock_corr,
        mock_fingerprint,
        mock_trend_1h,
        mock_regime_7d,
        mock_macro_30d,
        mock_trend_state,
    ) -> None:
        mock_compute_features_gpu.side_effect = _fake_gpu_features
        mock_rsi.side_effect = lambda prices, lookback=14: np.full(prices.shape[0], 55.0, dtype=np.float64)
        mock_atr.side_effect = lambda highs, lows, prices, lookback=14: np.full(prices.shape[0], 1.0, dtype=np.float64)
        mock_northstar.side_effect = lambda prices: {
            "hurst": np.full(prices.shape[0], 0.5, dtype=np.float64),
            "entropy": np.full(prices.shape[0], 0.5, dtype=np.float64),
            "autocorr": np.zeros(prices.shape[0], dtype=np.float64),
        }
        mock_bollinger.side_effect = lambda prices, lookback=20, num_std=2.0: {
            "middle": prices[:, -1],
            "upper": prices[:, -1] + 1.0,
            "lower": prices[:, -1] - 1.0,
            "bandwidth": np.full(prices.shape[0], 0.1, dtype=np.float64),
        }
        mock_corr.side_effect = lambda prices: np.eye(prices.shape[0], dtype=np.float64)
        mock_fingerprint.return_value = {
            "metrics": np.zeros(8, dtype=np.float64),
            "r_mkt": 0.0,
            "r_btc": 0.0,
            "r_eth": 0.0,
            "breadth": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "rv_mkt": 0.0,
            "corr_avg": 0.0,
        }
        mock_trend_1h.side_effect = lambda mapping, lookback=8: {
            "symbols": list(mapping.keys()),
            "trend_1h": np.ones(len(mapping), dtype=np.int64),
        }
        mock_regime_7d.side_effect = lambda mapping, lookback=6: {
            "symbols": list(mapping.keys()),
            "regime_7d": ["trending"] * len(mapping),
        }
        mock_macro_30d.side_effect = lambda mapping, lookback=8: {
            "symbols": list(mapping.keys()),
            "macro_30d": ["bull"] * len(mapping),
        }
        mock_trend_state.side_effect = lambda prices, bandwidth, **kwargs: {
            "ma_7": prices[:, -1],
            "ma_26": prices[:, -1] - 1.0,
            "macd": np.zeros(prices.shape[0], dtype=np.float64),
            "macd_signal": np.zeros(prices.shape[0], dtype=np.float64),
            "macd_hist": np.zeros(prices.shape[0], dtype=np.float64),
            "adx": np.zeros(prices.shape[0], dtype=np.float64),
            "trend_confirmed": np.ones(prices.shape[0], dtype=bool),
            "ranging_market": np.zeros(prices.shape[0], dtype=bool),
        }

        features = compute_features_batch(
            {
                "BTC/USD": _frame(40, start=100.0),
                "DOGE/USD": _frame(12, start=10.0),
            }
        )

        self.assertEqual(features["symbols"], ["BTC/USD", "DOGE/USD"])
        self.assertEqual(features["history_points"].tolist(), [40, 12])
        self.assertEqual(features["indicators_ready"].tolist(), [True, False])
        self.assertEqual(features["feature_status"], ["ready", "warmup"])
        self.assertEqual(features["feature_failure_reason"][0], "")
        self.assertIn("insufficient_history", features["feature_failure_reason"][1])
        self.assertGreater(float(features["momentum_30"][0]), 0.0)
        self.assertEqual(float(features["momentum_30"][1]), 0.0)

        sliced = slice_features_for_asset(features, 0, lane_hint="L3")
        self.assertIn("coin_profile", sliced)
        self.assertIn("ema_26", sliced)
        self.assertIn("ema9_above_ema26", sliced)
        self.assertIn("ema_cross_distance_pct", sliced)
        self.assertEqual(
            set(sliced["coin_profile"].keys()),
            {
                "structure_quality",
                "momentum_quality",
                "volume_quality",
                "trade_quality",
                "market_support",
                "continuation_quality",
                "risk_quality",
            },
        )
        self.assertIsInstance(sliced["ema9_above_ema26"], bool)
        self.assertGreaterEqual(float(sliced["ema_cross_distance_pct"]), 0.0)
        for value in sliced["coin_profile"].values():
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 100.0)


if __name__ == "__main__":
    unittest.main()
