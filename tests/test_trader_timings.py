from __future__ import annotations

import unittest

from unittest.mock import patch

import numpy as np

from apps.trader.main import (
    _build_phi3_reflex_cache_key,
    _entry_nemo_limit_for_cycle,
    _entry_nemo_overflow_limit_for_cycle,
    _format_decision_timings,
)
from core.features.batch import slice_features_for_asset


class TraderTimingsTests(unittest.TestCase):
    def test_entry_nemo_limit_uses_real_candidate_budget(self) -> None:
        with patch("apps.trader.main.NEMOTRON_MAX_PER_CYCLE", 20):
            with patch("apps.trader.main.NEMOTRON_BATCH_TOP_N", 4):
                with patch("apps.trader.main.get_runtime_setting", return_value=20):
                    self.assertEqual(_entry_nemo_limit_for_cycle(), 20)

    def test_entry_nemo_overflow_limit_uses_runtime_setting(self) -> None:
        with patch("apps.trader.main.get_runtime_setting", return_value=6):
            self.assertEqual(_entry_nemo_overflow_limit_for_cycle(), 6)

    def test_format_decision_timings_separates_decision_and_cycle_total(self) -> None:
        payload = _format_decision_timings(
            70.0,
            {
                "phi3_ms": 337.0,
                "advisory_ms": 0.0,
                "nemotron_ms": 5897.0,
                "execution_ms": 2.0,
                "total_ms": 64447.0,
            },
        )
        self.assertEqual(payload["features_ms"], 70.0)
        self.assertEqual(payload["decision_ms"], 64447.0)
        self.assertEqual(payload["cycle_total_ms"], 64517.0)
        self.assertNotIn("total_ms", payload)

    def test_phi3_reflex_cache_key_ignores_micro_book_noise_within_bar(self) -> None:
        features_a = {
            "bar_ts": "2026-03-31T21:30:00Z",
            "spread_pct": 0.12,
            "book_valid": True,
        }
        features_b = {
            "bar_ts": "2026-03-31T21:30:00Z",
            "spread_pct": 0.48,
            "book_valid": False,
        }
        key_a = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-123", features=features_a)
        key_b = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-123", features=features_b)
        self.assertEqual(key_a, key_b)

    def test_phi3_reflex_cache_key_changes_when_bar_changes(self) -> None:
        features = {
            "bar_ts": "2026-03-31T21:30:00Z",
            "spread_pct": 0.12,
            "book_valid": True,
        }
        key_a = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-123", features=features)
        key_b = _build_phi3_reflex_cache_key(symbol="CRV/USD", bar_key="bar-124", features=features)
        self.assertNotEqual(key_a, key_b)

    def test_slice_features_for_asset_reads_expansion_score_from_batch(self) -> None:
        features_batch = {
            "symbols": ["BTC/USD"],
            "momentum": np.array([0.0]),
            "momentum_1": np.array([0.01]),
            "momentum_5": np.array([0.02]),
            "momentum_14": np.array([0.03]),
            "momentum_30": np.array([0.04]),
            "rotation_score": np.array([0.1]),
            "atr_expanding": np.array([True]),
            "volatility": np.array([0.2]),
            "volume": np.array([1000.0]),
            "volume_ratio": np.array([1.2]),
            "volume_surge": np.array([0.3]),
            "volume_surge_flag": np.array([True]),
            "price_zscore": np.array([0.0]),
            "history_points": np.array([50]),
            "indicators_ready": np.array([True]),
            "feature_status": np.array(["ready"], dtype=object),
            "feature_failure_reason": np.array([""], dtype=object),
            "rsi": np.array([55.0]),
            "atr": np.array([1.0]),
            "atr_pct": np.array([0.01]),
            "hurst": np.array([0.5]),
            "entropy": np.array([0.5]),
            "autocorr": np.array([0.1]),
            "bb_middle": np.array([1.0]),
            "bb_upper": np.array([1.1]),
            "bb_lower": np.array([0.9]),
            "bb_bandwidth": np.array([0.2]),
            "price": np.array([1.0]),
            "bar_ts": ["2026-04-03T00:00:00Z"],
            "bar_idx": np.array([1]),
            "bar_interval_seconds": np.array([60]),
            "correlation": np.array([[1.0]]),
            "market_fingerprint": "fp",
            "market_regime": {"breadth": 0.5, "risk_on": 0.5},
            "trend_1h": np.array([1]),
            "regime_7d": ["trend"],
            "macro_30d": ["up"],
            "ma_7": np.array([1.0]),
            "ma_26": np.array([1.0]),
            "macd": np.array([0.1]),
            "macd_signal": np.array([0.05]),
            "macd_hist": np.array([0.05]),
            "adx": np.array([25.0]),
            "obv_divergence": np.array([0]),
            "bullish_divergence": np.array([False]),
            "bearish_divergence": np.array([False]),
            "divergence_strength": np.array([0.0]),
            "divergence_age_bars": np.array([0]),
            "rsi_divergence": np.array([0]),
            "rsi_divergence_strength": np.array([0.0]),
            "rsi_divergence_age": np.array([0]),
            "vwio": np.array([0.0]),
            "trend_confirmed": np.array([True]),
            "ranging_market": np.array([False]),
            "volatility_percentile": np.array([60.0]),
            "compression_score": np.array([45.0]),
            "expansion_score": np.array([70.0]),
            "volatility_state": np.array(["expanding"], dtype=object),
            "vwrs": np.array([0.2]),
            "momentum_5m": np.array([0.01]),
            "trend_5m": np.array([1]),
            "short_tf_ready_5m": np.array([True]),
            "momentum_15m": np.array([0.02]),
            "trend_15m": np.array([1]),
            "short_tf_ready_15m": np.array([True]),
            "finbert_score": np.array([0.0]),
            "xgb_score": np.array([50.0]),
            "struct_l1": {
                "ema_9": np.array([1.0]),
                "ema_20": np.array([1.0]),
                "ema_26": np.array([1.0]),
                "ema9_above_ema20": np.array([True]),
                "ema9_above_ema26": np.array([True]),
                "price_above_ema20": np.array([True]),
                "ema_slope_9": np.array([0.01]),
                "ema_cross_distance_pct": np.array([0.01]),
                "range_pos_1h": np.array([0.6]),
                "range_pos_4h": np.array([0.6]),
                "range_breakout_1h": np.array([True]),
                "higher_low_count": np.array([3]),
                "pivot_break": np.array([True]),
                "pullback_hold": np.array([True]),
            },
            "struct_l23": {
                "ema_9": np.array([1.0]),
                "ema_20": np.array([1.0]),
                "ema_26": np.array([1.0]),
                "ema9_above_ema20": np.array([True]),
                "ema9_above_ema26": np.array([True]),
                "price_above_ema20": np.array([True]),
                "ema_slope_9": np.array([0.01]),
                "ema_cross_distance_pct": np.array([0.01]),
                "range_pos_1h": np.array([0.6]),
                "range_pos_4h": np.array([0.6]),
                "range_breakout_1h": np.array([True]),
                "higher_low_count": np.array([3]),
                "pivot_break": np.array([True]),
                "pullback_hold": np.array([True]),
            },
            "struct_l4": {
                "ema_9": np.array([1.0]),
                "ema_20": np.array([1.0]),
                "ema_26": np.array([1.0]),
                "ema9_above_ema20": np.array([True]),
                "ema9_above_ema26": np.array([True]),
                "price_above_ema20": np.array([True]),
                "ema_slope_9": np.array([0.01]),
                "ema_cross_distance_pct": np.array([0.01]),
                "range_pos_1h": np.array([0.6]),
                "range_pos_4h": np.array([0.6]),
                "range_breakout_1h": np.array([True]),
                "higher_low_count": np.array([3]),
                "pivot_break": np.array([True]),
                "pullback_hold": np.array([True]),
            },
        }

        features = slice_features_for_asset(features_batch, 0, lane_hint="L3")
        self.assertEqual(features["expansion_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
