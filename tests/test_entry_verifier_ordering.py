from __future__ import annotations

import unittest

from core.policy.entry_verifier import compute_entry_verification


def _base_features() -> dict[str, object]:
    return {
        "symbol": "TEST/USD",
        "lane": "L3",
        "lane_filter_pass": True,
        "lane_filter_reason": "",
        "regime_state": "neutral",
        "regime_7d": "unknown",
        "macro_30d": "sideways",
        "atr_expanding": False,
        "atr_pct": 0.012,
        "volatility_percentile": 50.0,
        "compression_score": 50.0,
        "expansion_score": 50.0,
        "volatility_state": "normal",
        "momentum": 0.0,
        "momentum_5": 0.0,
        "momentum_14": 0.0,
        "momentum_30": 0.0,
        "rotation_score": 0.0,
        "trend_1h": 0,
        "rsi": 55.0,
        "volume_ratio": 1.0,
        "volume_surge": 0.0,
        "price_zscore": 0.4,
        "hurst": 0.5,
        "autocorr": 0.0,
        "entropy": 0.5,
        "correlation_row": [],
        "book_imbalance": 0.0,
        "book_wall_pressure": 0.0,
        "sentiment_market_cap_change_24h": 0.0,
        "sentiment_symbol_trending": False,
        "ema9_above_ema20": False,
        "price_above_ema20": False,
        "ema_slope_9": 0.0,
        "short_tf_ready_5m": True,
        "short_tf_ready_15m": True,
        "range_pos_1h": 0.55,
        "range_pos_4h": 0.55,
        "range_breakout_1h": False,
        "higher_low_count": 1,
        "pivot_break": False,
        "pullback_hold": False,
        "structure_quality": 50.0,
        "momentum_quality": 50.0,
        "volume_quality": 50.0,
        "trade_quality": 55.0,
        "market_support": 50.0,
        "continuation_quality": 50.0,
        "risk_quality": 52.0,
        "price": 100.0,
        "atr": 1.2,
        "spread_pct": 0.2,
        "indicators_ready": True,
        "trend_confirmed": False,
        "vwio": 0.0,
        "obv_divergence": 0,
        "bullish_divergence": False,
        "bearish_divergence": False,
        "divergence_strength": 0.0,
        "divergence_age_bars": 99,
        "rsi_divergence": 0,
    }


class EntryVerifierOrderingTests(unittest.TestCase):
    def test_weak_structure_caps_confirmation_bonuses(self) -> None:
        weak_structure = _base_features()
        weak_structure.update(
            {
                "momentum_5": 0.012,
                "momentum_14": 0.01,
                "momentum_30": 0.006,
                "volume_ratio": 1.8,
                "volume_surge": 0.7,
                "trend_1h": 1,
                "structure_quality": 48.0,
                "continuation_quality": 49.0,
                "momentum_quality": 70.0,
                "volume_quality": 72.0,
            }
        )
        strong_structure = _base_features()
        strong_structure.update(
            {
                "momentum_5": 0.006,
                "momentum_14": 0.005,
                "momentum_30": 0.003,
                "volume_ratio": 1.1,
                "volume_surge": 0.12,
                "trend_1h": 1,
                "ema9_above_ema20": True,
                "price_above_ema20": True,
                "ema_slope_9": 0.002,
                "higher_low_count": 4,
                "pullback_hold": True,
                "structure_quality": 72.0,
                "continuation_quality": 70.0,
                "trade_quality": 62.0,
                "risk_quality": 60.0,
            }
        )

        weak_result = compute_entry_verification(weak_structure)
        strong_result = compute_entry_verification(strong_structure)

        self.assertIn("confirmation_capped_by_structure", weak_result["entry_reasons"])
        self.assertGreater(strong_result["entry_score"], weak_result["entry_score"])

    def test_strong_structure_keeps_full_confirmation_path(self) -> None:
        features = _base_features()
        features.update(
            {
                "momentum_5": 0.009,
                "momentum_14": 0.008,
                "volume_ratio": 1.3,
                "volume_surge": 0.25,
                "trend_1h": 1,
                "ema9_above_ema20": True,
                "price_above_ema20": True,
                "ema_slope_9": 0.002,
                "higher_low_count": 3,
                "range_breakout_1h": True,
                "structure_quality": 68.0,
                "continuation_quality": 66.0,
                "trade_quality": 60.0,
                "risk_quality": 58.0,
                "trend_confirmed": True,
            }
        )

        result = compute_entry_verification(features)

        self.assertNotIn("confirmation_capped_by_structure", result["entry_reasons"])
        self.assertGreaterEqual(result["entry_score"], 54.0)

    def test_compressed_breakout_scores_better_than_overheated_late_extension(self) -> None:
        compressed = _base_features()
        compressed.update(
            {
                "trend_1h": 1,
                "trend_confirmed": True,
                "range_breakout_1h": True,
                "higher_low_count": 4,
                "structure_quality": 68.0,
                "continuation_quality": 66.0,
                "compression_score": 78.0,
                "expansion_score": 44.0,
                "volatility_percentile": 35.0,
                "volatility_state": "compressed",
            }
        )
        overheated = _base_features()
        overheated.update(
            {
                "trend_1h": 1,
                "trend_confirmed": True,
                "range_breakout_1h": True,
                "structure_quality": 68.0,
                "continuation_quality": 66.0,
                "price_zscore": 1.9,
                "range_pos_1h": 0.95,
                "range_pos_4h": 0.94,
                "atr_expanding": True,
                "atr_pct": 0.055,
                "compression_score": 22.0,
                "expansion_score": 82.0,
                "volatility_percentile": 96.0,
                "volatility_state": "overheated",
            }
        )

        compressed_result = compute_entry_verification(compressed)
        overheated_result = compute_entry_verification(overheated)

        self.assertIn("vol_compression_setup", compressed_result["entry_reasons"])
        self.assertIn("vol_overheated", overheated_result["entry_reasons"])
        self.assertGreater(compressed_result["entry_score"], overheated_result["entry_score"])


if __name__ == "__main__":
    unittest.main()
