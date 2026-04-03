from __future__ import annotations

import unittest

from core.policy.lane_classifier import classify_lane
from core.policy.entry_verifier import compute_entry_verification
from core.policy.lane_filters import apply_lane_filters
from core.policy.nemotron_gate import should_run_nemotron
from core.risk.portfolio import PositionState


class LaneIdentityTests(unittest.TestCase):
    def test_midcap_rotation_setup_maps_to_lane2(self) -> None:
        lane = classify_lane(
            "APT/USD",
            {
                "momentum_5": 0.004,
                "momentum_14": 0.006,
                "momentum_30": 0.001,
                "trend_1h": 0,
                "volume_ratio": 0.95,
                "rotation_score": 0.08,
                "price_zscore": 0.4,
                "regime_7d": "unknown",
                "rsi": 58.0,
                "price": 5.0,
                "bb_lower": 4.8,
                "bb_upper": 5.2,
            },
        )
        self.assertEqual(lane, "L2")

    def test_balanced_setup_maps_to_lane3(self) -> None:
        lane = classify_lane(
            "TON/USD",
            {
                "momentum_5": 0.001,
                "momentum_14": 0.004,
                "momentum_30": 0.003,
                "trend_1h": 0,
                "volume_ratio": 0.9,
                "rotation_score": 0.0,
                "price_zscore": 0.2,
                "regime_7d": "unknown",
                "rsi": 54.0,
                "price": 3.0,
                "bb_lower": 2.9,
                "bb_upper": 3.1,
            },
        )
        self.assertEqual(lane, "L3")

    def test_hot_balanced_candidate_does_not_fall_into_lane3(self) -> None:
        lane = classify_lane(
            "TON/USD",
            {
                "momentum_5": 0.011,
                "momentum_14": 0.004,
                "momentum_30": 0.003,
                "trend_1h": 0,
                "volume_ratio": 0.9,
                "rotation_score": 0.02,
                "price_zscore": 0.3,
                "regime_7d": "unknown",
                "rsi": 57.0,
                "price": 3.0,
                "bb_lower": 2.9,
                "bb_upper": 3.1,
            },
        )
        self.assertNotEqual(lane, "L3")

    def test_lane4_filter_allows_hot_symbol_with_modest_volume_ratio(self) -> None:
        result = apply_lane_filters(
            {
                "lane": "L4",
                "volume_ratio": 1.0,
                "volume_surge": 0.4,
                "momentum_5": 0.004,
                "spread_pct": 1.5,
                "sentiment_symbol_trending": True,
            }
        )
        self.assertTrue(result.passed)

    def test_lane2_filter_allows_midcap_rotation_mover(self) -> None:
        result = apply_lane_filters(
            {
                "lane": "L2",
                "volume_ratio": 0.9,
                "volume_surge": 0.2,
                "momentum_5": 0.004,
                "momentum_14": 0.006,
                "trend_1h": 0,
                "rsi": 60.0,
                "atr": 0.1,
                "price": 5.0,
                "bb_bandwidth": 0.04,
                "spread_pct": 0.6,
                "price_zscore": 0.4,
            }
        )
        self.assertTrue(result.passed)

    def test_lane2_low_volume_is_soft_downgrade_not_hard_block(self) -> None:
        result = apply_lane_filters(
            {
                "lane": "L2",
                "volume_ratio": 0.12,
                "volume_surge": 0.04,
                "momentum_5": 0.002,
                "momentum_14": 0.003,
                "trend_1h": 0,
                "rsi": 57.0,
                "atr": 0.08,
                "price": 5.0,
                "spread_pct": 0.4,
            }
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "lane2_vol_low")
        self.assertEqual(result.severity, "soft")

    def test_lane3_filter_rejects_hot_setup(self) -> None:
        result = apply_lane_filters(
            {
                "lane": "L3",
                "volume_ratio": 0.95,
                "volume_surge": 0.55,
                "momentum_5": 0.015,
                "momentum_14": 0.005,
                "trend_1h": 0,
                "rsi": 58.0,
                "atr": 0.08,
                "price": 2.0,
                "spread_pct": 0.4,
            }
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "lane3_heat_high")

    def test_lane3_low_volume_is_soft_downgrade_not_hard_block(self) -> None:
        result = apply_lane_filters(
            {
                "lane": "L3",
                "volume_ratio": 0.35,
                "volume_surge": 0.02,
                "momentum_5": 0.002,
                "momentum_14": 0.004,
                "trend_1h": 0,
                "rsi": 55.0,
                "atr": 0.05,
                "price": 3.0,
                "spread_pct": 0.4,
            }
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "lane3_vol_low_warning")
        self.assertEqual(result.severity, "soft")

    def test_lane4_entry_verifier_promotes_hot_setup_more_easily(self) -> None:
        verdict = compute_entry_verification(
            {
                "symbol": "DOGE/USD",
                "lane": "L4",
                "lane_filter_pass": True,
                "regime_state": "bullish",
                "momentum_5": 0.011,
                "momentum_14": 0.006,
                "momentum_30": 0.002,
                "rotation_score": 0.15,
                "trend_1h": 0,
                "rsi": 69.0,
                "volume_ratio": 1.05,
                "volume_surge": 0.8,
                "price_zscore": 1.2,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "hurst": 0.6,
                "autocorr": 0.2,
                "entropy": 0.5,
                "correlation_row": [0.2, 0.3],
                "book_imbalance": 0.25,
                "book_wall_pressure": 0.1,
                "sentiment_market_cap_change_24h": 1.2,
                "sentiment_symbol_trending": True,
                "finbert_score": 0.3,
                "xgb_score": 58.0,
            }
        )
        self.assertIn(verdict["entry_recommendation"], {"BUY", "STRONG_BUY"})
        self.assertIn("meme_heat", verdict["entry_reasons"])

    def test_lane4_can_reach_nemotron_with_heat(self) -> None:
        allowed = should_run_nemotron(
            symbol="DOGE/USD",
            features={
                "lane": "L4",
                "entry_recommendation": "WATCH",
                "promotion_tier": "probe",
                "reversal_risk": "MEDIUM",
                "entry_score": 46.0,
                "rotation_score": 0.05,
                "momentum_5": 0.01,
                "volume_surge": 0.5,
                "sentiment_symbol_trending": True,
                "trend_confirmed": False,
            },
            positions_state=PositionState(),
            universe_context={"current_symbol_is_top_candidate": False},
        )
        self.assertTrue(allowed)

    def test_lane2_entry_verifier_promotes_rotation_earlier_than_main_lane(self) -> None:
        lane2 = compute_entry_verification(
            {
                "symbol": "APT/USD",
                "lane": "L2",
                "lane_filter_pass": True,
                "regime_state": "unknown",
                "momentum_5": 0.003,
                "momentum_14": 0.004,
                "momentum_30": 0.001,
                "rotation_score": 0.09,
                "trend_1h": 0,
                "rsi": 57.0,
                "volume_ratio": 0.95,
                "volume_surge": 0.22,
                "price_zscore": 0.4,
                "regime_7d": "unknown",
                "macro_30d": "sideways",
                "hurst": 0.52,
                "autocorr": 0.08,
                "entropy": 0.55,
                "correlation_row": [0.2, 0.1],
                "xgb_score": 56.0,
            }
        )
        lane3 = compute_entry_verification(
            {
                "symbol": "TRX/USD",
                "lane": "L3",
                "lane_filter_pass": True,
                "regime_state": "unknown",
                "momentum_5": 0.003,
                "momentum_14": 0.004,
                "momentum_30": 0.001,
                "rotation_score": 0.09,
                "trend_1h": 0,
                "rsi": 57.0,
                "volume_ratio": 0.95,
                "volume_surge": 0.22,
                "price_zscore": 0.4,
                "regime_7d": "unknown",
                "macro_30d": "sideways",
                "hurst": 0.52,
                "autocorr": 0.08,
                "entropy": 0.55,
                "correlation_row": [0.2, 0.1],
                "xgb_score": 56.0,
            }
        )
        self.assertIn(lane2["entry_recommendation"], {"BUY", "STRONG_BUY"})
        self.assertIn(lane3["entry_recommendation"], {"WATCH", "BUY"})
        self.assertGreaterEqual(lane2["entry_score"], lane3["entry_score"])

    def test_unpromoted_watch_fails_nemotron_gate(self) -> None:
        base_features = {
            "entry_recommendation": "WATCH",
            "promotion_tier": "skip",
            "reversal_risk": "MEDIUM",
            "entry_score": 58.0,
            "rotation_score": 0.04,
            "momentum_5": 0.002,
            "volume_surge": 0.15,
            "sentiment_symbol_trending": False,
            "trend_confirmed": False,
            "momentum_14": 0.003,
        }
        allowed_l1 = should_run_nemotron(
            symbol="BTC/USD",
            features={"lane": "L1", **base_features},
            positions_state=PositionState(),
            universe_context={"current_symbol_is_top_candidate": False},
        )
        allowed_l2 = should_run_nemotron(
            symbol="APT/USD",
            features={"lane": "L2", **base_features},
            positions_state=PositionState(),
            universe_context={"current_symbol_is_top_candidate": False},
        )
        self.assertFalse(allowed_l1)
        self.assertFalse(allowed_l2)


if __name__ == "__main__":
    unittest.main()
