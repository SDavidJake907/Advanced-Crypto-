from __future__ import annotations

import unittest

from core.policy.entry_verifier import compute_entry_verification
from core.policy.lane_filters import apply_lane_filters
from core.policy.nemotron_gate import should_run_nemotron
from core.risk.portfolio import PositionState


class LaneIdentityTests(unittest.TestCase):
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
            candidate_review={"promotion_decision": "neutral", "priority": 0.45},
        )
        self.assertTrue(allowed)


if __name__ == "__main__":
    unittest.main()
