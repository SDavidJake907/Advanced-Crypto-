import unittest

from core.policy.candidate_score import score_candidate
from core.policy.pipeline import apply_policy_pipeline
from core.policy.verdict import extract_policy_verdict


class PolicyPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_explicit_lane(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L2",
                "momentum": 0.01,
                "momentum_5": 0.02,
                "momentum_14": 0.01,
                "momentum_30": 0.0,
                "trend_1h": 1,
                "volume_ratio": 1.3,
                "price_zscore": -1.0,
                "rsi": 45.0,
                "regime_7d": "choppy",
                "macro_30d": "sideways",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.1,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.5,
                "autocorr": 0.0,
                "entropy": 0.5,
            },
        )
        self.assertEqual(policy["lane"], "L2")

    def test_candidate_score_uses_policy_pipeline_fields(self) -> None:
        scored = score_candidate(
            lane="L3",
            momentum_5=0.02,
            momentum_14=0.015,
            momentum_30=0.01,
            trend_1h=1,
            volume_ratio=1.6,
            price_zscore=0.4,
            rsi=58.0,
            spread_bps=8.0,
            rotation_score=0.12,
        )
        self.assertGreaterEqual(scored.candidate_score, 50.0)
        self.assertIn(scored.candidate_recommendation, {"BUY", "STRONG_BUY", "WATCH"})
        self.assertIn(scored.candidate_risk, {"LOW", "MEDIUM", "HIGH"})
        self.assertTrue(scored.candidate_reasons)

    def test_pipeline_attaches_policy_verdict(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "momentum": 0.01,
                "momentum_5": 0.02,
                "momentum_14": 0.01,
                "momentum_30": 0.0,
                "trend_1h": 1,
                "volume_ratio": 1.3,
                "price_zscore": -1.0,
                "rsi": 45.0,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.1,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.6,
                "autocorr": 0.1,
                "entropy": 0.5,
            },
        )
        verdict = extract_policy_verdict(policy)
        self.assertEqual(policy["policy_verdict"]["lane"], "L3")
        self.assertEqual(verdict.lane, "L3")
        self.assertEqual(verdict.entry_recommendation, policy["entry_recommendation"])
        self.assertEqual(verdict.reversal_risk, policy["reversal_risk"])

    def test_entry_reasons_are_not_silently_truncated(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "momentum": 0.03,
                "momentum_5": 0.03,
                "momentum_14": 0.02,
                "momentum_30": 0.01,
                "trend_1h": 1,
                "volume_ratio": 1.6,
                "price_zscore": 1.8,
                "rsi": 80.0,
                "regime_7d": "choppy",
                "macro_30d": "bull",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.2,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [0.5, 0.4, 0.3],
                "hurst": 0.6,
                "autocorr": 0.1,
                "entropy": 0.5,
                "lane_filter_pass": False,
                "lane_filter_reason": "lane3_rsi_hot",
            },
        )
        reasons = policy["entry_reasons"]
        self.assertGreater(len(reasons), 6)
        self.assertIn("regime_choppy", reasons)
        self.assertIn("macro_bull", reasons)
        self.assertIn("persistent_trend", reasons)
        self.assertIn("overbought_penalty", reasons)
        self.assertIn("fast_momentum", reasons)
        self.assertIn("volume_expansion", reasons)
        self.assertIn("rotation_leader", reasons)
        self.assertIn("lane3_rsi_hot", reasons)


if __name__ == "__main__":
    unittest.main()
