import unittest
from unittest.mock import patch

from core.llm.orchestrator import build_advisory_bundle


class AdvisoryBundleTests(unittest.TestCase):
    @patch("core.llm.orchestrator.phi3_reflex")
    def test_build_advisory_bundle_uses_deterministic_market_state_review(self, mock_reflex) -> None:
        mock_reflex.return_value.to_dict.return_value = {
            "reflex": "allow",
            "micro_state": "stable",
            "reason": "data_ok",
        }
        bundle = build_advisory_bundle(
            symbol="CRV/USD",
            features={
                "entry_recommendation": "BUY",
                "reversal_risk": "MEDIUM",
                "entry_score": 58.0,
                "volume_ratio": 1.1,
                "rotation_score": 0.05,
                "momentum_5": 0.01,
                "trend_confirmed": False,
                "ranging_market": True,
                "volume_surge": 0.1,
                "volume_ratio": 1.1,
                "sentiment_symbol_trending": False,
            },
            universe_context={"current_symbol_is_top_candidate": True},
        )
        self.assertEqual(bundle.reflex["reason"], "data_ok")
        self.assertEqual(bundle.market_state_review["market_state"], "ranging")
        self.assertEqual(bundle.market_state_review["lane_bias"], "favor_selective")
        self.assertEqual(bundle.market_state_review["reason"], "selective_range_mover_with_relative_strength")
        self.assertEqual(bundle.market_state_review["breakout_state"], "breakout_attempt")
        self.assertEqual(bundle.market_state_review["trend_stage"], "mixed")
        self.assertEqual(bundle.market_state_review["volume_confirmation"], "neutral")
        self.assertEqual(bundle.market_state_review["pattern_explanation"]["structure_phase"], "attempting_breakout")
        self.assertEqual(bundle.market_state_review["pattern_explanation"]["context"]["breakout_level_state"], "pressing")
        self.assertEqual(bundle.market_state_review["candle_evidence"]["primary_candle"], "none")

    @patch("core.llm.orchestrator.deterministic_market_state_review")
    @patch("core.llm.orchestrator.phi3_reflex")
    def test_build_advisory_bundle_reuses_precomputed_market_state_review(self, mock_reflex, mock_market_state) -> None:
        mock_reflex.return_value.to_dict.return_value = {
            "reflex": "allow",
            "micro_state": "stable",
            "reason": "data_ok",
        }
        precomputed_review = {
            "market_state": "trending",
            "lane_bias": "favor_trend",
            "reason": "cached_review",
        }

        bundle = build_advisory_bundle(
            symbol="FET/USD",
            features={"entry_score": 80.0},
            market_state_review=precomputed_review,
        )

        self.assertEqual(bundle.market_state_review, precomputed_review)
        mock_market_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
