import unittest
from unittest.mock import patch

from core.llm.orchestrator import build_advisory_bundle


class AdvisoryBundleTests(unittest.TestCase):
    @patch("core.llm.orchestrator._load_recent_visual_review", return_value={"chart_state": "clean"})
    @patch("core.llm.orchestrator.phi3_reflex")
    def test_build_advisory_bundle_uses_deterministic_market_state_review(self, mock_reflex, _mock_visual) -> None:
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
                "sentiment_symbol_trending": False,
            },
            universe_context={"current_symbol_is_top_candidate": True},
        )
        self.assertEqual(bundle.reflex["reason"], "data_ok")
        self.assertEqual(bundle.market_state_review["market_state"], "ranging")
        self.assertEqual(bundle.market_state_review["lane_bias"], "favor_selective")
        self.assertEqual(bundle.market_state_review["reason"], "range_state_but_mover_present")


if __name__ == "__main__":
    unittest.main()
