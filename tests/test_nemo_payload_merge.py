from __future__ import annotations

import unittest

from core.policy.nemo_payload_merge import merge_candidate_with_phi3


class NemoPayloadMergeTests(unittest.TestCase):
    def test_merge_preserves_legacy_quality_fields_and_adds_richer_verification(self) -> None:
        merged = merge_candidate_with_phi3(
            {"symbol": "SOL/USD", "entry_score": 82.0},
            {
                "validity": "valid",
                "confidence": 0.83,
                "structure_validity": "valid",
                "structure_confidence": 0.82,
                "candle_confirmation_validity": "valid",
                "candle_confirmation_score": 0.74,
                "pattern_tradeability_score": 0.79,
                "location_quality_score": 0.81,
                "breakout_quality_score": 0.76,
                "retest_quality_score": 0.71,
                "extension_risk_score": 0.5,
                "overall_bias": "bullish",
                "recommended_nemo_interpretation": {"structure_bonus": 7, "candle_bonus": 3, "skepticism_penalty": 0, "prefer_action": "OPEN"},
                "summary": "Supported structure and candle context.",
            },
        )

        verification = merged["pattern_verification"]
        self.assertEqual(verification["validity"], "valid")
        self.assertEqual(verification["structure_validity"], "valid")
        self.assertAlmostEqual(verification["tradeability_score"], 0.79, places=6)
        self.assertAlmostEqual(verification["extension_risk_score"], 0.5, places=6)
        self.assertGreater(verification["pattern_quality_score"], 0.0)
        self.assertEqual(verification["recommended_nemo_interpretation"]["prefer_action"], "OPEN")
        self.assertFalse(merged["phi3_veto_flag"])
        self.assertAlmostEqual(merged["entry_score"], 87.01, places=2)


if __name__ == "__main__":
    unittest.main()
