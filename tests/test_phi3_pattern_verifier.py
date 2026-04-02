from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from core.llm.phi3_pattern_verifier import verify_pattern_evidence


class Phi3PatternVerifierTests(unittest.TestCase):
    def test_verifier_parses_phi3_json(self) -> None:
        payload = {
            "validity": "valid",
            "confidence": 0.84,
            "structure_validity": "valid",
            "structure_confidence": 0.82,
            "candle_confirmation_validity": "valid",
            "candle_confirmation_score": 0.74,
            "pattern_tradeability_score": 0.78,
            "location_quality_score": 0.81,
            "breakout_quality_score": 0.76,
            "retest_quality_score": 0.71,
            "extension_risk_score": 0.4,
            "overall_bias": "bullish",
            "warnings": [],
            "reasons_for_validity": ["breakout confirmed"],
            "reasons_against_validity": [],
            "missing_confirmation": [],
            "conflicts": [],
            "recommended_nemo_interpretation": {"structure_bonus": 7, "candle_bonus": 3, "skepticism_penalty": 0, "prefer_action": "OPEN"},
            "summary": "Clean double bottom.",
        }
        with patch("core.llm.phi3_pattern_verifier.phi3_chat", return_value=json.dumps(payload)):
            result = verify_pattern_evidence({"pattern": "double_bottom"})

        self.assertEqual(result["validity"], "valid")
        self.assertEqual(result["structure_validity"], "valid")
        self.assertAlmostEqual(result["pattern_tradeability_score"], 0.64, places=6)
        self.assertAlmostEqual(result["extension_risk_score"], 0.4, places=6)
        self.assertAlmostEqual(result["pattern_quality_score"], 0.801, places=3)


if __name__ == "__main__":
    unittest.main()
