from __future__ import annotations

import unittest
from unittest.mock import patch

from core.llm.micro_prompts import Phi3LaneAdvice
from core.llm.phi3_scan import ScanCandidate, phi3_supervise_lanes


class LaneSupervisionCalibrationTests(unittest.TestCase):
    def test_major_symbol_soft_l4_relabel_does_not_conflict(self) -> None:
        candidate = ScanCandidate(
            symbol="BTC/USD",
            lane="L3",
            momentum_5=0.011,
            momentum_14=0.006,
            momentum_30=0.003,
            rotation_score=0.52,
            rsi=69.0,
            volume=1_000_000.0,
            trend_1h=1,
            regime_7d="trending",
            macro_30d="bull",
        )
        with patch(
            "core.llm.phi3_scan.phi3_supervise_lane",
            return_value=Phi3LaneAdvice(
                lane_candidate="meme",
                lane_confidence=0.91,
                lane_conflict=True,
                narrative_tag="early_acceleration",
                reason="mock",
            ),
        ):
            result = phi3_supervise_lanes([candidate])[0]

        self.assertEqual(result.lane_candidate, "L3")
        self.assertFalse(result.lane_conflict)

    def test_adjacent_lane_mismatch_stays_soft_even_with_high_confidence(self) -> None:
        candidate = ScanCandidate(
            symbol="NEAR/USD",
            lane="L3",
            momentum_5=0.02,
            momentum_14=0.01,
            momentum_30=0.01,
            rotation_score=0.8,
            rsi=64.0,
            volume=500_000.0,
            trend_1h=1,
            regime_7d="trending",
            macro_30d="bull",
        )
        with patch(
            "core.llm.phi3_scan.phi3_supervise_lane",
            return_value=Phi3LaneAdvice(
                lane_candidate="breakout",
                lane_confidence=0.95,
                lane_conflict=True,
                narrative_tag="trend_continuation",
                reason="mock",
            ),
        ):
            result = phi3_supervise_lanes([candidate])[0]

        self.assertEqual(result.lane_candidate, "L1")
        self.assertFalse(result.lane_conflict)


if __name__ == "__main__":
    unittest.main()
