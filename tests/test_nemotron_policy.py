import unittest

from core.policy.nemotron_gate import should_run_nemotron
from core.risk.portfolio import PositionState


class NemotronPolicyTests(unittest.TestCase):
    def test_should_run_nemotron_allows_buy_candidate_with_positive_technicals(self) -> None:
        allowed = should_run_nemotron(
            symbol="TEST/USD",
            features={
                "entry_recommendation": "BUY",
                "reversal_risk": "MEDIUM",
                "entry_score": 61.0,
                "rotation_score": 0.12,
                "momentum_5": 0.01,
                "trend_confirmed": True,
            },
            positions_state=PositionState(),
            universe_context={"current_symbol_is_top_candidate": False},
            candidate_review={
                "promotion_decision": "demote",
                "priority": 0.2,
                "action_bias": "hold_preferred",
                "reason": "light_advisory_pre_filter",
            },
        )
        self.assertTrue(allowed)

    def test_should_run_nemotron_still_blocks_watch_without_real_promotion(self) -> None:
        allowed = should_run_nemotron(
            symbol="TEST/USD",
            features={
                "entry_recommendation": "WATCH",
                "reversal_risk": "MEDIUM",
                "entry_score": 68.0,
                "rotation_score": 0.2,
                "momentum_5": 0.01,
                "trend_confirmed": True,
            },
            positions_state=PositionState(),
            universe_context={"current_symbol_is_top_candidate": False},
            candidate_review={
                "promotion_decision": "demote",
                "priority": 0.2,
                "action_bias": "hold_preferred",
                "reason": "light_advisory_pre_filter",
            },
        )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
