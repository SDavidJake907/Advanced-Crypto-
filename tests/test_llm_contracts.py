import unittest

from core.llm.contracts import (
    DEFER_REASON_CONFLICTING_SIGNALS,
    DEFER_REASON_EXECUTION_UNSUITABLE,
    ROLE_CANDIDATE_REVIEWER,
    ROLE_RUNTIME_ADVISOR,
    ROLE_TRADE_REVIEWER,
    classify_defer_reason,
    normalize_candidate_review,
    normalize_runtime_advice,
    normalize_trade_reviewer_output,
)


class LLMContractsTests(unittest.TestCase):
    def test_trade_reviewer_normalizes_to_contract_envelope(self) -> None:
        parsed = normalize_trade_reviewer_output(
            {
                "final_decision": {
                    "symbol": "BTC/USD",
                    "action": "OPEN",
                    "side": "BUY",
                    "size": 0.4,
                    "reason": "good_setup",
                    "contradictions": ["mixed_regime"],
                }
            },
            symbol="BTC/USD",
        )
        decision = parsed["final_decision"]
        self.assertEqual(decision["action"], "OPEN")
        self.assertEqual(decision["side"], "LONG")
        self.assertEqual(decision["contract"]["role"], ROLE_TRADE_REVIEWER)
        self.assertEqual(decision["contract"]["reasons"], ["good_setup"])
        self.assertEqual(decision["contract"]["contradictions"], ["mixed_regime"])

    def test_trade_reviewer_normalizes_placeholder_symbol(self) -> None:
        parsed = normalize_trade_reviewer_output(
            {
                "final_decision": {
                    "symbol": "CURRENT_SYMBOL",
                    "action": "HOLD",
                    "side": None,
                    "size": 0,
                    "reason": "trend_unconfirmed",
                }
            },
            symbol="TEST/USD",
        )
        self.assertEqual(parsed["final_decision"]["symbol"], "TEST/USD")
        self.assertIn("symbol", parsed["final_decision"]["contract"]["meta"]["normalized_fields"])

    def test_trade_reviewer_defaults_reason_when_missing(self) -> None:
        parsed = normalize_trade_reviewer_output(
            {
                "final_decision": {
                    "symbol": "CURRENT_SYMBOL",
                    "action": "HOLD",
                    "side": None,
                    "size": 0,
                }
            },
            symbol="TEST/USD",
        )
        self.assertEqual(parsed["final_decision"]["reason"], "hold_unspecified")

    def test_candidate_review_normalizes_invalid_fields(self) -> None:
        parsed = normalize_candidate_review(
            {
                "promotion_decision": "bad_value",
                "priority": 2.5,
                "action_bias": "bad_bias",
                "reason": "candidate_review",
            }
        )
        self.assertEqual(parsed["promotion_decision"], "neutral")
        self.assertEqual(parsed["action_bias"], "hold_preferred")
        self.assertEqual(parsed["priority"], 1.0)
        self.assertEqual(parsed["contract"]["role"], ROLE_CANDIDATE_REVIEWER)

    def test_runtime_advice_filters_unapproved_keys(self) -> None:
        parsed = normalize_runtime_advice(
            {
                "summary": "small tune",
                "recommended_overrides": {"GOOD_KEY": 1, "BAD_KEY": 2},
                "issues": ["chop"],
                "confidence": 0.7,
            },
            allowed_keys={"GOOD_KEY"},
        )
        self.assertEqual(parsed["recommended_overrides"], {"GOOD_KEY": 1})
        self.assertEqual(parsed["issues"], ["chop"])
        self.assertEqual(parsed["contract"]["role"], ROLE_RUNTIME_ADVISOR)

    def test_defer_reason_taxonomy_classifies_common_cases(self) -> None:
        self.assertEqual(classify_defer_reason("mixed regime conflict"), DEFER_REASON_CONFLICTING_SIGNALS)
        self.assertEqual(classify_defer_reason("spread too wide for execution"), DEFER_REASON_EXECUTION_UNSUITABLE)


if __name__ == "__main__":
    unittest.main()
