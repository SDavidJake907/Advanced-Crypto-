import unittest

from apps.trader.shadow import extract_llm_metrics
from apps.operator_ui.main import _summarize_lane_metrics


class LLMLaneMetricsTests(unittest.TestCase):
    def test_market_reviewer_ranging_selective_is_defer_not_reject(self) -> None:
        metrics = extract_llm_metrics(
            "UNI/USD",
            "L3",
            {
                "nemotron": {
                    "action": "OPEN",
                    "market_state_review": {
                        "market_state": "ranging",
                        "lane_bias": "favor_selective",
                        "reason": "range_state_but_mover_present",
                        "contract": {
                            "confidence": 0.6,
                            "version": "1.0",
                            "reasons": ["range_state_but_mover_present"],
                            "contradictions": [],
                            "risks": [],
                            "meta": {
                                "schema_valid": True,
                                "normalized_field_count": 0,
                                "provider": "phi3",
                                "model": "phi3",
                                "prompt_version": "market_reviewer/v1",
                            },
                        },
                    },
                }
            },
        )
        market_role = next(item for item in metrics["roles"] if item["role"] == "market_reviewer")
        self.assertEqual(market_role["decision"], "defer")
        self.assertEqual(metrics["aggregate"]["reject"], 0)
        self.assertEqual(metrics["aggregate"]["defer"], 1)

    def test_lane_metrics_include_llm_role_aggregates(self) -> None:
        summary = _summarize_lane_metrics(
            [
                {
                    "lane": "L4",
                    "signal": "LONG",
                    "execution_status": "filled",
                    "nemotron": {"reason": "strong_entry"},
                    "llm_metrics": {
                        "roles": [
                            {
                                "decision": "defer",
                                "defer_reason_category": "weak_evidence",
                            },
                            {
                                "decision": "defer",
                                "defer_reason_category": "weak_evidence",
                            },
                        ],
                        "aggregate": {
                            "approve": 2,
                            "reject": 1,
                            "defer": 1,
                            "normalized_fields": 3,
                        }
                    },
                }
            ]
        )
        lane = summary["lanes"][0]
        self.assertEqual(lane["lane"], "L4")
        self.assertEqual(lane["opens"], 1)
        self.assertEqual(lane["llm_approve"], 2)
        self.assertEqual(lane["llm_reject"], 1)
        self.assertEqual(lane["llm_defer"], 1)
        self.assertEqual(lane["llm_normalized_fields"], 3)
        self.assertEqual(lane["llm_defer_reasons"][0]["reason"], "weak_evidence")
        self.assertEqual(lane["llm_defer_reasons"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
