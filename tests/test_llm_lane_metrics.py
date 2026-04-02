import unittest

from apps.operator_ui.main import _summarize_lane_metrics


class LLMLaneMetricsTests(unittest.TestCase):
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
