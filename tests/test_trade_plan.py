import unittest

from core.policy.trade_plan import build_trade_plan_metadata, infer_expected_hold_style


class TradePlanTests(unittest.TestCase):
    def test_l1_strong_structure_becomes_leader_runner(self) -> None:
        hold_style = infer_expected_hold_style(
            {
                "lane": "L1",
                "breakout_confirmed": True,
                "structure_build": True,
                "ema9_above_ema26": True,
                "continuation_quality": 75.0,
                "structure_quality": 78.0,
            }
        )
        self.assertEqual(hold_style, "leader_runner")

    def test_trade_plan_metadata_includes_timeframe_roles(self) -> None:
        metadata = build_trade_plan_metadata(
            "LONG",
            {
                "lane": "L2",
                "governing_timeframe": "1h",
                "macro_timeframe": "4h",
                "trigger_timeframe": "15m",
                "expected_move_pct": 3.2,
                "total_cost_pct": 1.1,
                "required_edge_pct": 2.4,
                "net_edge_pct": 2.1,
                "tp_after_cost_valid": True,
            },
        )
        self.assertEqual(metadata["governing_timeframe"], "1h")
        self.assertEqual(metadata["macro_timeframe"], "4h")
        self.assertEqual(metadata["trigger_timeframe"], "15m")
        self.assertEqual(metadata["expected_move_pct"], 3.2)
        self.assertEqual(metadata["total_cost_pct"], 1.1)
        self.assertEqual(metadata["required_edge_pct"], 2.4)
        self.assertEqual(metadata["net_edge_pct"], 2.1)
        self.assertTrue(metadata["tp_after_cost_valid"])


if __name__ == "__main__":
    unittest.main()
