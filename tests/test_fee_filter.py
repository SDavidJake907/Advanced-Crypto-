from __future__ import annotations

import unittest
from unittest.mock import patch

from core.risk.fee_filter import evaluate_trade_cost


class FeeFilterTests(unittest.TestCase):
    def test_trade_cost_blocks_when_edge_does_not_clear_cost_floor(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
            assessment = evaluate_trade_cost(
                {
                    "lane": "L3",
                    "price": 100.0,
                    "atr": 0.05,
                    "spread_pct": 0.3,
                    "structure_quality": 40.0,
                    "continuation_quality": 40.0,
                    "momentum_quality": 35.0,
                    "trade_quality": 60.0,
                },
                "LONG",
            )
        self.assertFalse(assessment.actionable)
        self.assertTrue(any("edge_below_cost_floor" in reason for reason in assessment.reasons))

    def test_trade_cost_allows_when_edge_clears_cost_floor(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
            assessment = evaluate_trade_cost(
                {
                    "lane": "L2",
                    "price": 100.0,
                    "atr": 2.0,
                    "spread_pct": 0.08,
                    "structure_quality": 82.0,
                    "continuation_quality": 78.0,
                    "momentum_quality": 74.0,
                    "trade_quality": 95.0,
                },
                "LONG",
            )
        self.assertTrue(assessment.actionable)
        self.assertGreater(assessment.expected_edge_pct, assessment.total_cost_pct)
        self.assertGreater(assessment.net_edge_pct, 0.0)

    def test_trade_cost_uses_consistent_percent_units_for_fees_and_slippage(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
            assessment = evaluate_trade_cost(
                {
                    "lane": "L3",
                    "price": 100.0,
                    "atr": 1.0,
                    "spread_pct": 0.10,
                    "structure_quality": 60.0,
                    "continuation_quality": 60.0,
                    "momentum_quality": 60.0,
                    "trade_quality": 60.0,
                },
                "LONG",
            )
        self.assertGreater(assessment.fee_round_trip_pct, 0.50)
        self.assertGreater(assessment.slippage_pct, 0.05)
        self.assertGreater(assessment.total_cost_pct, 0.80)

    def test_trade_cost_can_be_relaxed_for_stabilization_profile(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "TRADE_COST_MIN_EDGE_MULT": 0.05,
                "TRADE_COST_MIN_EXPECTED_EDGE_PCT": 0.03,
                "TRADE_COST_SAFETY_BUFFER_PCT": 0.0,
                "TRADE_COST_ASSUME_AGGRESSIVE_ENTRY_TAKER": False,
            },
        ):
            assessment = evaluate_trade_cost(
                {
                    "lane": "L2",
                    "price": 66860.3,
                    "atr": 7.664,
                    "spread_pct": 0.0001496,
                    "structure_quality": 60.64,
                    "continuation_quality": 63.78,
                    "momentum_quality": 65.0,
                    "trade_quality": 99.92,
                    "promotion_tier": "promote",
                    "entry_recommendation": "BUY",
                },
                "LONG",
            )

        self.assertTrue(assessment.actionable)
        self.assertGreater(assessment.expected_edge_pct, assessment.required_edge_pct)


if __name__ == "__main__":
    unittest.main()
