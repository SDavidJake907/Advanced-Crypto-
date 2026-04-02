import unittest
from unittest.mock import patch

from core.execution.clamp import clamp_order_size
from core.execution.kraken_rules import KrakenPairRule
from core.execution.mock_exec import MockExecutor


class ExecutionGuardTests(unittest.TestCase):
    def test_mock_executor_blocks_non_positive_net_edge_even_if_cost_snapshot_is_actionable(self) -> None:
        executor = MockExecutor()
        features = {
            "price": 10.0,
            "lane": "L3",
            "spread_pct": 0.01,
            "atr": 0.1,
            "trend_confirmed": True,
            "tp_after_cost_valid": True,
            "expected_move_pct": 3.0,
            "fees_pct": 0.1,
            "slippage_pct": 0.05,
            "stop_pct": 4.0,
        }
        state = {"cash": 100.0}

        with patch("core.execution.mock_exec.evaluate_trade_cost") as mock_cost:
            mock_cost.return_value.actionable = True
            mock_cost.return_value.net_edge_pct = -0.25
            mock_cost.return_value.reasons = []
            mock_cost.return_value.to_dict.return_value = {"net_edge_pct": -0.25}
            result = executor.execute("LONG", "ALGO/USD", features, state)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], ["net_edge_non_positive"])
        self.assertEqual(result["cost"]["net_edge_pct"], -0.25)

    def test_clamp_rejects_trade_that_exceeds_risk_budget(self) -> None:
        result = clamp_order_size(
            pair="TESTUSD",
            price=100.0,
            desired_usd=25.0,
            rule=KrakenPairRule(
                wsname="TEST/USD",
                ordermin=0.0,
                lot_decimals=4,
                pair_decimals=2,
                costmin=0.0,
            ),
            equity_usd=1000.0,
            max_position_usd=1000.0,
            risk_per_trade_pct=1.0,
            min_notional_usd=10.0,
            clamp_up=True,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "exceeds_risk_guard")

    def test_clamp_rejects_zero_qty_after_rounding_even_when_ordermin_is_zero(self) -> None:
        result = clamp_order_size(
            pair="TESTUSD",
            price=100.0,
            desired_usd=0.01,
            rule=KrakenPairRule(
                wsname="TEST/USD",
                ordermin=0.0,
                lot_decimals=0,
                pair_decimals=2,
                costmin=0.0,
            ),
            equity_usd=1000.0,
            max_position_usd=1000.0,
            risk_per_trade_pct=100.0,
            min_notional_usd=0.0,
            clamp_up=False,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "rounded_qty_zero")

    def test_clamp_allows_position_when_stop_risk_fits_budget(self) -> None:
        result = clamp_order_size(
            pair="TESTUSD",
            price=100.0,
            desired_usd=25.0,
            rule=KrakenPairRule(
                wsname="TEST/USD",
                ordermin=0.0,
                lot_decimals=4,
                pair_decimals=2,
                costmin=0.0,
            ),
            equity_usd=1000.0,
            max_position_usd=1000.0,
            risk_per_trade_pct=1.0,
            min_notional_usd=10.0,
            clamp_up=True,
            risk_fraction_of_notional=0.04,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "ok")
        self.assertAlmostEqual(result.notional_usd, 25.0, places=4)

    def test_clamp_allows_min_trade_ticket_when_only_risk_guard_blocks_small_account(self) -> None:
        result = clamp_order_size(
            pair="TESTUSD",
            price=100.0,
            desired_usd=4.0,
            rule=KrakenPairRule(
                wsname="TEST/USD",
                ordermin=0.05,
                lot_decimals=4,
                pair_decimals=2,
                costmin=7.0,
            ),
            equity_usd=30.0,
            max_position_usd=30.0,
            risk_per_trade_pct=10.0,
            min_notional_usd=7.0,
            max_min_trade_risk_mult=3.0,
            clamp_up=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "ok")
        self.assertAlmostEqual(result.notional_usd, 7.0, places=4)

    def test_clamp_rejects_min_trade_that_forces_too_much_over_risk(self) -> None:
        result = clamp_order_size(
            pair="TESTUSD",
            price=100.0,
            desired_usd=4.0,
            rule=KrakenPairRule(
                wsname="TEST/USD",
                ordermin=0.05,
                lot_decimals=4,
                pair_decimals=2,
                costmin=7.0,
            ),
            equity_usd=30.0,
            max_position_usd=30.0,
            risk_per_trade_pct=10.0,
            min_notional_usd=7.0,
            max_min_trade_risk_mult=1.5,
            clamp_up=True,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "min_trade_exceeds_risk_budget")

    def test_clamp_allows_min_trade_when_stop_risk_fits_budget_multiplier(self) -> None:
        result = clamp_order_size(
            pair="TESTUSD",
            price=100.0,
            desired_usd=4.0,
            rule=KrakenPairRule(
                wsname="TEST/USD",
                ordermin=0.05,
                lot_decimals=4,
                pair_decimals=2,
                costmin=7.0,
            ),
            equity_usd=30.0,
            max_position_usd=30.0,
            risk_per_trade_pct=10.0,
            min_notional_usd=7.0,
            max_min_trade_risk_mult=1.5,
            clamp_up=True,
            risk_fraction_of_notional=0.04,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "ok")


if __name__ == "__main__":
    unittest.main()
