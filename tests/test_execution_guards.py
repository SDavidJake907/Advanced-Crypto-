import unittest

from core.execution.clamp import clamp_order_size
from core.execution.kraken_rules import KrakenPairRule


class ExecutionGuardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
