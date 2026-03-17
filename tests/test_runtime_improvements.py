import unittest

from core.execution.order_policy import build_order_plan
from core.risk.fee_filter import TradeCostAssessment
from core.risk.portfolio import PortfolioConfig, Position, PositionState, evaluate_trade
from core.state.position_state_store import merge_persisted_positions


class RuntimeImprovementTests(unittest.TestCase):
    def test_merge_persisted_positions_keeps_existing_exit_plan(self) -> None:
        synced = PositionState()
        synced.add_or_update(Position(symbol="BTC/USD", side="LONG", weight=0.2, entry_price=None, lane="L3"))

        persisted = PositionState()
        persisted.add_or_update(
            Position(
                symbol="BTC/USD",
                side="LONG",
                weight=0.1,
                lane="L3",
                entry_price=95.0,
                stop_loss=90.0,
                take_profit=110.0,
                risk_r=5.0,
                trailing_armed=True,
                trail_stop=99.0,
            )
        )

        merged = merge_persisted_positions(synced, persisted).get("BTC/USD")
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.entry_price, 95.0)
        self.assertEqual(merged.stop_loss, 90.0)
        self.assertEqual(merged.take_profit, 110.0)
        self.assertTrue(merged.trailing_armed)
        self.assertEqual(merged.trail_stop, 99.0)

    def test_evaluate_trade_blocks_when_sector_cap_reached(self) -> None:
        positions = PositionState()
        positions.add_or_update(Position(symbol="AAA/USD", side="LONG", weight=0.1, lane="L3"))
        positions.add_or_update(Position(symbol="BBB/USD", side="LONG", weight=0.1, lane="L3"))

        decision = evaluate_trade(
            config=PortfolioConfig(max_positions_per_sector=2),
            positions=positions,
            symbol="CCC/USD",
            side="LONG",
            proposed_weight=0.1,
            correlation_row=[0.1, 0.1, 1.0],
            symbols=["AAA/USD", "BBB/USD", "CCC/USD"],
            lane="L3",
        )
        self.assertEqual(decision["decision"], "block")
        self.assertTrue(any("sector_limit_reached" in reason for reason in decision["reasons"]))

    def test_evaluate_trade_scales_down_on_average_correlation(self) -> None:
        positions = PositionState()
        positions.add_or_update(Position(symbol="AAA/USD", side="LONG", weight=0.1, lane="L2"))
        positions.add_or_update(Position(symbol="BBB/USD", side="LONG", weight=0.1, lane="L2"))

        decision = evaluate_trade(
            config=PortfolioConfig(
                max_positions_per_sector=5,
                avg_corr_scale_threshold=0.7,
                avg_corr_scale_down=0.6,
                corr_threshold=0.95,
            ),
            positions=positions,
            symbol="CCC/USD",
            side="LONG",
            proposed_weight=0.1,
            correlation_row=[0.74, 0.72, 1.0],
            symbols=["AAA/USD", "BBB/USD", "CCC/USD"],
            lane="L2",
        )
        self.assertEqual(decision["decision"], "scale_down")
        self.assertEqual(decision["size_factor"], 0.6)
        self.assertTrue(any("avg_corr_scale_down" in reason for reason in decision["reasons"]))

    def test_order_plan_prefers_limit_when_quote_is_available(self) -> None:
        plan = build_order_plan(
            {
                "lane": "L3",
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
            },
            "LONG",
            TradeCostAssessment(
                actionable=True,
                expected_edge_pct=1.0,
                spread_pct=0.2,
                fee_round_trip_pct=0.4,
                slippage_pct=0.1,
                total_cost_pct=0.7,
                reasons=[],
            ),
        )
        self.assertEqual(plan.order_type, "limit")
        self.assertTrue(plan.maker_preference)
        self.assertIsNotNone(plan.limit_price)


if __name__ == "__main__":
    unittest.main()
