import tempfile
import unittest
from unittest.mock import patch

from core.memory.trade_memory import build_outcome_record
from core.risk.exits import build_exit_plan
from core.execution.order_policy import build_exit_order_plan, build_order_plan
from apps.trader.positions import _deterministic_outcome_review, execute_replacement_exit
from core.risk.fee_filter import TradeCostAssessment
from core.risk.portfolio import PortfolioConfig, Position, PositionState, build_opportunity_snapshot, evaluate_trade
from core.state.portfolio import PortfolioState
from core.state import position_state_store
from core.state.position_state_store import load_position_state, merge_persisted_positions, save_position_state


class RuntimeImprovementTests(unittest.TestCase):
    def test_deterministic_outcome_review_classifies_good_winner(self) -> None:
        review = _deterministic_outcome_review(
            {
                "pnl_pct": 0.035,
                "exit_reason": "take_profit",
                "capture_vs_mfe_pct": 0.72,
                "structure_state": "intact",
            }
        )
        self.assertEqual(review["outcome_class"], "good_breakout")
        self.assertEqual(review["suggested_adjustment"], "keep_current_posture")

    def test_deterministic_outcome_review_classifies_broken_structure_loss(self) -> None:
        review = _deterministic_outcome_review(
            {
                "pnl_pct": -0.018,
                "exit_reason": "exit_posture:l3_structure_broken_neg",
                "capture_vs_mfe_pct": 0.1,
                "structure_state": "broken",
            }
        )
        self.assertEqual(review["outcome_class"], "weak_follow_through")
        self.assertEqual(review["suggested_adjustment"], "tighten_promotion")

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
                monitor_state="WEAKEN",
                monitor_reason="persisted_monitor",
                monitor_confidence=0.7,
                structure_state="fragile",
                max_price_seen=101.0,
                min_price_seen=94.0,
                mfe_pct=6.0,
                mae_pct=-1.0,
                mfe_r=1.2,
                mae_r=-0.2,
                etd_pct=2.5,
                etd_r=0.5,
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
        self.assertEqual(merged.monitor_state, "WEAKEN")
        self.assertEqual(merged.monitor_reason, "persisted_monitor")
        self.assertEqual(merged.monitor_confidence, 0.7)
        self.assertEqual(merged.structure_state, "fragile")
        self.assertEqual(merged.max_price_seen, 101.0)
        self.assertEqual(merged.min_price_seen, 94.0)
        self.assertEqual(merged.mfe_pct, 6.0)
        self.assertEqual(merged.mae_pct, -1.0)
        self.assertEqual(merged.mfe_r, 1.2)
        self.assertEqual(merged.mae_r, -0.2)
        self.assertEqual(merged.etd_pct, 2.5)
        self.assertEqual(merged.etd_r, 0.5)

    def test_merge_persisted_positions_keeps_richer_excursion_state_when_synced_is_sparse(self) -> None:
        synced = PositionState()
        synced.add_or_update(
            Position(
                symbol="ALGO/USD",
                side="LONG",
                weight=0.2,
                lane="L3",
                entry_price=0.095,
                max_price_seen=None,
                min_price_seen=None,
                mfe_pct=0.0,
                mae_pct=0.0,
                mfe_ts="",
                mae_ts="",
                mfe_r=0.0,
                mae_r=0.0,
                etd_pct=0.0,
                etd_r=0.0,
            )
        )

        persisted = PositionState()
        persisted.add_or_update(
            Position(
                symbol="ALGO/USD",
                side="LONG",
                weight=0.2,
                lane="L3",
                entry_price=0.095,
                monitor_state="RUN",
                monitor_reason="trend_intact_let_run",
                monitor_confidence=0.8,
                exit_posture="RUN",
                exit_posture_reason="trend_intact_let_run",
                exit_posture_confidence=0.8,
                structure_state="intact",
                max_price_seen=0.0997,
                min_price_seen=0.0944,
                mfe_pct=5.19,
                mae_pct=-0.46,
                mfe_ts="2026-04-01T01:56:00+00:00",
                mae_ts="2026-04-01T00:31:00+00:00",
                mfe_r=1.29,
                mae_r=-0.11,
                etd_pct=3.16,
                etd_r=0.79,
            )
        )

        merged = merge_persisted_positions(synced, persisted).get("ALGO/USD")
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.monitor_reason, "trend_intact_let_run")
        self.assertEqual(merged.exit_posture_reason, "trend_intact_let_run")
        self.assertEqual(merged.max_price_seen, 0.0997)
        self.assertEqual(merged.min_price_seen, 0.0944)
        self.assertEqual(merged.mfe_pct, 5.19)
        self.assertEqual(merged.mae_pct, -0.46)
        self.assertEqual(merged.mfe_ts, "2026-04-01T01:56:00+00:00")
        self.assertEqual(merged.mae_ts, "2026-04-01T00:31:00+00:00")
        self.assertEqual(merged.mfe_r, 1.29)
        self.assertEqual(merged.mae_r, -0.11)
        self.assertEqual(merged.etd_pct, 3.16)
        self.assertEqual(merged.etd_r, 0.79)

    def test_merge_persisted_positions_keeps_expected_edge_pct(self) -> None:
        synced = PositionState()
        synced.add_or_update(
            Position(symbol="BTC/USD", side="LONG", weight=0.2, entry_price=None, lane="L3", expected_edge_pct=1.7, risk_reward_ratio=2.4)
        )

        persisted = PositionState()
        persisted.add_or_update(
            Position(symbol="BTC/USD", side="LONG", weight=0.1, lane="L3", entry_price=95.0, expected_edge_pct=1.3, risk_reward_ratio=1.8)
        )

        merged = merge_persisted_positions(synced, persisted).get("BTC/USD")
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.expected_edge_pct, 1.7)
        self.assertEqual(merged.risk_reward_ratio, 2.4)

    def test_position_state_round_trip_preserves_expected_edge_pct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = position_state_store.Path(tmpdir) / "position_state.json"
            with patch.object(position_state_store, "POSITION_STATE_PATH", path):
                state = PositionState()
                state.add_or_update(
                    Position(symbol="ETH/USD", side="LONG", weight=0.15, lane="L2", entry_price=2500.0, expected_edge_pct=0.82, risk_reward_ratio=2.15)
                )
                save_position_state(state)

                loaded = load_position_state()
                position = loaded.get("ETH/USD")

        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.expected_edge_pct, 0.82)
        self.assertEqual(position.risk_reward_ratio, 2.15)

    def test_portfolio_clamps_proposed_weight_to_symbol_cap(self) -> None:
        positions = PositionState()
        decision = evaluate_trade(
            config=PortfolioConfig(
                max_weight_per_symbol=0.15,
                max_total_gross_exposure=0.95,
                max_open_positions=6,
            ),
            positions=positions,
            symbol="LTC/USD",
            side="LONG",
            proposed_weight=0.20,
            correlation_row=[],
            symbols=[],
            lane="L2",
            features={
                "entry_score": 100.0,
                "entry_recommendation": "BUY",
                "tp_after_cost_valid": True,
                "net_edge_pct": 1.0,
            },
        )
        self.assertEqual(decision["decision"], "scale_down")
        self.assertAlmostEqual(decision["size_factor"], 0.75)
        self.assertIn("proposed_weight_exceeds_per_symbol_cap", decision["reasons"])

    def test_expected_edge_round_trip_survives_entry_to_outcome_record(self) -> None:
        position = build_exit_plan(
            symbol="SOL/USD",
            side="LONG",
            weight=0.1,
            entry_price=150.0,
            atr=3.0,
            entry_bar_ts="2026-03-27T00:00:00Z",
            entry_bar_idx=123,
            entry_reasons=["breakout"],
            lane="L3",
            expected_edge_pct=1.25,
        )

        outcome = build_outcome_record(
            symbol=position.symbol,
            side=position.side,
            pnl_pct=0.018,
            pnl_usd=12.5,
            hold_minutes=45.0,
            exit_reason="take_profit",
            entry_reasons=list(position.entry_reasons),
            regime_label="trending",
            expected_edge_pct=position.expected_edge_pct,
            realized_edge_pct=0.014,
            edge_capture_ratio=0.0112,
        )

        self.assertEqual(position.expected_edge_pct, 1.25)
        self.assertEqual(outcome.expected_edge_pct, 1.25)
        self.assertEqual(outcome.realized_edge_pct, 0.014)
        self.assertEqual(outcome.edge_capture_ratio, 0.0112)

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

    def test_evaluate_trade_prefers_replacing_weak_hold_when_book_is_full(self) -> None:
        positions = PositionState()
        positions.add_or_update(
            Position(
                symbol="OP/USD",
                side="LONG",
                weight=0.2,
                lane="L3",
                monitor_state="STALL",
                exit_posture="STALE",
                structure_state="broken",
            )
        )
        positions.add_or_update(Position(symbol="DOT/USD", side="LONG", weight=0.2, lane="L2"))

        decision = evaluate_trade(
            config=PortfolioConfig(max_open_positions=2, max_total_gross_exposure=0.6, max_positions_per_sector=3),
            positions=positions,
            symbol="ALGO/USD",
            side="LONG",
            proposed_weight=0.2,
            correlation_row=[0.1, 0.1, 1.0],
            symbols=["OP/USD", "DOT/USD", "ALGO/USD"],
            lane="L3",
            features={
                "entry_score": 82.0,
                "entry_recommendation": "BUY",
                "trend_confirmed": True,
                "tp_after_cost_valid": True,
                "point_breakdown": {"net_edge_pct": 0.8},
            },
        )

        self.assertEqual(decision["decision"], "replace")
        self.assertEqual(decision["replace_symbol"], "OP/USD")

    def test_evaluate_trade_does_not_replace_when_candidate_is_not_strong_enough(self) -> None:
        positions = PositionState()
        positions.add_or_update(
            Position(
                symbol="OP/USD",
                side="LONG",
                weight=0.2,
                lane="L3",
                monitor_state="STALL",
                exit_posture="STALE",
                structure_state="broken",
            )
        )
        positions.add_or_update(Position(symbol="DOT/USD", side="LONG", weight=0.2, lane="L2"))

        decision = evaluate_trade(
            config=PortfolioConfig(max_open_positions=2, max_total_gross_exposure=0.6, max_positions_per_sector=3),
            positions=positions,
            symbol="ALGO/USD",
            side="LONG",
            proposed_weight=0.2,
            correlation_row=[0.1, 0.1, 1.0],
            symbols=["OP/USD", "DOT/USD", "ALGO/USD"],
            lane="L3",
            features={
                "entry_score": 61.0,
                "entry_recommendation": "WATCH",
                "trend_confirmed": False,
                "tp_after_cost_valid": True,
                "point_breakdown": {"net_edge_pct": 0.8},
            },
        )

        self.assertEqual(decision["decision"], "block")
        self.assertIn("max_open_positions_reached", decision["reasons"])

    def test_evaluate_trade_can_replace_small_default_hold_starter(self) -> None:
        positions = PositionState()
        positions.add_or_update(
            Position(
                symbol="ALGO/USD",
                side="LONG",
                weight=0.04,
                lane="L3",
                monitor_state="RUN",
                monitor_reason="default_hold_state",
                exit_posture="RUN",
                exit_posture_reason="default_hold_state",
                structure_state="intact",
            )
        )
        positions.add_or_update(
            Position(
                symbol="RENDER/USD",
                side="LONG",
                weight=0.15,
                lane="L3",
                monitor_state="RUN",
                monitor_reason="trend_intact_let_run",
                exit_posture="RUN",
                exit_posture_reason="trend_intact_let_run",
                structure_state="intact",
            )
        )

        decision = evaluate_trade(
            config=PortfolioConfig(max_open_positions=2, max_total_gross_exposure=0.6, max_positions_per_sector=3),
            positions=positions,
            symbol="FET/USD",
            side="LONG",
            proposed_weight=0.15,
            correlation_row=[0.1, 0.1, 1.0],
            symbols=["ALGO/USD", "RENDER/USD", "FET/USD"],
            lane="L2",
            features={
                "entry_score": 77.0,
                "entry_recommendation": "BUY",
                "trend_confirmed": True,
                "tp_after_cost_valid": True,
                "point_breakdown": {"net_edge_pct": 0.8},
            },
        )

        self.assertEqual(decision["decision"], "replace")
        self.assertEqual(decision["replace_symbol"], "ALGO/USD")

    def test_execute_replacement_exit_removes_position_when_exit_fills(self) -> None:
        positions = PositionState()
        positions.add_or_update(Position(symbol="OP/USD", side="LONG", weight=0.1, lane="L3", entry_price=1.0))
        portfolio = PortfolioState(cash=50.0, positions={"OP/USD": 10.0}, position_marks={"OP/USD": 1.2})
        last_exit_ts: dict[str, float] = {}

        class _Executor:
            def execute_exit(self, **kwargs):
                return {
                    "status": "filled",
                    "symbol": kwargs["symbol"],
                    "side": "SELL",
                    "qty": kwargs["qty"],
                    "price": kwargs["price"],
                    "notional": kwargs["qty"] * kwargs["price"],
                    "fee": 0.0,
                    "mark_price": kwargs["price"],
                    "exit_reason": kwargs["exit_reason"],
                }

        exec_result, state_change = execute_replacement_exit(
            target_symbol="OP/USD",
            replacement_symbol="ALGO/USD",
            executor=_Executor(),
            portfolio=portfolio,
            positions_state=positions,
            last_exit_ts=last_exit_ts,
        )

        self.assertEqual(exec_result["status"], "filled")
        self.assertIsNone(positions.get("OP/USD"))
        self.assertNotIn("OP/USD", portfolio.positions)
        self.assertIn("OP/USD", last_exit_ts)
        self.assertIn("cash", state_change)

    def test_build_opportunity_snapshot_identifies_weakest_replaceable_hold(self) -> None:
        positions = PositionState()
        positions.add_or_update(
            Position(
                symbol="OP/USD",
                side="LONG",
                weight=0.1,
                lane="L3",
                monitor_state="STALL",
                exit_posture="STALE",
                structure_state="broken",
            )
        )
        positions.add_or_update(
            Position(
                symbol="DOT/USD",
                side="LONG",
                weight=0.1,
                lane="L2",
                monitor_state="RUN",
                exit_posture="RUN",
                structure_state="intact",
            )
        )

        snapshot = build_opportunity_snapshot(
            positions=positions,
            candidate_symbol="ALGO/USD",
            features={
                "entry_score": 81.0,
                "entry_recommendation": "BUY",
                "trend_confirmed": True,
                "tp_after_cost_valid": True,
                "point_breakdown": {"net_edge_pct": 0.75},
            },
        )

        self.assertEqual(snapshot["weakest_held_symbol"], "OP/USD")
        self.assertEqual(snapshot["replace_ready_symbol"], "OP/USD")
        self.assertTrue(snapshot["replace_ready"])

    def test_order_plan_prefers_limit_when_quote_is_available(self) -> None:
        plan = build_order_plan(
            {
                "lane": "L3",
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "book_valid": True,
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
        self.assertIn("lane3_balanced_maker", plan.reasons)

    def test_lane1_order_plan_uses_patient_maker_limit(self) -> None:
        plan = build_order_plan(
            {
                "lane": "L1",
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "book_valid": True,
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
        assert plan.limit_price is not None
        self.assertGreater(plan.limit_price, 99.9)
        self.assertLess(plan.limit_price, 100.1)
        self.assertIn("lane1_patient_maker", plan.reasons)

    def test_lane4_promote_prefers_maker_first_limit(self) -> None:
        plan = build_order_plan(
            {
                "lane": "L4",
                "promotion_tier": "promote",
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "spread_pct": 0.8,
                "book_valid": True,
            },
            "LONG",
            TradeCostAssessment(
                actionable=True,
                expected_edge_pct=1.0,
                spread_pct=0.8,
                fee_round_trip_pct=0.4,
                slippage_pct=0.1,
                total_cost_pct=1.3,
                reasons=[],
            ),
        )
        self.assertEqual(plan.order_type, "limit")
        self.assertTrue(plan.maker_preference)
        self.assertIn("lane4_maker_first", plan.reasons)

    def test_invalid_book_falls_back_to_market(self) -> None:
        plan = build_order_plan(
            {
                "lane": "L2",
                "promotion_tier": "probe",
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "book_valid": False,
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
        self.assertEqual(plan.order_type, "market")
        self.assertIn("book_invalid", plan.reasons)
        self.assertIn("no_bid_ask_fallback_market", plan.reasons)

    def test_exit_plan_uses_crossing_limit_for_stop_loss_when_book_is_valid(self) -> None:
        plan = build_exit_order_plan(
            {
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "book_valid": True,
            },
            side="LONG",
            exit_reason="stop_loss",
        )
        self.assertEqual(plan.order_type, "limit")
        self.assertFalse(plan.maker_preference)
        self.assertIsNotNone(plan.limit_price)
        assert plan.limit_price is not None
        self.assertLess(plan.limit_price, 99.9)
        self.assertIn("crossing_limit_exit", plan.reasons)
        self.assertIn("protective_exit", plan.reasons)
        self.assertEqual(plan.stale_ttl_sec, 20)

    def test_exit_plan_keeps_market_only_for_hard_fail(self) -> None:
        plan = build_exit_order_plan(
            {
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "book_valid": True,
            },
            side="LONG",
            exit_reason="hard_fail:broken_structure",
        )
        self.assertEqual(plan.order_type, "market")
        self.assertIn("emergency_exit", plan.reasons)

    def test_exit_plan_uses_longer_ttl_for_take_profit(self) -> None:
        plan = build_exit_order_plan(
            {
                "price": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "book_valid": True,
            },
            side="LONG",
            exit_reason="take_profit",
        )
        self.assertEqual(plan.order_type, "limit")
        self.assertIn("take_profit_exit", plan.reasons)
        self.assertEqual(plan.stale_ttl_sec, 45)

    def test_cpp_executor_does_not_initialize_live_client_in_paper_mode(self) -> None:
        with patch.dict("os.environ", {"EXECUTION_MODE": "paper"}, clear=False):
            with patch("core.execution.cpp_exec.KrakenLiveExecutor") as live_executor:
                from core.execution.cpp_exec import CppExecutor

                executor = CppExecutor()

        live_executor.assert_not_called()
        self.assertIsNone(executor._live)


if __name__ == "__main__":
    unittest.main()
