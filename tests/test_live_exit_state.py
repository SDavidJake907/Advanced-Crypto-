from __future__ import annotations

import unittest

from unittest.mock import patch

from core.llm.phi3_exit_posture import ExitPostureDecision
from core.risk.exits import build_exit_plan, maybe_apply_exit_posture, maybe_update_trailing, review_live_exit_state
from core.risk.portfolio import Position


class LiveExitStateTests(unittest.TestCase):
    def test_live_exit_state_runs_when_move_is_healthy(self) -> None:
        decision = review_live_exit_state(
            Position(symbol="TRX/USD", side="LONG", weight=0.1, lane="L3", entry_price=1.0),
            price=1.01,
            hold_minutes=8.0,
            features={
                "momentum": 0.003,
                "momentum_5": 0.004,
                "momentum_14": 0.006,
                "spread_pct": 0.25,
                "volume_ratio": 1.2,
            },
            universe_context={
                "current_symbol_is_top_candidate": True,
                "current_symbol_is_top_lane_candidate": True,
            },
        )
        self.assertEqual(decision.posture, "RUN")

    def test_live_exit_state_tightens_on_stalling_rank_decay_winner(self) -> None:
        decision = review_live_exit_state(
            Position(symbol="FET/USD", side="LONG", weight=0.1, lane="L3", entry_price=1.0),
            price=1.03,
            hold_minutes=50.0,
            features={
                "momentum": 0.001,
                "momentum_5": -0.004,
                "momentum_14": -0.001,
                "spread_pct": 0.9,
                "volume_ratio": 0.95,
                "structure_state": "broken",
                "entry_score": 60.0,
            },
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(decision.posture, "TIGHTEN")
        self.assertIn("live_tighten", decision.reason)

    def test_live_exit_state_exits_on_deteriorating_winner_with_rank_loss(self) -> None:
        decision = review_live_exit_state(
            Position(symbol="ASTER/USD", side="LONG", weight=0.03, lane="L4", entry_price=1.0),
            price=1.01,
            hold_minutes=12.0,
            features={
                "momentum": -0.003,
                "momentum_5": -0.003,
                "momentum_14": -0.001,
                "spread_pct": 1.9,
                "volume_ratio": 0.9,
                "structure_state": "broken",
                "entry_score": 58.0,
            },
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
                "top_scored": [{"candidate_score": 72.0, "leader_urgency": 7.0, "rank_delta": 5, "lane_rank_delta": 3}],
            },
        )
        self.assertEqual(decision.posture, "EXIT")
        self.assertIn("rotate_exit", decision.reason)

    def test_structured_runner_does_not_rotate_early_on_rank_loss(self) -> None:
        decision = review_live_exit_state(
            Position(
                symbol="ADA/USD",
                side="LONG",
                weight=0.1,
                lane="L3",
                entry_price=1.0,
                expected_hold_style="structured_runner",
            ),
            price=1.02,
            hold_minutes=90.0,
            features={
                "momentum": 0.0008,
                "momentum_5": -0.001,
                "momentum_14": 0.001,
                "spread_pct": 0.6,
                "volume_ratio": 0.95,
                "structure_quality": 82.0,
                "continuation_quality": 78.0,
            },
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(decision.posture, "RUN")

    def test_spread_and_rank_do_not_exit_when_structure_is_intact(self) -> None:
        decision = review_live_exit_state(
            Position(symbol="LINK/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=1.04,
            hold_minutes=70.0,
            features={
                "momentum": 0.0005,
                "momentum_5": -0.001,
                "momentum_14": 0.002,
                "spread_pct": 1.3,
                "volume_ratio": 0.95,
                "trend_confirmed": True,
                "ema9_above_ema26": True,
                "ema_cross_distance_pct": 0.003,
                "structure_quality": 78.0,
                "continuation_quality": 75.0,
            },
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(decision.posture, "RUN")

    def test_structured_runner_trail_arms_later_than_standard_l2(self) -> None:
        standard = maybe_update_trailing(
            Position(symbol="LINK/USD", side="LONG", weight=0.1, lane="L2", entry_price=100.0, risk_r=2.0),
            price=105.2,
            atr=1.0,
        )
        structured = maybe_update_trailing(
            Position(
                symbol="LINK/USD",
                side="LONG",
                weight=0.1,
                lane="L2",
                entry_price=100.0,
                risk_r=2.0,
                expected_hold_style="structured_runner",
            ),
            price=105.2,
            atr=1.0,
        )
        self.assertTrue(standard.trailing_armed)
        self.assertFalse(structured.trailing_armed)

    def test_structured_runner_tighten_stop_is_looser(self) -> None:
        standard = maybe_apply_exit_posture(
            Position(symbol="LINK/USD", side="LONG", weight=0.1, lane="L2", entry_price=100.0),
            price=110.0,
            atr=2.0,
            posture=ExitPostureDecision("TIGHTEN", "protect_open_profit", 0.74),
        )
        structured = maybe_apply_exit_posture(
            Position(
                symbol="LINK/USD",
                side="LONG",
                weight=0.1,
                lane="L2",
                entry_price=100.0,
                expected_hold_style="structured_runner",
            ),
            price=110.0,
            atr=2.0,
            posture=ExitPostureDecision("TIGHTEN", "protect_open_profit", 0.74),
        )
        self.assertIsNotNone(standard.trail_stop)
        self.assertIsNotNone(structured.trail_stop)
        assert standard.trail_stop is not None and structured.trail_stop is not None
        self.assertLess(structured.trail_stop, standard.trail_stop)

    def test_live_exit_state_exits_failed_follow_through_after_open_profit_fades(self) -> None:
        position = Position(
            symbol="FET/USD",
            side="LONG",
            weight=0.1,
            lane="L3",
            entry_price=1.0,
            mfe_pct=3.2,
            etd_pct=2.7,
        )
        decision = review_live_exit_state(
            position,
            price=1.005,
            hold_minutes=45.0,
            features={
                "momentum": -0.001,
                "momentum_5": -0.0025,
                "momentum_14": -0.001,
                "spread_pct": 0.4,
                "volume_ratio": 0.95,
                "structure_state": "broken",
                "entry_score": 58.0,
            },
            universe_context={},
        )
        self.assertEqual(decision.posture, "EXIT")
        self.assertIn("failed_follow_through", decision.reason)

    @patch("core.config.runtime.load_runtime_overrides", return_value={})
    def test_build_exit_plan_uses_realistic_primary_tp_for_standard_hold(self, mock_override) -> None:
        position = build_exit_plan(
            symbol="LINK/USD",
            side="LONG",
            weight=0.1,
            entry_price=100.0,
            atr=1.0,
            entry_bar_ts=None,
            entry_bar_idx=None,
            lane="L3",
            expected_hold_style="standard",
        )
        self.assertIsNotNone(position.take_profit)
        assert position.take_profit is not None
        self.assertLessEqual(position.take_profit, 105.0)
        self.assertGreaterEqual(position.take_profit, 101.0)
        self.assertGreater(position.risk_reward_ratio, 0.0)

    def test_build_exit_plan_runner_has_no_fixed_take_profit(self) -> None:
        position = build_exit_plan(
            symbol="ADA/USD",
            side="LONG",
            weight=0.1,
            entry_price=100.0,
            atr=1.0,
            entry_bar_ts=None,
            entry_bar_idx=None,
            lane="L2",
            expected_hold_style="rotation_runner",
        )
        self.assertIsNone(position.take_profit)
        self.assertGreater(position.risk_reward_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
