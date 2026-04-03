from __future__ import annotations

import unittest

from core.llm.phi3_exit_posture import ExitPostureDecision
from core.risk.portfolio import Position
from core.risk.position_monitor import monitor_open_position


class PositionMonitorTests(unittest.TestCase):
    def test_l1_monitor_gives_early_breakout_room(self) -> None:
        result = monitor_open_position(
            Position(symbol="BTC/USD", side="LONG", weight=0.1, lane="L1", entry_price=1.0),
            price=1.006,
            atr=0.01,
            hold_minutes=8.0,
            features={
                "momentum": 0.003,
                "momentum_5": 0.003,
                "momentum_14": 0.005,
                "trend_1h": 1,
                "volume_ratio": 1.1,
                "rsi": 58.0,
                "spread_pct": 0.2,
            },
            phi3_posture=ExitPostureDecision("RUN", "trend_intact_let_run", 0.72),
            universe_context={
                "current_symbol_is_top_candidate": True,
                "current_symbol_is_top_lane_candidate": True,
            },
        )
        self.assertEqual(result.final_state.state, "RUN")
        self.assertEqual(result.position.exit_posture, "RUN")
        self.assertEqual(result.position.monitor_state, "RUN")

    def test_l2_monitor_exits_failed_bounce_faster(self) -> None:
        result = monitor_open_position(
            Position(symbol="APT/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=0.988,
            atr=0.01,
            hold_minutes=18.0,
            features={
                "momentum": -0.002,
                "momentum_5": -0.002,
                "momentum_14": -0.001,
                "trend_1h": 0,
                "volume_ratio": 0.9,
                "rsi": 48.0,
                "spread_pct": 0.4,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(result.final_state.state, "FAIL")
        self.assertEqual(result.position.exit_posture, "EXIT")
        self.assertEqual(result.position.monitor_state, "FAIL")
        self.assertIn("l2_bounce_failed", result.position.exit_posture_reason)

    def test_monitor_open_position_applies_live_exit_overlay(self) -> None:
        result = monitor_open_position(
            Position(symbol="ASTER/USD", side="LONG", weight=0.03, lane="L4", entry_price=1.0),
            price=1.01,
            atr=0.02,
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
            phi3_posture=ExitPostureDecision("RUN", "trend_intact_let_run", 0.72),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
                "top_scored": [{"candidate_score": 72.0, "leader_urgency": 7.0, "rank_delta": 5, "lane_rank_delta": 3}],
            },
        )
        self.assertEqual(result.live_state.state, "ROTATE")
        self.assertEqual(result.final_state.state, "ROTATE")
        self.assertEqual(result.live_posture.posture, "EXIT")
        self.assertEqual(result.position.exit_posture, "EXIT")
        self.assertEqual(result.position.monitor_state, "ROTATE")

    def test_l3_weaken_is_explicit_monitor_state(self) -> None:
        result = monitor_open_position(
            Position(symbol="LINK/USD", side="LONG", weight=0.1, lane="L3", entry_price=1.0),
            price=1.02,
            atr=0.01,
            hold_minutes=40.0,
            features={
                "momentum": -0.0015,
                "momentum_5": -0.001,
                "momentum_14": 0.002,
                "spread_pct": 0.2,
                "volume_ratio": 1.0,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.55),
            universe_context={
                "current_symbol_is_top_candidate": True,
                "current_symbol_is_top_lane_candidate": True,
            },
        )
        self.assertEqual(result.lane_state.state, "WEAKEN")
        self.assertEqual(result.final_state.state, "WEAKEN")
        self.assertEqual(result.position.monitor_state, "WEAKEN")
        self.assertEqual(result.position.exit_posture, "TIGHTEN")

    def test_l3_structured_runner_gets_more_room_before_weaken(self) -> None:
        result = monitor_open_position(
            Position(
                symbol="ADA/USD",
                side="LONG",
                weight=0.1,
                lane="L3",
                entry_price=1.0,
                expected_hold_style="structured_runner",
            ),
            price=1.012,
            atr=0.01,
            hold_minutes=120.0,
            features={
                "momentum": -0.0015,
                "momentum_5": -0.001,
                "momentum_14": 0.002,
                "spread_pct": 0.2,
                "volume_ratio": 1.0,
                "structure_quality": 84.0,
                "continuation_quality": 79.0,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.55),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(result.final_state.state, "RUN")
        self.assertEqual(result.position.monitor_state, "RUN")

    def test_lane_specific_hold_keeps_l2_room_before_fail(self) -> None:
        result = monitor_open_position(
            Position(symbol="RENDER/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=0.992,
            atr=0.01,
            hold_minutes=30.0,
            features={
                "momentum": -0.002,
                "momentum_5": -0.002,
                "momentum_14": -0.001,
                "trend_1h": 0,
                "volume_ratio": 0.9,
                "rsi": 48.0,
                "spread_pct": 0.4,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(result.final_state.state, "RUN")

    def test_l2_structure_intact_suppresses_stall_and_rotate(self) -> None:
        result = monitor_open_position(
            Position(symbol="ADA/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=0.999,
            atr=0.01,
            hold_minutes=75.0,
            features={
                "momentum": 0.0004,
                "momentum_5": -0.0004,
                "momentum_14": 0.001,
                "trend_1h": 1,
                "volume_ratio": 0.95,
                "rsi": 55.0,
                "spread_pct": 0.3,
                "trend_confirmed": True,
                "ema9_above_ema26": True,
                "ema_cross_distance_pct": 0.003,
                "structure_quality": 79.0,
                "continuation_quality": 74.0,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
                "top_scored": [{"candidate_score": 88.0, "leader_urgency": 3.5}],
            },
        )
        self.assertEqual(result.lane_state.state, "RUN")
        self.assertEqual(result.final_state.state, "RUN")

    def test_l3_structure_intact_suppresses_soft_weaken(self) -> None:
        result = monitor_open_position(
            Position(symbol="LINK/USD", side="LONG", weight=0.1, lane="L3", entry_price=1.0),
            price=1.02,
            atr=0.01,
            hold_minutes=55.0,
            features={
                "momentum": -0.001,
                "momentum_5": -0.001,
                "momentum_14": 0.001,
                "spread_pct": 0.2,
                "volume_ratio": 1.0,
                "trend_confirmed": True,
                "ema9_above_ema26": True,
                "ema_cross_distance_pct": 0.0025,
                "structure_quality": 76.0,
                "continuation_quality": 73.0,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.55),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(result.lane_state.state, "RUN")
        self.assertEqual(result.final_state.state, "RUN")

    def test_monitor_tracks_mfe_mae_and_structure_state(self) -> None:
        result = monitor_open_position(
            Position(symbol="SOL/USD", side="LONG", weight=0.1, lane="L2", entry_price=100.0, risk_r=1.0),
            price=104.0,
            atr=1.0,
            hold_minutes=20.0,
            features={
                "momentum": 0.003,
                "momentum_5": 0.003,
                "momentum_14": 0.004,
                "trend_confirmed": True,
                "ema9_above_ema26": True,
                "ema_cross_distance_pct": 0.003,
                "structure_quality": 80.0,
                "continuation_quality": 76.0,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.55),
            universe_context={"current_symbol_is_top_candidate": True},
        )
        self.assertEqual(result.position.structure_state, "intact")
        self.assertGreater(result.position.mfe_pct, 3.9)
        self.assertEqual(result.position.mae_pct, 0.0)
        self.assertGreater(result.position.mfe_r, 3.9)
        self.assertEqual(result.position.etd_r, 0.0)

    def test_rotation_requires_persistent_replacement(self) -> None:
        result = monitor_open_position(
            Position(symbol="ADA/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=0.995,
            atr=0.01,
            hold_minutes=80.0,
            features={
                "momentum": -0.0005,
                "momentum_5": -0.0005,
                "momentum_14": 0.0002,
                "trend_1h": 0,
                "volume_ratio": 1.0,
                "rsi": 50.0,
                "spread_pct": 0.3,
                "entry_score": 55.0,
                "structure_state": "broken",
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
                "top_scored": [{"candidate_score": 70.0, "leader_urgency": 3.0, "rank_delta": 1, "lane_rank_delta": 1}],
            },
        )
        self.assertNotEqual(result.final_state.state, "ROTATE")

    def test_l2_weak_hold_rotates_into_materially_stronger_leader(self) -> None:
        result = monitor_open_position(
            Position(symbol="NEAR/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=1.004,
            atr=0.01,
            hold_minutes=47.0,
            features={
                "momentum": -0.0004,
                "momentum_5": -0.0004,
                "momentum_14": 0.0005,
                "trend_1h": 1,
                "volume_ratio": 1.0,
                "rsi": 54.0,
                "spread_pct": 0.2,
                "entry_score": 58.0,
                "structure_quality": 58.0,
                "continuation_quality": 59.0,
                "structure_state": "fragile",
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
                "top_scored": [
                    {
                        "candidate_score": 84.0,
                        "leader_urgency": 3.5,
                        "rank_delta": 5,
                        "lane_rank_delta": 3,
                    }
                ],
            },
        )
        self.assertEqual(result.lane_state.state, "ROTATE")
        self.assertEqual(result.final_state.state, "ROTATE")
        self.assertEqual(result.position.monitor_state, "ROTATE")
        self.assertEqual(result.position.exit_posture, "EXIT")
        self.assertIn("rotation_to_stronger_candidate", result.position.exit_posture_reason)

    def test_l2_weak_hold_does_not_rotate_before_weak_hold_threshold(self) -> None:
        result = monitor_open_position(
            Position(symbol="FIL/USD", side="LONG", weight=0.1, lane="L2", entry_price=1.0),
            price=1.004,
            atr=0.01,
            hold_minutes=32.0,
            features={
                "momentum": -0.0004,
                "momentum_5": -0.0004,
                "momentum_14": 0.0005,
                "trend_1h": 1,
                "volume_ratio": 1.0,
                "rsi": 54.0,
                "spread_pct": 0.2,
                "entry_score": 58.0,
                "structure_quality": 58.0,
                "continuation_quality": 59.0,
                "structure_state": "fragile",
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
                "top_scored": [
                    {
                        "candidate_score": 84.0,
                        "leader_urgency": 3.5,
                        "rank_delta": 5,
                        "lane_rank_delta": 3,
                    }
                ],
            },
        )
        self.assertNotEqual(result.lane_state.state, "ROTATE")
        self.assertNotEqual(result.final_state.state, "ROTATE")

    def test_l3_broken_stale_loser_no_longer_sits_in_run(self) -> None:
        result = monitor_open_position(
            Position(symbol="XDC/USD", side="LONG", weight=0.1, lane="L3", entry_price=1.0),
            price=0.972,
            atr=0.01,
            hold_minutes=844.0,
            features={
                "momentum": -0.0008,
                "momentum_5": -0.0008,
                "momentum_14": -0.0002,
                "spread_pct": 0.2,
                "volume_ratio": 0.95,
                "structure_break_risk": True,
            },
            phi3_posture=ExitPostureDecision("RUN", "default_hold_state", 0.58),
            universe_context={
                "current_symbol_is_top_candidate": False,
                "current_symbol_is_top_lane_candidate": False,
            },
        )
        self.assertEqual(result.final_state.state, "FAIL")
        self.assertEqual(result.position.monitor_state, "FAIL")
        self.assertEqual(result.position.exit_posture, "EXIT")
        self.assertIn("l3_structure_broken_stale", result.position.exit_posture_reason)


if __name__ == "__main__":
    unittest.main()
