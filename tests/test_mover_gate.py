import unittest
from unittest.mock import patch

from core.policy.nemotron_gate import passes_deterministic_candidate_gate, symbol_in_top_candidates
from core.risk.portfolio import PositionState


class MoverGateTests(unittest.TestCase):
    def test_ranging_market_allows_strong_mover_buy(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "L2_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.0,
                "STABILIZATION_STRICT_ENTRY_ENABLED": False,
            },
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="XLM/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "XLM/USD",
                    "lane": "L2",
                    "indicators_ready": True,
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 59.97,
                    "rotation_score": 0.12,
                    "momentum_5": 0.01,
                    "volume_ratio": 1.15,
                    "volume_surge": 0.36,
                    "xgb_score": 66.5,
                    "trend_confirmed": False,
                    "short_tf_ready_5m": True,
                    "short_tf_ready_15m": True,
                    "ranging_market": True,
                    "structure_quality": 66.0,
                    "continuation_quality": 68.0,
                    "trade_quality": 62.0,
                    "tp_after_cost_valid": True,
                    "net_edge_pct": 0.35,
                    "range_breakout_1h": True,
                    "sentiment_symbol_trending": False,
                    "macro_30d": "sideways",
                    "regime_7d": "choppy",
                },
            )
        self.assertTrue(passed)
        self.assertEqual(reason, "passed")

    def test_ranging_market_no_longer_auto_blocks_buy_candidate(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "L2_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.0,
                "STABILIZATION_STRICT_ENTRY_ENABLED": False,
            },
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="ICP/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "ICP/USD",
                    "lane": "L2",
                    "indicators_ready": True,
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 55.0,
                    "rotation_score": 0.0,
                    "momentum_5": 0.0,
                    "volume_ratio": 1.0,
                    "volume_surge": 0.0,
                    "xgb_score": 45.9,
                    "trend_confirmed": False,
                    "short_tf_ready_5m": True,
                    "short_tf_ready_15m": True,
                    "ranging_market": True,
                    "sentiment_symbol_trending": False,
                    "macro_30d": "sideways",
                    "regime_7d": "choppy",
                },
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "passed")

    def test_ranging_unconfirmed_no_longer_adds_second_deterministic_block(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "L2_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.0,
                "STABILIZATION_STRICT_ENTRY_ENABLED": False,
            },
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="CRV/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "CRV/USD",
                    "lane": "L2",
                    "indicators_ready": True,
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 91.63,
                    "rotation_score": 0.44,
                    "momentum_5": 0.004,
                    "volume_ratio": 1.1,
                    "volume_surge": 0.18,
                    "trend_confirmed": False,
                    "short_tf_ready_5m": True,
                    "short_tf_ready_15m": True,
                    "ranging_market": True,
                    "structure_quality": 62.14,
                    "continuation_quality": 48.68,
                    "trade_quality": 99.87,
                    "tp_after_cost_valid": True,
                    "net_edge_pct": 0.25,
                    "range_breakout_1h": False,
                    "pullback_hold": False,
                },
            )
        self.assertTrue(passed)
        self.assertEqual(reason, "passed")

    def test_ranging_unconfirmed_allows_valid_l3_retest_mover(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "L3_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.0,
                "STABILIZATION_STRICT_ENTRY_ENABLED": True,
                "STABILIZATION_ALLOWED_LANES": "L2,L3",
                "STABILIZATION_MIN_ENTRY_SCORE": 52.0,
                "STABILIZATION_MIN_NET_EDGE_PCT": 0.0,
                "STABILIZATION_REQUIRE_TP_AFTER_COST_VALID": True,
                "STABILIZATION_REQUIRE_TREND_CONFIRMED": False,
                "STABILIZATION_REQUIRE_SHORT_TF_READY_15M": True,
                "STABILIZATION_BLOCK_RANGING_MARKET": False,
                "STABILIZATION_REQUIRE_BUY_RECOMMENDATION": False,
            },
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="TIA/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "TIA/USD",
                    "lane": "L3",
                    "indicators_ready": True,
                    "entry_recommendation": "BUY",
                    "reversal_risk": "LOW",
                    "entry_score": 62.84,
                    "rotation_score": -0.22,
                    "momentum_5": 0.006,
                    "volume_ratio": 0.88,
                    "volume_surge": 0.12,
                    "trend_confirmed": False,
                    "short_tf_ready_5m": True,
                    "short_tf_ready_15m": True,
                    "ranging_market": True,
                    "structure_quality": 63.07,
                    "continuation_quality": 48.87,
                    "trade_quality": 99.88,
                    "tp_after_cost_valid": True,
                    "net_edge_pct": 0.10,
                    "range_breakout_1h": False,
                    "pullback_hold": True,
                },
            )
        self.assertTrue(passed)
        self.assertEqual(reason, "passed")

    def test_candidate_gate_blocks_low_net_edge_even_when_score_is_high(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.35,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.0,
                "STABILIZATION_STRICT_ENTRY_ENABLED": False,
            },
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="ALGO/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "ALGO/USD",
                    "lane": "L3",
                    "indicators_ready": True,
                    "entry_score": 88.0,
                    "volume_ratio": 1.1,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.12,
                    "expected_move_pct": 3.0,
                    "price": 1.0,
                    "atr": 0.01,
                },
            )
        self.assertFalse(passed)
        self.assertEqual(reason, "net_edge_below_gate(0.12<0.35)")

    def test_candidate_gate_uses_lane_specific_net_edge_threshold(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "L1_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.20,
                "L2_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.30,
                "L3_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.35,
                "L4_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.45,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.0,
                "STABILIZATION_STRICT_ENTRY_ENABLED": False,
            },
        ):
            passed_l1, reason_l1 = passes_deterministic_candidate_gate(
                symbol="LTC/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "LTC/USD",
                    "lane": "L1",
                    "indicators_ready": True,
                    "entry_score": 88.0,
                    "volume_ratio": 1.2,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.22,
                    "expected_move_pct": 3.0,
                    "price": 1.0,
                    "atr": 0.01,
                    "promotion_tier": "promote",
                    "promotion_reason": "strong_buy_low_risk",
                },
            )
            passed_l3, reason_l3 = passes_deterministic_candidate_gate(
                symbol="ALGO/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "ALGO/USD",
                    "lane": "L3",
                    "indicators_ready": True,
                    "entry_score": 88.0,
                    "volume_ratio": 1.2,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.22,
                    "expected_move_pct": 3.0,
                    "price": 1.0,
                    "atr": 0.01,
                },
            )
        self.assertTrue(passed_l1)
        self.assertEqual(reason_l1, "passed")
        self.assertFalse(passed_l3)
        self.assertEqual(reason_l3, "net_edge_below_gate(0.22<0.35)")

    def test_candidate_gate_blocks_low_risk_reward_even_when_net_edge_is_positive(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_GATE_MIN_ENTRY_SCORE": 48.0,
                "NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.7,
                "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "L3_NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.0,
                "NEMOTRON_GATE_MIN_RISK_REWARD_RATIO": 1.8,
                "EXIT_ATR_STOP_MULT": 1.8,
                "EXIT_MIN_STOP_PCT": 1.5,
                "STABILIZATION_STRICT_ENTRY_ENABLED": False,
            },
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="DOT/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "DOT/USD",
                    "lane": "L3",
                    "indicators_ready": True,
                    "entry_score": 90.0,
                    "volume_ratio": 1.2,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.6,
                    "expected_move_pct": 2.0,
                    "price": 10.0,
                    "atr": 0.10,
                },
            )
        self.assertFalse(passed)
        self.assertEqual(reason, "risk_reward_below_gate(1.11<1.80)")

    def test_strong_mover_can_bypass_top_candidate_filter(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
            allowed = symbol_in_top_candidates(
                "XRP/USD",
                PositionState(),
                {
                    "symbol": "XRP/USD",
                    "lane": "L2",
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 60.29,
                    "rotation_score": 0.14,
                    "momentum_5": 0.02,
                    "volume_ratio": 1.1,
                    "volume_surge": 0.0,
                    "xgb_score": 77.54,
                    "trend_confirmed": False,
                    "short_tf_ready_5m": True,
                    "short_tf_ready_15m": True,
                    "ranging_market": True,
                    "sentiment_symbol_trending": False,
                    "macro_30d": "sideways",
                    "regime_7d": "choppy",
                },
            )
        self.assertTrue(allowed)

    def test_watch_lane_conflict_bypass_requires_real_quality(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={
                "NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT": True,
                "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE": 40.0,
                "NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO": 0.7,
                "LEADER_URGENCY_OVERRIDE_THRESHOLD": 6.0,
            },
        ):
            with patch(
                "core.policy.nemotron_gate.load_universe_candidate_context",
                return_value={
                    "top_scored": [],
                    "hot_candidates": [],
                    "avoid_candidates": [],
                    "top_ranked": [],
                    "lane_supervision": [],
                    "current_symbol_is_top_candidate": False,
                },
            ):
                allowed = symbol_in_top_candidates(
                    "LINK/USD",
                    PositionState(),
                    {
                        "symbol": "LINK/USD",
                        "lane": "L3",
                        "entry_recommendation": "WATCH",
                        "promotion_tier": "probe",
                        "reversal_risk": "LOW",
                        "entry_score": 39.0,
                        "rotation_score": 0.0,
                        "momentum_5": 0.0,
                        "volume_ratio": 0.68,
                        "volume_surge": 0.0,
                        "lane_conflict": True,
                        "leader_takeover": False,
                        "leader_urgency": 0.0,
                    },
                )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
