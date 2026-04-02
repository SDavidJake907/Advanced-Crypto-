import unittest
from unittest.mock import patch

from core.policy.nemotron_gate import passes_deterministic_candidate_gate, symbol_in_top_candidates
from core.risk.portfolio import PositionState


class MoverGateTests(unittest.TestCase):
    def test_ranging_market_allows_strong_mover_buy(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
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

    def test_ranging_market_still_blocks_weak_setup(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="ICP/USD",
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
                features={
                    "symbol": "ICP/USD",
                    "lane": "L2",
                    "indicators_ready": True,
                    "entry_recommendation": "WATCH",
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
        self.assertFalse(passed)
        self.assertEqual(reason, "ranging_unconfirmed_structure_weak")

    def test_ranging_unconfirmed_blocks_high_score_weak_continuation_setup(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
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
        self.assertFalse(passed)
        self.assertEqual(reason, "ranging_unconfirmed_structure_weak")

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
        allowed = symbol_in_top_candidates(
            "LINK/USD",
            PositionState(),
            {
                "symbol": "LINK/USD",
                "lane": "L3",
                "entry_recommendation": "WATCH",
                "promotion_tier": "probe",
                "reversal_risk": "LOW",
                "entry_score": 42.0,
                "rotation_score": 0.02,
                "momentum_5": 0.001,
                "volume_ratio": 0.7,
                "volume_surge": 0.0,
                "lane_conflict": True,
                "leader_takeover": False,
                "leader_urgency": 0.0,
            },
        )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
