import unittest
from unittest.mock import patch

from core.policy.nemotron_gate import _passes_market_state_entry_gate, passes_deterministic_candidate_gate
from core.policy.candidate_score import score_candidate
from core.policy.pipeline import apply_policy_pipeline
from core.policy.verdict import extract_policy_verdict
from core.risk.portfolio import PositionState


class PolicyPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_explicit_lane(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L2",
                "momentum": 0.01,
                "momentum_5": 0.02,
                "momentum_14": 0.01,
                "momentum_30": 0.0,
                "trend_1h": 1,
                "volume_ratio": 1.3,
                "price_zscore": -1.0,
                "rsi": 45.0,
                "regime_7d": "choppy",
                "macro_30d": "sideways",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.1,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.5,
                "autocorr": 0.0,
                "entropy": 0.5,
            },
        )
        self.assertEqual(policy["lane"], "L2")

    def test_candidate_score_uses_policy_pipeline_fields(self) -> None:
        scored = score_candidate(
            lane="L3",
            momentum_5=0.02,
            momentum_14=0.015,
            momentum_30=0.01,
            trend_1h=1,
            volume_ratio=1.6,
            price_zscore=0.4,
            rsi=58.0,
            spread_bps=8.0,
            rotation_score=0.12,
        )
        self.assertGreaterEqual(scored.candidate_score, 48.0)
        self.assertIn(scored.candidate_recommendation, {"BUY", "STRONG_BUY", "WATCH", "MEDIUM"})
        self.assertIn(scored.candidate_risk, {"LOW", "MEDIUM", "HIGH"})
        self.assertTrue(scored.candidate_reasons)

    def test_pipeline_attaches_policy_verdict(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "momentum": 0.01,
                "momentum_5": 0.02,
                "momentum_14": 0.01,
                "momentum_30": 0.0,
                "trend_1h": 1,
                "volume_ratio": 1.3,
                "price_zscore": -1.0,
                "rsi": 45.0,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.1,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.6,
                "autocorr": 0.1,
                "entropy": 0.5,
            },
        )
        verdict = extract_policy_verdict(policy)
        self.assertEqual(policy["policy_verdict"]["lane"], "L3")
        self.assertEqual(verdict.lane, "L3")
        self.assertEqual(verdict.entry_recommendation, policy["entry_recommendation"])
        self.assertEqual(verdict.reversal_risk, policy["reversal_risk"])

    def test_entry_reasons_are_not_silently_truncated(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "momentum": 0.03,
                "momentum_5": 0.03,
                "momentum_14": 0.02,
                "momentum_30": 0.01,
                "trend_1h": 1,
                "volume_ratio": 1.6,
                "price_zscore": 1.8,
                "rsi": 80.0,
                "regime_7d": "choppy",
                "macro_30d": "bull",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.2,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [0.5, 0.4, 0.3],
                "hurst": 0.6,
                "autocorr": 0.1,
                "entropy": 0.5,
                "lane_filter_pass": False,
                "lane_filter_reason": "lane3_rsi_hot",
            },
        )
        reasons = policy["entry_reasons"]
        self.assertGreater(len(reasons), 6)
        self.assertIn("regime_choppy", reasons)
        self.assertIn("macro_bull", reasons)
        self.assertIn("persistent_trend", reasons)
        self.assertIn("overbought_penalty", reasons)
        self.assertIn("fast_momentum", reasons)
        self.assertIn("volume_expansion", reasons)
        self.assertIn("rotation_leader", reasons)
        self.assertIn("lane3_rsi_hot", reasons)

    def test_low_volume_becomes_soft_warning_not_hard_filter(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "momentum": 0.01,
                "momentum_5": 0.004,
                "momentum_14": 0.006,
                "momentum_30": 0.002,
                "trend_1h": 1,
                "volume_ratio": 0.35,
                "volume_surge": 0.01,
                "price_zscore": 0.3,
                "rsi": 55.0,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.08,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.58,
                "autocorr": 0.05,
                "entropy": 0.5,
            },
        )
        self.assertTrue(policy["lane_filter_pass"])
        self.assertEqual(policy["lane_filter_reason"], "lane3_vol_low_warning")
        self.assertEqual(policy["lane_filter_severity"], "soft")
        self.assertEqual(policy["entry_recommendation"], "BUY")
        self.assertEqual(policy["promotion_tier"], "skip")

    def test_lane2_real_mover_gets_promoted(self) -> None:
        policy = apply_policy_pipeline(
            "APT/USD",
            {
                "symbol": "APT/USD",
                "lane": "L2",
                "indicators_ready": True,
                "momentum": 0.01,
                "momentum_5": 0.006,
                "momentum_14": 0.01,
                "momentum_30": 0.002,
                "trend_1h": 1,
                "volume_ratio": 0.95,
                "volume_surge": 0.3,
                "price_zscore": 0.2,
                "rsi": 55.0,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.09,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.58,
                "autocorr": 0.05,
                "entropy": 0.5,
                "short_tf_ready_5m": True,
                "trade_quality": 58.0,
                "continuation_quality": 62.0,
                "structure_quality": 60.0,
            },
        )
        self.assertEqual(policy["entry_recommendation"], "STRONG_BUY")
        self.assertEqual(policy["promotion_tier"], "promote")
        self.assertEqual(policy["promotion_reason"], "strong_buy_low_risk")

    def test_watch_path_without_mover_signal_does_not_crash(self) -> None:
        policy = apply_policy_pipeline(
            "TEST/USD",
            {
                "symbol": "TEST/USD",
                "lane": "L2",
                "indicators_ready": True,
                "momentum": 0.0,
                "momentum_5": 0.0,
                "momentum_14": 0.001,
                "momentum_30": 0.0,
                "trend_1h": 0,
                "trend_confirmed": False,
                "volume_ratio": 0.2,
                "volume_surge": 0.0,
                "price_zscore": 0.0,
                "rsi": 50.0,
                "regime_7d": "sideways",
                "macro_30d": "sideways",
                "price": 100.0,
                "atr": 1.0,
                "bb_bandwidth": 0.02,
                "spread_pct": 0.001,
                "rotation_score": 0.0,
                "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                "correlation_row": [],
                "hurst": 0.5,
                "autocorr": 0.0,
                "entropy": 0.5,
                "short_tf_ready_5m": False,
                "short_tf_ready_15m": False,
            },
        )
        self.assertEqual(policy["promotion_tier"], "skip")
        self.assertIn(policy["entry_recommendation"], {"WATCH", "MEDIUM", "AVOID"})

    def test_market_state_gate_blocks_lane2_downtrend_without_strong_override(self) -> None:
        passed, reason = _passes_market_state_entry_gate(
            {
                "symbol": "TEST/USD",
                "lane": "L2",
                "trend_1h": -1,
                "trend_confirmed": False,
                "momentum_5": -0.001,
                "momentum_14": -0.002,
                "ranging_market": False,
            }
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "market_state_downtrend_block")

    def test_market_state_gate_blocks_lane3_transition_without_breakout_support(self) -> None:
        passed, reason = _passes_market_state_entry_gate(
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "trend_1h": 0,
                "trend_confirmed": False,
                "momentum_5": 0.0,
                "momentum_14": 0.002,
                "ranging_market": False,
                "range_breakout_1h": False,
                "pullback_hold": False,
            }
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "market_state_transition_weak_block")

    def test_market_state_gate_keeps_range_safe_lane2_mover_override(self) -> None:
        passed, reason = _passes_market_state_entry_gate(
            {
                "symbol": "TEST/USD",
                "lane": "L2",
                "entry_score": 68.0,
                "volume_ratio": 1.05,
                "volume_surge": 0.22,
                "net_edge_pct": 0.4,
                "tp_after_cost_valid": True,
                "trend_1h": -1,
                "trend_confirmed": False,
                "momentum_5": 0.01,
                "momentum_14": 0.004,
                "ranging_market": True,
                "short_tf_ready_15m": True,
                "range_breakout_1h": True,
                "pullback_hold": False,
                "trade_quality": 65.0,
                "structure_quality": 66.0,
                "continuation_quality": 67.0,
                "rotation_score": 0.12,
            }
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "market_state_ranging_override")

    def test_deterministic_gate_returns_market_state_reason_before_stabilization(self) -> None:
        passed, reason = passes_deterministic_candidate_gate(
            symbol="TEST/USD",
            positions_state=PositionState(),
            universe_context={},
            features={
                "symbol": "TEST/USD",
                "lane": "L3",
                "indicators_ready": True,
                "entry_score": 62.0,
                "volume_ratio": 1.2,
                "net_edge_pct": 0.2,
                "trend_1h": 0,
                "trend_confirmed": False,
                "momentum_5": 0.0,
                "momentum_14": 0.002,
                "ranging_market": False,
                "promotion_tier": "promote",
                "promotion_reason": "strong_buy_low_risk",
            },
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "market_state_transition_weak_block")

    def test_deterministic_gate_no_longer_blocks_only_on_non_positive_net_edge(self) -> None:
        with patch(
            "core.config.runtime.load_runtime_overrides",
            return_value={"STABILIZATION_STRICT_ENTRY_ENABLED": False},
        ):
            passed, reason = passes_deterministic_candidate_gate(
                symbol="TEST/USD",
                positions_state=PositionState(),
                universe_context={},
                features={
                    "symbol": "TEST/USD",
                    "lane": "L3",
                    "indicators_ready": True,
                    "entry_score": 70.0,
                    "volume_ratio": 1.2,
                    "net_edge_pct": 0.0,
                    "trend_1h": 1,
                    "trend_confirmed": True,
                    "momentum_5": 0.01,
                    "momentum_14": 0.01,
                    "ranging_market": False,
                    "promotion_tier": "promote",
                    "promotion_reason": "strong_buy_low_risk",
                },
            )
        self.assertTrue(passed)
        self.assertEqual(reason, "passed")


if __name__ == "__main__":
    unittest.main()
