import unittest
from unittest.mock import patch

from core.policy.nemotron_gate import build_universe_candidate_context, should_run_nemotron
from core.risk.portfolio import PositionState


class NemotronPolicyTests(unittest.TestCase):
    def test_build_universe_candidate_context_accepts_cached_universe_shape(self) -> None:
        context = build_universe_candidate_context(
            "FET/USD",
            {
                "meta": {
                    "top_scored": [{"symbol": "FET/USD", "leader_urgency": 7.5}],
                    "hot_candidates": [{"symbol": "DOGE/USD"}],
                    "avoid_candidates": [{"symbol": "UNI/USD"}],
                    "top_ranked": ["FET/USD", "DOGE/USD"],
                    "lane_supervision": [{"symbol": "FET/USD", "lane": "meme"}],
                }
            },
        )

        self.assertTrue(context["current_symbol_is_top_candidate"])
        self.assertEqual(context["top_scored"][0]["symbol"], "FET/USD")
        self.assertEqual(context["top_ranked"][0], "FET/USD")

    def test_should_run_nemotron_blocks_ranging_market_under_stabilization_gate(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "STABILIZATION_STRICT_ENTRY_ENABLED": "true",
                "STABILIZATION_ALLOWED_LANES": "L2,L3",
                "STABILIZATION_MIN_ENTRY_SCORE": "70",
                "STABILIZATION_MIN_NET_EDGE_PCT": "0.0",
                "STABILIZATION_REQUIRE_TP_AFTER_COST_VALID": "true",
                "STABILIZATION_REQUIRE_TREND_CONFIRMED": "true",
                "STABILIZATION_REQUIRE_SHORT_TF_READY_15M": "true",
                "STABILIZATION_BLOCK_RANGING_MARKET": "true",
                "STABILIZATION_REQUIRE_BUY_RECOMMENDATION": "true",
            },
            clear=False,
        ):
            with patch("core.config.runtime.load_runtime_overrides", return_value={}):
                allowed = should_run_nemotron(
                    symbol="BTC/USD",
                    features={
                        "lane": "L3",
                        "promotion_tier": "promote",
                        "entry_recommendation": "BUY",
                        "reversal_risk": "MEDIUM",
                        "entry_score": 75.0,
                        "volume_ratio": 1.2,
                        "trend_confirmed": True,
                        "short_tf_ready_15m": True,
                        "ranging_market": True,
                        "tp_after_cost_valid": True,
                        "net_edge_pct": 0.5,
                    },
                    positions_state=PositionState(),
                    universe_context={"current_symbol_is_top_candidate": True},
                )
        self.assertFalse(allowed)

    def test_should_run_nemotron_allows_clean_setup_under_stabilization_gate(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "STABILIZATION_STRICT_ENTRY_ENABLED": "true",
                "STABILIZATION_ALLOWED_LANES": "L2,L3",
                "STABILIZATION_MIN_ENTRY_SCORE": "70",
                "STABILIZATION_MIN_NET_EDGE_PCT": "0.0",
                "STABILIZATION_REQUIRE_TP_AFTER_COST_VALID": "true",
                "STABILIZATION_REQUIRE_TREND_CONFIRMED": "true",
                "STABILIZATION_REQUIRE_SHORT_TF_READY_15M": "true",
                "STABILIZATION_BLOCK_RANGING_MARKET": "true",
                "STABILIZATION_REQUIRE_BUY_RECOMMENDATION": "true",
            },
            clear=False,
        ):
            with patch("core.config.runtime.load_runtime_overrides", return_value={}):
                allowed = should_run_nemotron(
                    symbol="BTC/USD",
                    features={
                        "lane": "L3",
                        "promotion_tier": "promote",
                        "entry_recommendation": "BUY",
                        "reversal_risk": "MEDIUM",
                        "entry_score": 75.0,
                        "volume_ratio": 1.2,
                        "trend_confirmed": True,
                        "short_tf_ready_15m": True,
                        "ranging_market": False,
                        "tp_after_cost_valid": True,
                        "net_edge_pct": 0.5,
                    },
                    positions_state=PositionState(),
                    universe_context={"current_symbol_is_top_candidate": True},
                )
        self.assertTrue(allowed)

    def test_should_run_nemotron_blocks_non_buy_recommendation_under_stabilization_gate(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "STABILIZATION_STRICT_ENTRY_ENABLED": "true",
                "STABILIZATION_ALLOWED_LANES": "L2,L3",
                "STABILIZATION_MIN_ENTRY_SCORE": "70",
                "STABILIZATION_MIN_NET_EDGE_PCT": "0.0",
                "STABILIZATION_REQUIRE_TP_AFTER_COST_VALID": "true",
                "STABILIZATION_REQUIRE_TREND_CONFIRMED": "true",
                "STABILIZATION_REQUIRE_SHORT_TF_READY_15M": "true",
                "STABILIZATION_BLOCK_RANGING_MARKET": "false",
                "STABILIZATION_REQUIRE_BUY_RECOMMENDATION": "true",
            },
            clear=False,
        ):
            with patch("core.config.runtime.load_runtime_overrides", return_value={}):
                allowed = should_run_nemotron(
                    symbol="BTC/USD",
                    features={
                        "lane": "L3",
                        "promotion_tier": "promote",
                        "entry_recommendation": "WATCH",
                        "reversal_risk": "MEDIUM",
                        "entry_score": 83.0,
                        "volume_ratio": 1.2,
                        "trend_confirmed": True,
                        "short_tf_ready_15m": True,
                        "ranging_market": False,
                        "tp_after_cost_valid": True,
                        "net_edge_pct": 0.5,
                    },
                    positions_state=PositionState(),
                    universe_context={"current_symbol_is_top_candidate": True},
                )
        self.assertFalse(allowed)

    def test_should_run_nemotron_respects_promote_tier(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={"NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0, "L3_NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0, "L2_NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0}):
            allowed = should_run_nemotron(
                symbol="TEST/USD",
                features={
                    "promotion_tier": "promote",
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 61.0,
                    "rotation_score": 0.12,
                    "momentum_5": 0.01,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.5,
                    "volume_ratio": 1.5,
                },
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
            )
        self.assertTrue(allowed)

    def test_should_run_nemotron_no_longer_blocks_only_on_non_positive_net_edge(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={"NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0, "L3_NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0}):
            allowed = should_run_nemotron(
                symbol="TEST/USD",
                features={
                    "promotion_tier": "promote",
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 85.0,
                    "volume_ratio": 1.4,
                    "trend_confirmed": True,
                    "short_tf_ready_15m": True,
                    "ranging_market": False,
                    "tp_after_cost_valid": True,
                    "net_edge_pct": 0.0,
                },
                positions_state=PositionState(),
                    universe_context={"current_symbol_is_top_candidate": True},
                )
        self.assertTrue(allowed)

    def test_should_run_nemotron_allows_buy_candidate_with_positive_technicals(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={"NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0, "L3_NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0, "L2_NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0}):
            allowed = should_run_nemotron(
                symbol="TEST/USD",
                features={
                    "entry_recommendation": "BUY",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 61.0,
                    "rotation_score": 0.12,
                    "momentum_5": 0.01,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.5,
                    "volume_ratio": 1.5,
                },
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
            )
        self.assertTrue(allowed)

    def test_should_run_nemotron_allows_probe_when_candidate_review_promotes(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={"NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0, "L3_NEMOTRON_GATE_MIN_VOLUME_RATIO": 0.0}):
            allowed = should_run_nemotron(
                symbol="TEST/USD",
                features={
                    "promotion_tier": "probe",
                    "entry_recommendation": "WATCH",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 70.0,
                    "rotation_score": 0.2,
                    "momentum_5": 0.01,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.5,
                    "volume_ratio": 1.5,
                },
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
            )
        self.assertTrue(allowed)

    def test_should_run_nemotron_still_blocks_watch_without_real_promotion(self) -> None:
        with patch("core.config.runtime.load_runtime_overrides", return_value={}):
            allowed = should_run_nemotron(
                symbol="TEST/USD",
                features={
                    "entry_recommendation": "WATCH",
                    "reversal_risk": "MEDIUM",
                    "entry_score": 68.0,
                    "rotation_score": 0.2,
                    "momentum_5": 0.01,
                    "trend_confirmed": True,
                    "net_edge_pct": 0.5,
                },
                positions_state=PositionState(),
                universe_context={"current_symbol_is_top_candidate": False},
            )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
