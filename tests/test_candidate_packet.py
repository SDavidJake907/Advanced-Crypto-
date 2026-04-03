import unittest

from core.risk.portfolio import Position, PositionState
from core.policy.candidate_packet import (
    build_candidate_packet,
    build_local_nemotron_candidate_packet,
    compute_candidate_economics,
)
from core.policy.final_score import compute_final_score


class CandidatePacketTests(unittest.TestCase):
    def test_candidate_packet_matches_final_score_truth(self) -> None:
        features = {
            "symbol": "ETH/USD",
            "lane": "L3",
            "price": 3000.0,
            "entry_score": 62.5,
            "entry_recommendation": "BUY",
            "reversal_risk": "LOW",
            "momentum_5": 0.01,
            "momentum_14": 0.02,
            "rotation_score": 0.625,
            "volume_ratio": 1.2,
            "volume_surge": 0.3,
            "rsi": 56.0,
            "spread_pct": 0.4,
            "price_zscore": 0.2,
            "trend_1h": 1,
            "regime_7d": "trending",
            "macro_30d": "bull",
            "atr_pct": 0.012,
            "volatility_percentile": 44.0,
            "compression_score": 71.0,
            "expansion_score": 39.0,
            "volatility_state": "compressed",
            "bullish_divergence": True,
            "bearish_divergence": False,
            "divergence_strength": 0.5,
            "divergence_age_bars": 2,
            "sentiment_fng_value": 75,
            "trend_confirmed": True,
            "ranging_market": False,
            "point_breakdown": {"cost_penalty_pts": 5.0, "net_edge_pct": 0.45},
        }
        expected = compute_final_score(
            features,
            reliability_map={"ETH/USD": {"win_rate": 0.61, "trade_count": 10}},
            held_correlation_map={"BTC/USD": 0.35, "SOL/USD": 0.25},
        )
        positions_state = PositionState()
        positions_state.add_or_update(Position(symbol="BTC/USD", side="LONG", weight=0.1))
        positions_state.add_or_update(Position(symbol="SOL/USD", side="LONG", weight=0.1))
        packet = build_candidate_packet(
            features=features,
            positions_state=positions_state,
            lesson_summary=["avoid thin follow-through", "prefer clean pullback holds"],
            behavior_score={"score": 0.71},
            reliability_map={"ETH/USD": {"win_rate": 0.61, "trade_count": 10}},
            held_correlation_map={"BTC/USD": 0.35, "SOL/USD": 0.25},
        )

        self.assertEqual(packet["final_score"], expected.final_score)
        self.assertEqual(packet["score_breakdown"], expected.score_breakdown)
        self.assertEqual(packet["net_edge_pct"], expected.net_edge_pct)
        self.assertEqual(packet["fear_greed_bonus"], expected.fear_greed_bonus)
        self.assertEqual(packet["reliability_bonus"], expected.reliability_bonus)
        self.assertEqual(packet["basket_fit_bonus"], expected.basket_fit_bonus)
        self.assertEqual(packet["breakdown_notes"], expected.breakdown_notes)
        self.assertEqual(packet["spread_penalty"], expected.spread_penalty)
        self.assertEqual(packet["cost_penalty"], expected.cost_penalty)
        self.assertEqual(packet["correlation_penalty"], expected.correlation_penalty)
        self.assertEqual(packet["behavior_score"], {"score": 0.71})
        self.assertEqual(packet["lesson_summary"], ["avoid thin follow-through", "prefer clean pullback holds"])
        self.assertEqual(packet["held_correlation_map"], {"BTC/USD": 0.35, "SOL/USD": 0.25})
        self.assertEqual(packet["volatility_state"], "compressed")

    def test_compute_candidate_economics_is_direct_final_score_wrapper(self) -> None:
        features = {
            "symbol": "DOGE/USD",
            "lane": "L4",
            "entry_score": 58.0,
            "spread_pct": 1.2,
            "point_breakdown": {"cost_penalty_pts": 10.0, "net_edge_pct": -0.75},
            "bearish_divergence": True,
            "divergence_strength": 1.0,
            "divergence_age_bars": 12,
            "reflex": {"reflex": "block"},
        }
        wrapped = compute_candidate_economics(
            features,
            reliability_map={"DOGE/USD": {"win_rate": 0.32, "trade_count": 8}},
        )
        direct = compute_final_score(
            features,
            reliability_map={"DOGE/USD": {"win_rate": 0.32, "trade_count": 8}},
        )

        self.assertEqual(wrapped.final_score, direct.final_score)
        self.assertEqual(wrapped.score_breakdown, direct.score_breakdown)

    def test_candidate_packet_builds_held_correlation_map_and_normalizes_lessons(self) -> None:
        positions_state = PositionState()
        positions_state.add_or_update(Position(symbol="BTC/USD", side="LONG", weight=0.1))
        positions_state.add_or_update(Position(symbol="ADA/USD", side="LONG", weight=0.1))
        features = {
            "symbol": "ETH/USD",
            "lane": "L3",
            "price": 3000.0,
            "entry_score": 55.0,
            "spread_pct": 0.4,
            "correlation_row": [0.91, 0.22, 0.35],
            "correlation_symbols": ["BTC/USD", "SOL/USD", "ADA/USD"],
            "point_breakdown": {"cost_penalty_pts": 0.0, "net_edge_pct": 0.2},
        }

        packet = build_candidate_packet(
            features=features,
            positions_state=positions_state,
            lesson_summary="=== RECENT ETH/USD LESSONS ===\n- keep sizes smaller\n- avoid late chase\n",
            behavior_score=0.63,
            reliability_map={},
        )

        self.assertEqual(packet["held_correlation_map"], {"BTC/USD": 0.91, "ADA/USD": 0.35})
        self.assertEqual(packet["lesson_summary"], ["keep sizes smaller", "avoid late chase"])
        self.assertEqual(packet["behavior_score"], 0.63)

    def test_local_nemotron_candidate_packet_is_compact(self) -> None:
        packet = build_candidate_packet(
            features={
                "symbol": "PENDLE/USD",
                "lane": "L2",
                "price": 1.2,
                "entry_score": 74.0,
                "entry_recommendation": "BUY",
                "reversal_risk": "LOW",
                "momentum_5": 0.02,
                "momentum_14": 0.03,
                "rotation_score": 0.65,
                "volume_ratio": 1.6,
                "rsi": 58.0,
                "spread_pct": 0.2,
                "trend_1h": 1,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "atr_pct": 0.011,
                "volatility_percentile": 48.0,
                "compression_score": 67.0,
                "expansion_score": 42.0,
                "volatility_state": "normal",
                "trend_confirmed": True,
                "ranging_market": False,
                "short_tf_ready_15m": True,
                "ema9_above_ema20": True,
                "pullback_hold": True,
                "structure_quality": 72.0,
                "pattern_candidate": {"pattern": "double_bottom", "confidence_raw": 0.79},
                "pattern_verification": {"validity": "valid", "pattern_quality_score": 0.81, "summary": "Clean breakout"},
                "phi3_veto_flag": False,
                "point_breakdown": {"cost_penalty_pts": 0.0, "net_edge_pct": 1.1},
            },
            lesson_summary=["prefer clean pullbacks", "avoid late extension", "keep size normal", "extra noise"],
            behavior_score={
                "score": 0.74,
                "threshold_advice": "keep_threshold",
                "confidence": 0.61,
                "notes": "drop this",
            },
            held_correlation_map={"BTC/USD": 0.5},
        )

        compact = build_local_nemotron_candidate_packet(packet)

        self.assertEqual(compact["symbol"], "PENDLE/USD")
        self.assertEqual(compact["lesson_summary"], ["prefer clean pullbacks", "avoid late extension", "keep size normal"])
        self.assertEqual(
            compact["behavior_score"],
            {"score": 0.74, "threshold_advice": "keep_threshold", "confidence": 0.61},
        )
        self.assertEqual(compact["pattern_candidate"], {"pattern": "double_bottom", "confidence_raw": 0.79})
        self.assertEqual(
            compact["pattern_verification"],
            {"validity": "valid", "pattern_quality_score": 0.81, "summary": "Clean breakout"},
        )
        self.assertEqual(compact["volatility_state"], "normal")
        self.assertNotIn("score_breakdown", compact)
        self.assertNotIn("breakdown_notes", compact)
        self.assertNotIn("held_correlation_map", compact)
        self.assertNotIn("trade_quality", compact)
        self.assertNotIn("risk_quality", compact)
        self.assertNotIn("continuation_quality", compact)


if __name__ == "__main__":
    unittest.main()
