import unittest

from core.llm.micro_prompts import (
    _heuristic_candidate_review,
    _heuristic_market_state_review,
    _heuristic_posture_review,
)


class MoverAdvisoryTests(unittest.TestCase):
    def test_candidate_review_promotes_early_mover_probe(self) -> None:
        review = _heuristic_candidate_review(
            {
                "entry_recommendation": "WATCH",
                "reversal_risk": "MEDIUM",
                "entry_score": 54.0,
                "rotation_score": 0.12,
                "momentum_5": 0.01,
                "volume_surge": 0.2,
                "xgb_score": 65.0,
                "short_tf_ready_5m": True,
                "short_tf_ready_15m": True,
                "sentiment_symbol_trending": False,
            }
        )
        self.assertEqual(review.promotion_decision, "promote")
        self.assertEqual(review.action_bias, "reduce_size")
        self.assertEqual(review.reason, "watch_mover_probe")

    def test_market_state_ranging_stays_selective_for_mover(self) -> None:
        review = _heuristic_market_state_review(
            {
                "ranging_market": True,
                "entry_score": 58.0,
                "momentum_5": 0.01,
                "rotation_score": 0.05,
                "volume_surge": 0.1,
                "sentiment_symbol_trending": False,
            }
        )
        self.assertEqual(review.market_state, "ranging")
        self.assertEqual(review.lane_bias, "favor_selective")
        self.assertEqual(review.reason, "selective_range_mover_with_relative_strength")
        self.assertEqual(review.breakout_state, "breakout_attempt")
        self.assertEqual(review.trend_stage, "mixed")
        self.assertEqual(review.late_move_risk, "moderate")
        self.assertEqual(review.pattern_explanation["structure_pattern"], "breakout_attempt")
        self.assertEqual(review.pattern_explanation["structure_validity"], "developing")
        self.assertEqual(review.candle_evidence["primary_candle"], "none")

    def test_market_state_trending_has_strict_chart_handoff_fields(self) -> None:
        review = _heuristic_market_state_review(
            {
                "ranging_market": False,
                "trend_confirmed": True,
                "entry_score": 74.0,
                "momentum_5": 0.012,
                "momentum_14": 0.02,
                "rotation_score": 0.1,
                "volume_ratio": 1.35,
                "volume_surge": 0.4,
                "macd_hist": 0.002,
                "ema9_above_ema20": True,
                "ema9_above_ema26": True,
                "range_breakout_1h": True,
                "pullback_hold": False,
                "higher_low_count": 3,
            }
        )
        self.assertEqual(review.market_state, "trending")
        self.assertEqual(review.breakout_state, "fresh_breakout")
        self.assertEqual(review.trend_stage, "confirmed")
        self.assertEqual(review.volume_confirmation, "supportive")
        self.assertEqual(review.late_move_risk, "contained")
        self.assertEqual(review.pattern_explanation["structure_pattern"], "range_breakout")
        self.assertEqual(review.pattern_explanation["structure_validity"], "valid")
        self.assertEqual(review.pattern_explanation["recommended_nemo_interpretation"]["prefer_action"], "OPEN")
        self.assertEqual(review.candle_evidence["primary_candle"], "bullish_engulfing")
        self.assertEqual(review.candle_evidence["candle_bias"], "bullish")

    def test_posture_ranging_promoted_mover_is_not_defensive(self) -> None:
        review = _heuristic_posture_review(
            {},
            {"market_state": "ranging"},
            {"promotion_decision": "promote"},
        )
        self.assertEqual(review.posture, "neutral")
        self.assertEqual(review.reason, "heuristic_from_candidate_review")


if __name__ == "__main__":
    unittest.main()
