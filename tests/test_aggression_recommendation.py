import unittest

from core.policy.aggression_recommendation import recommend_aggression_mode


class AggressionRecommendationTests(unittest.TestCase):
    def test_recommends_defensive_in_bearish_high_dominance_fearful_market(self) -> None:
        result = recommend_aggression_mode(
            runtime_values={"AGGRESSION_MODE": "NORMAL"},
            universe_meta={
                "news_context": {
                    "btc_dominance": 60.2,
                    "fng_value": 18,
                    "market_cap_change_24h": -2.4,
                }
            },
            recent_decision_debug=[
                {"market_trend": "bear", "market_trend_strength": -3},
                {"symbol": "BTC/USD", "signal": "FLAT"},
                {"symbol": "ETH/USD", "signal": "FLAT"},
                {"symbol": "SOL/USD", "signal": "FLAT"},
            ],
        )
        self.assertEqual(result.mode, "DEFENSIVE")
        self.assertGreaterEqual(result.confidence, 0.65)
        self.assertEqual(result.btc_dominance_level, "alt_high_caution")
        self.assertEqual(result.alt_market_posture, "fragile_alts")

    def test_recommends_offensive_in_constructive_bull_market(self) -> None:
        result = recommend_aggression_mode(
            runtime_values={"AGGRESSION_MODE": "NORMAL"},
            universe_meta={
                "news_context": {
                    "btc_dominance": 53.4,
                    "fng_value": 61,
                    "market_cap_change_24h": 2.1,
                }
            },
            recent_decision_debug=[
                {"market_trend": "bull", "market_trend_strength": 3},
                {"symbol": "BTC/USD", "signal": "LONG", "execution_status": "submitted"},
                {"symbol": "ETH/USD", "signal": "LONG", "execution_status": "submitted"},
                {"symbol": "SOL/USD", "signal": "FLAT"},
            ],
        )
        self.assertIn(result.mode, {"OFFENSIVE", "HIGH_OFFENSIVE"})
        self.assertGreaterEqual(result.score, 1.5)
        self.assertEqual(result.btc_dominance_level, "neutral")
        self.assertEqual(result.alt_market_posture, "constructive_alts")

    def test_recommends_normal_when_inputs_are_mixed(self) -> None:
        result = recommend_aggression_mode(
            runtime_values={"AGGRESSION_MODE": "OFFENSIVE"},
            universe_meta={
                "news_context": {
                    "btc_dominance": 56.0,
                    "fng_value": 44,
                    "market_cap_change_24h": 0.2,
                }
            },
            recent_decision_debug=[
                {"market_trend": "neutral", "market_trend_strength": 0},
                {"symbol": "BTC/USD", "signal": "FLAT"},
                {"symbol": "ETH/USD", "signal": "FLAT"},
                {"symbol": "SOL/USD", "signal": "LONG", "execution_status": "submitted"},
            ],
        )
        self.assertEqual(result.mode, "NORMAL")
        self.assertEqual(result.btc_dominance_level, "alt_caution")
        self.assertEqual(result.alt_market_posture, "cautious_alts")


if __name__ == "__main__":
    unittest.main()
