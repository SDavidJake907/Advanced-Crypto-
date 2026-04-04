import unittest

from core.policy.final_score import compute_final_score


class FinalScoreRefactorTests(unittest.TestCase):
    def test_final_score_preserves_existing_outputs_for_bullish_case(self) -> None:
        result = compute_final_score(
            {
                "symbol": "ETH/USD",
                "entry_score": 62.5,
                "reflex": {"reflex": "allow"},
                "lane": "L3",
                "bullish_divergence": True,
                "divergence_strength": 0.5,
                "divergence_age_bars": 2,
                "spread_pct": 0.4,
                "point_breakdown": {"cost_penalty_pts": 5.0, "net_edge_pct": 0.45},
                "correlation_row": [0.2, 0.4, 0.65],
            },
            reliability_map={"ETH/USD": {"win_rate": 0.61, "trade_count": 10}},
            held_correlation_map={"BTC/USD": 0.35, "SOL/USD": 0.25},
        )

        self.assertEqual(result.final_score, 72.7)
        self.assertEqual(result.entry_score, 62.5)
        self.assertEqual(result.reflex_bonus, 5.0)
        self.assertEqual(result.divergence_bonus, 3.0)
        self.assertEqual(result.reliability_bonus, 4.0)
        self.assertEqual(result.basket_fit_bonus, 8.0)
        self.assertAlmostEqual(result.spread_penalty, 4.8)
        self.assertEqual(result.cost_penalty, 5.0)
        self.assertEqual(result.correlation_penalty, 0.0)
        self.assertEqual(result.net_edge_pct, 0.45)
        self.assertEqual(result.fear_greed_bonus, 0.0)
        self.assertEqual(result.btc_dominance_bonus, 0.0)

    def test_score_breakdown_groups_contributions_without_renaming_public_fields(self) -> None:
        result = compute_final_score(
            {
                "symbol": "DOGE/USD",
                "entry_score": 58.0,
                "reflex": {"reflex": "block"},
                "lane": "L4",
                "bearish_divergence": True,
                "divergence_strength": 1.0,
                "divergence_age_bars": 12,
                "spread_pct": 1.2,
                "point_breakdown": {"cost_penalty_pts": 10.0, "net_edge_pct": -0.75},
                "correlation_row": [0.85, 0.91],
            },
            reliability_map={"DOGE/USD": {"win_rate": 0.32, "trade_count": 8}},
            held_correlation_map={"BTC/USD": 0.35, "SOL/USD": 0.25},
        )

        breakdown = result.score_breakdown
        self.assertEqual(result.final_score, 20.82)
        self.assertEqual(breakdown["setup_contribution"], 52.22)
        self.assertEqual(breakdown["reliability_contribution"], -10.0)
        self.assertEqual(breakdown["cost_penalty_contribution"], -24.4)
        self.assertEqual(breakdown["basket_contribution"], 3.0)
        self.assertEqual(breakdown["setup_pts"], 58.0)
        self.assertEqual(breakdown["reflex_bonus"], -5.0)
        self.assertEqual(breakdown["divergence_bonus"], -0.8)
        self.assertEqual(breakdown["fear_greed_bonus"], 0.0)
        self.assertEqual(breakdown["btc_dominance_bonus"], 0.0)
        self.assertEqual(breakdown["reliability_bonus"], -10.0)
        self.assertEqual(breakdown["basket_fit"], 8.0)
        self.assertEqual(breakdown["spread_penalty"], -14.4)
        self.assertEqual(breakdown["cost_penalty"], -10.0)
        self.assertEqual(breakdown["correlation_penalty"], -5.0)
        self.assertEqual(breakdown["net_edge_pct"], -0.75)
        self.assertEqual(breakdown["notes"], result.breakdown_notes)

    def test_fear_greed_adds_small_bonus_for_confirmed_trend(self) -> None:
        result = compute_final_score(
            {
                "symbol": "BTC/USD",
                "entry_score": 60.0,
                "reflex": {"reflex": "allow"},
                "lane": "L1",
                "trend_confirmed": True,
                "ranging_market": False,
                "sentiment_fng_value": 75,
                "spread_pct": 0.2,
                "point_breakdown": {"cost_penalty_pts": 2.0, "net_edge_pct": 0.5},
            },
        )

        self.assertAlmostEqual(result.fear_greed_bonus, 1.5)
        self.assertEqual(result.score_breakdown["fear_greed_bonus"], 1.5)
        self.assertEqual(result.btc_dominance_bonus, 0.0)

    def test_fear_greed_penalizes_extreme_fear_when_trend_unconfirmed(self) -> None:
        result = compute_final_score(
            {
                "symbol": "ETH/USD",
                "entry_score": 60.0,
                "reflex": {"reflex": "allow"},
                "lane": "L3",
                "trend_confirmed": False,
                "ranging_market": False,
                "sentiment_fng_value": 10,
                "spread_pct": 0.2,
                "point_breakdown": {"cost_penalty_pts": 2.0, "net_edge_pct": 0.5},
            },
        )

        self.assertAlmostEqual(result.fear_greed_bonus, -1.2)
        self.assertEqual(result.score_breakdown["fear_greed_bonus"], -1.2)
        self.assertEqual(result.btc_dominance_bonus, 0.0)

    def test_btc_dominance_penalizes_unconfirmed_alt_setup(self) -> None:
        result = compute_final_score(
            {
                "symbol": "ALGO/USD",
                "entry_score": 60.0,
                "reflex": {"reflex": "allow"},
                "lane": "L2",
                "trend_confirmed": False,
                "ranging_market": True,
                "sentiment_btc_dominance": 60.0,
                "spread_pct": 0.2,
                "point_breakdown": {"cost_penalty_pts": 2.0, "net_edge_pct": 0.5},
            },
        )

        self.assertLess(result.btc_dominance_bonus, 0.0)
        self.assertEqual(result.score_breakdown["btc_dominance_bonus"], round(result.btc_dominance_bonus, 1))

    def test_btc_dominance_gives_small_btc_tailwind(self) -> None:
        result = compute_final_score(
            {
                "symbol": "BTC/USD",
                "entry_score": 60.0,
                "reflex": {"reflex": "allow"},
                "lane": "L1",
                "trend_confirmed": True,
                "ranging_market": False,
                "sentiment_btc_dominance": 60.0,
                "spread_pct": 0.2,
                "point_breakdown": {"cost_penalty_pts": 2.0, "net_edge_pct": 0.5},
            },
        )

        self.assertEqual(result.btc_dominance_bonus, -0.6)
        self.assertEqual(result.score_breakdown["btc_dominance_bonus"], round(result.btc_dominance_bonus, 1))


if __name__ == "__main__":
    unittest.main()
