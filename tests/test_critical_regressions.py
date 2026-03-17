import unittest

import pandas as pd

from core.features.batch import _compute_volume_ratio
from core.risk.fee_filter import evaluate_trade_cost
from core.state.portfolio import PortfolioState


class CriticalRegressionTests(unittest.TestCase):
    def test_fee_filter_spread_is_not_divided_twice(self) -> None:
        assessment = evaluate_trade_cost(
            {
                "lane": "L3",
                "spread_pct": 0.10,
                "price": 100.0,
                "atr": 1.0,
                "momentum_5": 0.0,
                "momentum_14": 0.0,
                "rotation_score": 0.0,
                "entry_score": 0.0,
            },
            "LONG",
        )
        self.assertGreater(assessment.total_cost_pct, 0.10)

    def test_portfolio_mark_to_market_uses_symbol_specific_marks(self) -> None:
        portfolio = PortfolioState(
            cash=50.0,
            positions={"AAA/USD": 1.0, "BBB/USD": 2.0},
            position_marks={"AAA/USD": 10.0, "BBB/USD": 20.0},
            initial_equity=80.0,
        )
        self.assertEqual(portfolio._compute_pnl_mark_to_market(999.0), 20.0)

    def test_volume_ratio_uses_full_lookback_baseline(self) -> None:
        frame = pd.DataFrame({"volume": [1.0, 2.0, 3.0, 4.0]})
        ratios = _compute_volume_ratio([frame], lookback=3)
        self.assertAlmostEqual(float(ratios[0]), 4.0 / ((1.0 + 2.0 + 3.0) / 3.0))


if __name__ == "__main__":
    unittest.main()
