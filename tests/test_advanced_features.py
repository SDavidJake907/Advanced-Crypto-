import unittest

import pandas as pd

from core.data.live_buffer import LiveMarketDataFeed
from core.features.batch import _compute_volume_surge
from core.policy.entry_verifier import compute_entry_verification


class AdvancedFeatureTests(unittest.TestCase):
    def test_live_feed_market_snapshot_computes_book_imbalance(self) -> None:
        feed = LiveMarketDataFeed(["BTC/USD"])
        feed.on_book(
            "BTC/USD",
            bids=[(100.0, 5.0), (99.9, 3.0)],
            asks=[(100.1, 2.0), (100.2, 1.0)],
        )
        snapshot = feed.get_market_snapshot("BTC/USD")
        self.assertAlmostEqual(snapshot["bid"], 100.0)
        self.assertAlmostEqual(snapshot["ask"], 100.1)
        self.assertGreater(snapshot["book_imbalance"], 0.0)
        self.assertGreater(snapshot["book_wall_pressure"], 0.0)

    def test_compute_volume_surge_flags_large_relative_bar(self) -> None:
        frame = pd.DataFrame({"volume": [10.0, 10.0, 10.0, 20.0]})
        surge, flags = _compute_volume_surge([frame], lookback=3)
        self.assertGreater(float(surge[0]), 0.5)
        self.assertTrue(bool(flags[0]))

    def test_entry_verifier_uses_microstructure_and_surge(self) -> None:
        result = compute_entry_verification(
            {
                "symbol": "TEST/USD",
                "lane": "L3",
                "lane_filter_pass": True,
                "regime_state": "bullish",
                "momentum_5": 0.01,
                "momentum_14": 0.02,
                "momentum_30": 0.01,
                "rotation_score": 0.2,
                "trend_1h": 1,
                "rsi": 60.0,
                "volume_ratio": 1.8,
                "volume_surge": 0.8,
                "price_zscore": 0.3,
                "regime_7d": "trending",
                "macro_30d": "bull",
                "hurst": 0.6,
                "autocorr": 0.1,
                "entropy": 0.4,
                "correlation_row": [1.0, 0.2],
                "book_imbalance": 0.35,
                "book_wall_pressure": 0.3,
                "sentiment_market_cap_change_24h": 2.5,
                "sentiment_symbol_trending": True,
            }
        )
        self.assertIn("volume_surge", result["entry_reasons"])
        self.assertIn("bid_pressure", result["entry_reasons"])
        self.assertIn("symbol_trending", result["entry_reasons"])
        self.assertIn(result["entry_recommendation"], {"BUY", "STRONG_BUY"})


if __name__ == "__main__":
    unittest.main()
