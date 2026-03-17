import unittest

import pandas as pd

from apps.trader.main import _market_sentiment_fallback
from core.data.live_buffer import LiveMarketDataFeed
from core.features.batch import _compute_short_tf_features
from core.models.xgb_entry import XGBEntryModel


class _TrendingCoin:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _Snapshot:
    def __init__(self, fng_value: int, market_cap_change_24h: float, trending: list[str]) -> None:
        self.fng_value = fng_value
        self.market_cap_change_24h = market_cap_change_24h
        self.trending = [_TrendingCoin(symbol) for symbol in trending]


class RuntimeSignalHealthTests(unittest.TestCase):
    def test_partial_book_snapshot_does_not_publish_invalid_quotes(self) -> None:
        feed = LiveMarketDataFeed(["BTC/USD"])
        feed.on_book("BTC/USD", bids=[], asks=[(101.0, 1.0)])
        snapshot = feed.get_market_snapshot("BTC/USD")
        self.assertFalse(snapshot["book_valid"])
        self.assertEqual(snapshot["bid"], 0.0)
        self.assertEqual(snapshot["ask"], 0.0)
        self.assertEqual(snapshot["spread_pct"], 0.0)

    def test_short_tf_features_require_minimum_bar_count(self) -> None:
        short_frame = pd.DataFrame({"close": [1.0, 1.1, 1.2, 1.3, 1.4]})
        momentum, trend, ready = _compute_short_tf_features(
            {"BTC/USD": short_frame},
            ["BTC/USD"],
            lookback=3,
            min_bars=6,
        )
        self.assertEqual(momentum, [0.0])
        self.assertEqual(trend, [0])
        self.assertEqual(ready, [False])

    def test_market_sentiment_fallback_returns_non_zero_for_supportive_snapshot(self) -> None:
        score = _market_sentiment_fallback("DOGE/USD", _Snapshot(65, 2.0, ["DOGE"]))
        self.assertGreater(score, 0.0)

    def test_xgb_model_uses_heuristic_proxy_when_model_missing(self) -> None:
        model = XGBEntryModel()
        score = model.predict(
            {
                "momentum_5": 0.01,
                "momentum_14": 0.01,
                "momentum_30": 0.01,
                "rsi": 58.0,
                "atr": 1.0,
                "volume_surge": 0.7,
                "rotation_score": 0.2,
                "volume_ratio": 1.5,
                "book_imbalance": 0.3,
                "finbert_score": 0.2,
                "trend_1h": 1,
            }
        )
        self.assertGreater(score, 50.0)


if __name__ == "__main__":
    unittest.main()
