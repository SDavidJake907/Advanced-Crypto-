import unittest

import pandas as pd

from apps.trader.cycle import (
    apply_lane_supervision,
    market_sentiment_fallback,
    select_symbols_for_cycle,
)
from apps.trader.positions import prune_symbol_tracking_state
from core.data.live_buffer import LiveMarketDataFeed
from core.features.batch import _compute_short_tf_features
from core.models.xgb_entry import XGBEntryModel
from core.risk.portfolio import Position, PositionState


class _TrendingCoin:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _Snapshot:
    def __init__(self, fng_value: int, market_cap_change_24h: float, trending: list[str]) -> None:
        self.fng_value = fng_value
        self.market_cap_change_24h = market_cap_change_24h
        self.trending = [_TrendingCoin(symbol) for symbol in trending]


class RuntimeSignalHealthTests(unittest.TestCase):
    def test_select_symbols_for_cycle_is_symbol_local(self) -> None:
        frames = {
            "BTC/USD": pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")], "close": [100.0]}),
            "DOGE/USD": pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")], "close": [10.0]}),
            "SOL/USD": pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:01:00Z")], "close": [20.0]}),
        }
        active, static, prices = select_symbols_for_cycle(
            ["BTC/USD", "DOGE/USD", "SOL/USD"],
            frames,
            last_bar_keys={
                "BTC/USD": "2026-01-01T00:00:00+00:00",
                "DOGE/USD": "2026-01-01T00:00:00+00:00",
                "SOL/USD": "2026-01-01T00:00:00+00:00",
            },
            last_prices={"BTC/USD": 100.0, "DOGE/USD": 9.0, "SOL/USD": 20.0},
            intrabar_price_threshold=0.05,
        )
        self.assertEqual(active, ["DOGE/USD", "SOL/USD"])
        self.assertEqual(static, ["BTC/USD"])
        self.assertEqual(prices["DOGE/USD"], 10.0)

    def test_partial_book_snapshot_does_not_publish_invalid_quotes(self) -> None:
        feed = LiveMarketDataFeed(["BTC/USD"])
        feed.on_book("BTC/USD", bids=[], asks=[(101.0, 1.0)])
        snapshot = feed.get_market_snapshot("BTC/USD")
        self.assertFalse(snapshot["book_valid"])
        self.assertEqual(snapshot["bid"], 0.0)
        self.assertEqual(snapshot["ask"], 0.0)
        self.assertEqual(snapshot["spread_pct"], 0.0)

    def test_crossed_partial_book_levels_are_pruned(self) -> None:
        feed = LiveMarketDataFeed(["BTC/USD"])
        feed.on_book("BTC/USD", bids=[(100.0, 5.0), (99.9, 2.0)], asks=[(100.1, 1.0), (100.2, 1.0)])
        feed.on_book("BTC/USD", bids=[(101.0, 1.0)], asks=[])
        snapshot = feed.get_market_snapshot("BTC/USD")
        self.assertTrue(snapshot["book_valid"])
        self.assertLessEqual(snapshot["bid"], snapshot["ask"])
        self.assertGreater(snapshot["ask"], 0.0)

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
        score = market_sentiment_fallback("DOGE/USD", _Snapshot(65, 2.0, ["DOGE"]))
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

    def test_apply_lane_supervision_preserves_dynamic_lane(self) -> None:
        features = {"symbol": "DOGE/USD", "lane": "L1"}
        merged = apply_lane_supervision(
            features,
            {
                "universe_lane": "L4",
                "lane_candidate": "meme",
                "lane_conflict": True,
            },
        )
        self.assertEqual(merged["lane"], "L1")
        self.assertEqual(merged["universe_lane"], "L4")
        self.assertTrue(merged["lane_conflict"])

    def test_prune_symbol_tracking_state_removes_rotated_out_symbols(self) -> None:
        positions_state = PositionState()
        positions_state.add_or_update(Position(symbol="BTC/USD", side="LONG", weight=0.1))
        last_prices = {"BTC/USD": 100.0, "OLD/USD": 10.0}
        last_bar_keys = {"BTC/USD": "k1", "OLD/USD": "k2"}
        last_exit_ts = {"OLD/USD": 1.0, "DOGE/USD": 9999999999.0}

        prune_symbol_tracking_state(
            ["BTC/USD", "SOL/USD"],
            positions_state,
            last_prices=last_prices,
            last_bar_keys=last_bar_keys,
            last_exit_ts=last_exit_ts,
            retention_sec=10.0,
        )

        self.assertIn("BTC/USD", last_prices)
        self.assertNotIn("OLD/USD", last_prices)
        self.assertNotIn("OLD/USD", last_bar_keys)
        self.assertNotIn("OLD/USD", last_exit_ts)
        self.assertIn("DOGE/USD", last_exit_ts)


if __name__ == "__main__":
    unittest.main()
