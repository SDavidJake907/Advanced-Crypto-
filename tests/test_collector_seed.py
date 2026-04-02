import tempfile
import unittest
from pathlib import Path

import pandas as pd

from apps.collector.main import _to_kraken_ws_symbol, seed_feed_from_snapshots


class _FakeFeed:
    def __init__(self) -> None:
        self.seeded: dict[tuple[str, str], pd.DataFrame] = {}

    def seed_ohlc(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        self.seeded[(symbol, timeframe)] = frame.copy()


def _write_candles(path: Path, closes: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


class CollectorSeedTests(unittest.TestCase):
    def test_ws_symbol_mapping_keeps_canonical_btc_pair(self) -> None:
        self.assertEqual(_to_kraken_ws_symbol("BTC/USD"), "BTC/USD")
        self.assertEqual(_to_kraken_ws_symbol("XBT/USD"), "BTC/USD")

    def test_seed_feed_prefers_history_snapshots_over_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / "history"
            live_dir = root / "live"
            history_path = history_dir / "candles_ETHUSD_1h.csv"
            live_path = live_dir / "candles_ETHUSD_1h.csv"
            _write_candles(history_path, [10.0, 11.0, 12.0])
            _write_candles(live_path, [1.0, 2.0])

            feed = _FakeFeed()
            seed_feed_from_snapshots(
                feed,
                ["ETH/USD"],
                snapshot_dir=live_dir,
                history_dir=history_dir,
            )

            seeded = feed.seeded[("ETH/USD", "1h")]
            self.assertEqual(len(seeded), 3)
            self.assertEqual(float(seeded["close"].iloc[-1]), 12.0)

    def test_seed_feed_falls_back_to_live_when_history_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / "history"
            live_dir = root / "live"
            live_path = live_dir / "candles_ETHUSD_1h.csv"
            _write_candles(live_path, [1.0, 2.0, 3.0, 4.0])

            feed = _FakeFeed()
            seed_feed_from_snapshots(
                feed,
                ["ETH/USD"],
                snapshot_dir=live_dir,
                history_dir=history_dir,
            )

            seeded = feed.seeded[("ETH/USD", "1h")]
            self.assertEqual(len(seeded), 4)
            self.assertEqual(float(seeded["close"].iloc[-1]), 4.0)


if __name__ == "__main__":
    unittest.main()
