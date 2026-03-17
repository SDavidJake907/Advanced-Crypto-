from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque

import pandas as pd


def _floor_time(ts: datetime, interval: timedelta) -> datetime:
    epoch = int(ts.replace(tzinfo=timezone.utc).timestamp())
    seconds = int(interval.total_seconds())
    floored = epoch - (epoch % seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BookLevel:
    price: float
    qty: float


class RollingCandleSeries:
    def __init__(self, interval: timedelta, max_bars: int = 2048) -> None:
        self.interval = interval
        self.max_bars = max_bars
        self._closed: Deque[Candle] = deque(maxlen=max_bars)
        self._current: Candle | None = None

    def update(self, ts: datetime, price: float, volume: float) -> None:
        ts = ts.astimezone(timezone.utc)
        bucket = _floor_time(ts, self.interval)

        if self._current is None:
            self._current = Candle(bucket, price, price, price, price, volume)
            return

        if bucket > self._current.timestamp:
            self._closed.append(self._current)
            self._current = Candle(bucket, price, price, price, price, volume)
            return

        if bucket < self._current.timestamp:
            return

        self._current.high = max(self._current.high, price)
        self._current.low = min(self._current.low, price)
        self._current.close = price
        self._current.volume += volume

    def to_frame(self, limit: int = 200) -> pd.DataFrame:
        rows = list(self._closed)
        if self._current is not None:
            rows.append(self._current)
        if limit > 0:
            rows = rows[-limit:]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(
            [
                {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
                for candle in rows
            ]
        )

    def count(self) -> int:
        return len(self._closed)

    def seed_frame(self, frame: pd.DataFrame) -> None:
        self._closed.clear()
        self._current = None
        if frame.empty:
            return
        rows = frame.sort_values("timestamp").to_dict("records")
        candles = [
            Candle(
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime().astimezone(timezone.utc),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in rows
        ]
        self._closed.extend(candles)
        self._current = None


class LiveMarketDataFeed:
    def __init__(self, symbols: list[str], max_bars: int = 2048) -> None:
        self.symbols = []
        self._buffers: dict[str, dict[str, RollingCandleSeries]] = {}
        self._order_books: dict[str, dict[str, list[BookLevel]]] = {}
        intervals = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        self._intervals = intervals
        self._max_bars = max_bars
        self.set_symbols(symbols)

    def set_symbols(self, symbols: list[str]) -> None:
        ordered = []
        for symbol in symbols:
            if symbol in ordered:
                continue
            ordered.append(symbol)
            if symbol not in self._buffers:
                self._buffers[symbol] = {
                    key: RollingCandleSeries(interval=interval, max_bars=self._max_bars)
                    for key, interval in self._intervals.items()
                }
            if symbol not in self._order_books:
                self._order_books[symbol] = {"bids": [], "asks": []}
        stale_symbols = [symbol for symbol in self._buffers if symbol not in ordered]
        for symbol in stale_symbols:
            del self._buffers[symbol]
            self._order_books.pop(symbol, None)
        self.symbols = ordered

    def on_trade(self, symbol: str, ts: datetime, price: float, volume: float) -> None:
        if symbol not in self._buffers:
            return
        for series in self._buffers[symbol].values():
            series.update(ts, price, volume)

    def get_ohlc(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        if symbol not in self._buffers:
            raise KeyError(f"unknown symbol: {symbol}")
        if timeframe not in self._buffers[symbol]:
            raise KeyError(f"unknown timeframe: {timeframe}")
        return self._buffers[symbol][timeframe].to_frame(limit=limit)

    def get_bar_count(self, symbol: str, timeframe: str) -> int:
        if symbol not in self._buffers:
            raise KeyError(f"unknown symbol: {symbol}")
        if timeframe not in self._buffers[symbol]:
            raise KeyError(f"unknown timeframe: {timeframe}")
        return self._buffers[symbol][timeframe].count()

    def on_book(self, symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        if symbol not in self._order_books:
            return
        sanitized_bids = [
            BookLevel(price=float(price), qty=float(qty))
            for price, qty in bids
            if float(price) > 0.0 and float(qty) > 0.0
        ]
        sanitized_asks = [
            BookLevel(price=float(price), qty=float(qty))
            for price, qty in asks
            if float(price) > 0.0 and float(qty) > 0.0
        ]
        sanitized_bids.sort(key=lambda level: level.price, reverse=True)
        sanitized_asks.sort(key=lambda level: level.price)
        self._order_books[symbol] = {
            "bids": sanitized_bids[:10],
            "asks": sanitized_asks[:10],
        }

    def get_market_snapshot(self, symbol: str) -> dict[str, float]:
        if symbol not in self._order_books:
            raise KeyError(f"unknown symbol: {symbol}")
        book = self._order_books[symbol]
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0].price) if bids else 0.0
        best_ask = float(asks[0].price) if asks else 0.0
        book_valid = best_bid > 0.0 and best_ask > 0.0 and best_ask >= best_bid
        bid_depth = float(sum(level.qty for level in bids))
        ask_depth = float(sum(level.qty for level in asks))
        imbalance = 0.0
        total_depth = bid_depth + ask_depth
        if book_valid and total_depth > 0.0:
            imbalance = (bid_depth - ask_depth) / total_depth
        wall_pressure = 0.0
        if book_valid and bids and asks:
            top_bid = max(level.qty for level in bids)
            top_ask = max(level.qty for level in asks)
            denom = max(top_bid, top_ask, 1e-9)
            wall_pressure = (top_bid - top_ask) / denom
        spread_pct = 0.0
        mid = (best_bid + best_ask) / 2.0
        if book_valid and mid > 0.0:
            spread_pct = ((best_ask - best_bid) / mid) * 100.0
        return {
            "bid": best_bid if book_valid else 0.0,
            "ask": best_ask if book_valid else 0.0,
            "spread_pct": spread_pct,
            "book_bid_depth": bid_depth if book_valid else 0.0,
            "book_ask_depth": ask_depth if book_valid else 0.0,
            "book_imbalance": imbalance,
            "book_wall_pressure": wall_pressure,
            "book_valid": book_valid,
        }

    def seed_ohlc(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        if symbol not in self._buffers:
            raise KeyError(f"unknown symbol: {symbol}")
        if timeframe not in self._buffers[symbol]:
            raise KeyError(f"unknown timeframe: {timeframe}")
        self._buffers[symbol][timeframe].seed_frame(frame)
