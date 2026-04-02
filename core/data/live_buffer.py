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
        self._book_ts: dict[str, datetime] = {}
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
            self._book_ts.pop(symbol, None)
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
        existing = self._order_books[symbol]
        # Merge update into existing book — Kraken sends partial updates (only changed levels).
        # Replacing entirely would wipe unchanged side and set it to empty → book_valid=False.
        existing_bids: dict[float, float] = {lvl.price: lvl.qty for lvl in existing.get("bids", [])}
        existing_asks: dict[float, float] = {lvl.price: lvl.qty for lvl in existing.get("asks", [])}
        for price, qty in bids:
            p, q = float(price), float(qty)
            if p > 0.0:
                if q > 0.0:
                    existing_bids[p] = q
                else:
                    existing_bids.pop(p, None)  # qty=0 means level deleted
        for price, qty in asks:
            p, q = float(price), float(qty)
            if p > 0.0:
                if q > 0.0:
                    existing_asks[p] = q
                else:
                    existing_asks.pop(p, None)
        merged_bids = sorted(
            [BookLevel(price=p, qty=q) for p, q in existing_bids.items()],
            key=lambda lvl: lvl.price, reverse=True
        )
        merged_asks = sorted(
            [BookLevel(price=p, qty=q) for p, q in existing_asks.items()],
            key=lambda lvl: lvl.price
        )
        # Partial depth updates can leave stale crossed levels on the opposite side.
        # Prune impossible top-of-book states until the best bid/ask no longer cross.
        while merged_bids and merged_asks and merged_bids[0].price > merged_asks[0].price:
            if merged_bids[0].qty <= merged_asks[0].qty:
                merged_bids.pop(0)
            else:
                merged_asks.pop(0)
        self._order_books[symbol] = {
            "bids": merged_bids[:10],
            "asks": merged_asks[:10],
        }
        self._book_ts[symbol] = datetime.now(timezone.utc)

    def get_market_snapshot(self, symbol: str) -> dict[str, float]:
        if symbol not in self._order_books:
            raise KeyError(f"unknown symbol: {symbol}")
        book = self._order_books[symbol]
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0].price) if bids else 0.0
        best_ask = float(asks[0].price) if asks else 0.0
        book_valid = best_bid > 0.0 and best_ask > 0.0 and best_ask >= best_bid
        book_partial = best_bid > 0.0 or best_ask > 0.0
        if not book_valid:
            best_bid = 0.0
            best_ask = 0.0
        now_utc = datetime.now(timezone.utc)
        book_age_sec = (now_utc - self._book_ts.get(symbol, now_utc)).total_seconds()
        book_fresh = book_age_sec < 5.0
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
            "bid": best_bid,
            "ask": best_ask,
            "spread_pct": spread_pct,
            "book_bid_depth": bid_depth if book_valid else 0.0,
            "book_ask_depth": ask_depth if book_valid else 0.0,
            "book_imbalance": imbalance,
            "book_wall_pressure": wall_pressure,
            "book_valid": book_valid,
            "book_partial": book_partial,
            "book_fresh": book_fresh,
        }

    def seed_ohlc(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        if symbol not in self._buffers:
            raise KeyError(f"unknown symbol: {symbol}")
        if timeframe not in self._buffers[symbol]:
            raise KeyError(f"unknown timeframe: {timeframe}")
        self._buffers[symbol][timeframe].seed_frame(frame)
