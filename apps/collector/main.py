from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from websockets import connect

from core.data.loader import CandleLoader
from core.data.live_buffer import LiveMarketDataFeed
from core.symbols import normalize_symbol
from core.state.store import load_synced_position_symbols, load_universe, read_reload_signal

load_dotenv()


WS_URL = os.getenv("KRAKEN_WS_URL", "wss://ws.kraken.com/v2")
SYMBOLS = [s.strip() for s in os.getenv("TRADER_SYMBOLS", "ETH/USD").split(",") if s.strip()]
USE_ACTIVE_UNIVERSE = os.getenv("USE_ACTIVE_UNIVERSE", "true").lower() == "true"
UNIVERSE_REFRESH_SEC = int(os.getenv("UNIVERSE_REFRESH_SEC", "60"))
USE_MOCK_STREAM = os.getenv("USE_MOCK_STREAM", "false").lower() == "true"
SNAPSHOT_DIR = Path(os.getenv("LIVE_SNAPSHOT_DIR", "logs/live"))
COLLECTOR_TELEMETRY_PATH = Path(os.getenv("COLLECTOR_TELEMETRY_PATH", "logs/collector_telemetry.json"))


async def mock_price_stream():
    """Simulated price ticks for local development."""
    price = 2300.0
    while True:
        price += random.uniform(-1.0, 1.5)
        yield {
            "symbol": SYMBOLS[0] if SYMBOLS else "ETH/USD",
            "timestamp": datetime.utcnow(),
            "price": price,
            "volume": random.uniform(50, 200),
        }
        await asyncio.sleep(1)


async def _kraken_trade_stream(symbols: list[str]):
    async with connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({"method": "subscribe", "params": {"channel": "trade", "symbol": symbols}}))
        await ws.send(
            json.dumps(
                {
                    "method": "subscribe",
                    "params": {
                        "channel": "book",
                        "symbol": symbols,
                        "depth": 10,
                    },
                }
            )
        )
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            channel = msg.get("channel")
            if channel == "trade":
                for item in msg.get("data") or []:
                    raw_symbol = item.get("symbol") or item.get("pair")
                    symbol = normalize_symbol(raw_symbol) if raw_symbol else None
                    price = item.get("price")
                    volume = item.get("qty") or item.get("volume")
                    ts_raw = item.get("timestamp") or item.get("time") or item.get("ts")
                    if symbol is None or price is None or volume is None or ts_raw is None:
                        continue
                    try:
                        parsed_volume = float(volume)
                    except (TypeError, ValueError):
                        continue
                    if parsed_volume <= 0.0:
                        print(
                            {
                                "collector_warning": "invalid_trade_volume",
                                "raw_symbol": raw_symbol,
                                "symbol": symbol,
                                "raw_volume": volume,
                            }
                        )
                        continue
                    if isinstance(ts_raw, str):
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    else:
                        ts = datetime.fromtimestamp(float(ts_raw) / 1000.0, tz=timezone.utc)
                    yield {"type": "trade", "symbol": symbol, "timestamp": ts, "price": float(price), "volume": parsed_volume}
            elif channel == "book":
                data = msg.get("data") or []
                if not data:
                    continue
                item = data[0]
                raw_symbol = item.get("symbol") or item.get("pair")
                symbol = normalize_symbol(raw_symbol) if raw_symbol else None
                if symbol is None:
                    continue
                bids = []
                asks = []
                for level in item.get("bids") or []:
                    try:
                        bids.append((float(level.get("price")), float(level.get("qty"))))
                    except (TypeError, ValueError, AttributeError):
                        continue
                for level in item.get("asks") or []:
                    try:
                        asks.append((float(level.get("price")), float(level.get("qty"))))
                    except (TypeError, ValueError, AttributeError):
                        continue
                yield {"type": "book", "symbol": symbol, "bids": bids, "asks": asks}


def _write_collector_telemetry(payload: dict[str, object]) -> None:
    COLLECTOR_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTOR_TELEMETRY_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


async def _telemetry_loop(feed: LiveMarketDataFeed, counters: dict[str, object]) -> None:
    while True:
        per_symbol_bars: dict[str, dict[str, int]] = {}
        for symbol in feed.symbols:
            per_symbol_bars[symbol] = {
                timeframe: len(feed.get_ohlc(symbol, timeframe, limit=500))
                for timeframe in ("1m", "5m", "15m", "1h", "7d", "30d")
            }
        top_depth = {}
        for symbol in feed.symbols[:12]:
            try:
                top_depth[symbol] = feed.get_market_snapshot(symbol)
            except KeyError:
                continue
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbols": list(feed.symbols),
            "trade_counts": counters,
            "per_symbol_bars": per_symbol_bars,
            "top_depth": top_depth,
        }
        _write_collector_telemetry(payload)
        await asyncio.sleep(15)


def _normalize_active_symbol(symbol: str) -> str:
    return normalize_symbol(symbol)


def _load_active_symbols(default_symbols: list[str]) -> list[str]:
    if USE_ACTIVE_UNIVERSE:
        active = load_universe().get("active_pairs", [])
        normalized = [_normalize_active_symbol(symbol) for symbol in active if str(symbol).strip()]
    else:
        normalized = list(default_symbols)
    merged = list(dict.fromkeys(normalized + load_synced_position_symbols()))
    return merged or default_symbols


def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _snapshot_path(snapshot_dir: Path, symbol: str, timeframe: str) -> Path:
    return snapshot_dir / f"candles_{_symbol_token(symbol)}_{timeframe}.csv"


def seed_feed_from_snapshots(feed: LiveMarketDataFeed, symbols: list[str], snapshot_dir: Path = SNAPSHOT_DIR) -> None:
    for symbol in symbols:
        for timeframe in ("1m", "5m", "15m", "1h", "7d", "30d"):
            path = _snapshot_path(snapshot_dir, symbol, timeframe)
            if not path.exists():
                continue
            frame = CandleLoader(str(path)).load()
            if frame.empty:
                continue
            feed.seed_ohlc(symbol, timeframe, frame)


async def _snapshot_loop(feed: LiveMarketDataFeed, snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for symbol in feed.symbols:
            for timeframe in ("1m", "5m", "15m", "1h", "7d", "30d"):
                frame = feed.get_ohlc(symbol, timeframe, limit=200)
                path = snapshot_dir / f"candles_{_symbol_token(symbol)}_{timeframe}.csv"
                await asyncio.to_thread(frame.to_csv, path, index=False)
        await asyncio.sleep(5)


async def collector_loop(feed: LiveMarketDataFeed | None = None) -> LiveMarketDataFeed:
    active_symbols = _load_active_symbols(SYMBOLS)
    live_feed = feed or LiveMarketDataFeed(active_symbols)
    live_feed.set_symbols(active_symbols)
    seed_feed_from_snapshots(live_feed, active_symbols)
    snapshot_task = asyncio.create_task(_snapshot_loop(live_feed, SNAPSHOT_DIR))
    counters: dict[str, object] = {
        "received": 0,
        "processed": 0,
        "dropped_unknown_symbol": 0,
        "invalid_volume": 0,
        "last_warning": None,
    }
    telemetry_task = asyncio.create_task(_telemetry_loop(live_feed, counters))
    last_reload_token = read_reload_signal()
    try:
        while True:
            previous_symbols = list(live_feed.symbols)
            active_symbols = _load_active_symbols(SYMBOLS)
            live_feed.set_symbols(active_symbols)
            added_symbols = [symbol for symbol in active_symbols if symbol not in previous_symbols]
            if added_symbols:
                seed_feed_from_snapshots(live_feed, added_symbols)
            stream = mock_price_stream() if USE_MOCK_STREAM else _kraken_trade_stream(active_symbols)
            started = datetime.now(timezone.utc)
            try:
                async for tick in stream:
                    current_reload_token = read_reload_signal()
                    if current_reload_token != last_reload_token:
                        last_reload_token = current_reload_token
                        break
                    counters["received"] = int(counters["received"]) + 1
                    if tick["symbol"] not in live_feed.symbols:
                        counters["dropped_unknown_symbol"] = int(counters["dropped_unknown_symbol"]) + 1
                        counters["last_warning"] = {
                            "kind": "unknown_symbol",
                            "symbol": tick["symbol"],
                        }
                        continue
                    if tick.get("type") == "book":
                        live_feed.on_book(
                            tick["symbol"],
                            tick.get("bids", []),
                            tick.get("asks", []),
                        )
                        counters["processed"] = int(counters["processed"]) + 1
                    else:
                        if float(tick["volume"]) <= 0.0:
                            counters["invalid_volume"] = int(counters["invalid_volume"]) + 1
                            counters["last_warning"] = {
                                "kind": "invalid_volume",
                                "symbol": tick["symbol"],
                                "volume": tick["volume"],
                            }
                            continue
                        live_feed.on_trade(
                            tick["symbol"],
                            tick["timestamp"].replace(tzinfo=timezone.utc),
                            float(tick["price"]),
                            float(tick["volume"]),
                        )
                        counters["processed"] = int(counters["processed"]) + 1
                    if UNIVERSE_REFRESH_SEC > 0 and (datetime.now(timezone.utc) - started).total_seconds() >= UNIVERSE_REFRESH_SEC:
                        refreshed_symbols = _load_active_symbols(SYMBOLS)
                        if refreshed_symbols != active_symbols:
                            break
            except Exception:
                await asyncio.sleep(1)
    finally:
        for task in (snapshot_task, telemetry_task):
            task.cancel()
        for task in (snapshot_task, telemetry_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
    return live_feed


def main() -> None:
    asyncio.run(collector_loop())


if __name__ == "__main__":
    main()
