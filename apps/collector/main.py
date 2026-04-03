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
from core.data.kraken_rest import KrakenRestClient
from core.config.runtime import is_meme_symbol
from core.symbols import to_kraken_symbol
from core.symbols import normalize_symbol
from core.state.store import load_synced_position_symbols, load_universe, save_universe, read_reload_signal

load_dotenv()


# ---------------------------------------------------------------------------
# Real-time surge detection — injects surging symbols immediately into universe
# ---------------------------------------------------------------------------
_surge_state: dict[str, dict] = {}
_SURGE_THRESHOLD_PCT = float(os.getenv("SURGE_THRESHOLD_PCT", "3.0"))   # min 1m bar move %
_SURGE_VOLUME_RATIO = float(os.getenv("SURGE_VOLUME_RATIO", "2.0"))     # min volume vs baseline
_SURGE_COOLDOWN_SEC = int(os.getenv("SURGE_COOLDOWN_SEC", "300"))        # per-symbol cooldown
_SURGE_MIN_BARS = 6                                                        # min bars needed


def _meme_universe_enabled() -> bool:
    return os.getenv("MEME_UNIVERSE_ENABLED", "true").lower() == "true"


def _meme_active_limit() -> int:
    default_cap = os.getenv("MEME_MAX_OPEN_POSITIONS", "1")
    return max(0, int(os.getenv("MEME_ACTIVE_UNIVERSE_MAX", default_cap)))


def _check_and_maybe_inject_surge(feed: "LiveMarketDataFeed", symbol: str) -> None:
    """Check if a symbol just posted a momentum surge bar. If so, inject into universe.json."""
    now = datetime.now(timezone.utc).timestamp()
    state = _surge_state.setdefault(symbol, {"last_bar_ts": None, "cooldown_until": 0.0})

    if now < state["cooldown_until"]:
        return

    try:
        df = feed.get_ohlc(symbol, "1m", limit=10)
    except KeyError:
        return

    if len(df) < _SURGE_MIN_BARS:
        return

    # Last completed bar is second-to-last (the final row may still be building)
    completed = df.iloc[-2]
    bar_ts = str(completed["timestamp"])

    # Only evaluate once per bar — avoids re-triggering on every tick of the same bar
    if bar_ts == state["last_bar_ts"]:
        return
    state["last_bar_ts"] = bar_ts

    open_price = float(completed["open"])
    if open_price <= 0:
        return
    bar_move = (float(completed["close"]) - open_price) / open_price

    if bar_move < _SURGE_THRESHOLD_PCT / 100.0:
        return

    # Volume ratio vs 4-bar baseline
    baseline_vol = df.iloc[-6:-2]["volume"].mean()
    bar_vol = float(completed["volume"])
    if baseline_vol <= 0:
        return
    vol_ratio = bar_vol / baseline_vol

    if vol_ratio < _SURGE_VOLUME_RATIO:
        return

    # Surge confirmed — inject into universe if not already active
    universe = load_universe()
    active = universe.get("active_pairs", [])
    state["cooldown_until"] = now + _SURGE_COOLDOWN_SEC
    if symbol in active:
        return

    if len(active) >= int(os.getenv("ACTIVE_MAX", "30")):
        return

    if is_meme_symbol(symbol):
        if not _meme_universe_enabled():
            return
        meme_count = sum(1 for pair in active if is_meme_symbol(pair))
        if meme_count >= _meme_active_limit():
            return

    new_pairs = list(dict.fromkeys(active + [symbol]))
    save_universe(new_pairs, f"surge:{symbol}:{bar_move:.3f}x:{vol_ratio:.1f}vol", meta=universe.get("meta", {}))
    print({
        "collector_surge_inject": symbol,
        "bar_move_pct": round(bar_move * 100, 2),
        "vol_ratio": round(vol_ratio, 2),
    })


WS_URL = os.getenv("KRAKEN_WS_URL", "wss://ws.kraken.com/v2")
SYMBOLS = [s.strip() for s in os.getenv("TRADER_SYMBOLS", "ETH/USD").split(",") if s.strip()]
USE_ACTIVE_UNIVERSE = os.getenv("USE_ACTIVE_UNIVERSE", "true").lower() == "true"
UNIVERSE_REFRESH_SEC = int(os.getenv("UNIVERSE_REFRESH_SEC", "60"))
USE_MOCK_STREAM = os.getenv("USE_MOCK_STREAM", "false").lower() == "true"
SNAPSHOT_DIR = Path(os.getenv("LIVE_SNAPSHOT_DIR", "logs/live"))
HISTORY_SNAPSHOT_DIR = Path(os.getenv("HISTORY_SNAPSHOT_DIR", "logs/history"))
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
    # Kraken V2 WS limits per-subscription symbol count; chunk to avoid silent rejections
    _WS_CHUNK_SIZE = int(os.getenv("WS_SUBSCRIPTION_CHUNK_SIZE", "50"))
    kraken_symbols = [_to_kraken_ws_symbol(s) for s in symbols]
    async with connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        for i in range(0, len(kraken_symbols), _WS_CHUNK_SIZE):
            chunk = kraken_symbols[i : i + _WS_CHUNK_SIZE]
            await ws.send(json.dumps({"method": "subscribe", "params": {"channel": "trade", "symbol": chunk}}))
            await ws.send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "book",
                            "symbol": chunk,
                            "depth": 10,
                        },
                    }
                )
            )
            await asyncio.sleep(0.5)
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            channel = msg.get("channel")
            msg_type = msg.get("type") or ""
            # Log subscription confirmations and errors so we can diagnose silent WS failures
            if msg.get("method") == "subscribe" or msg_type in ("subscriptionStatus", "error", "heartbeat"):
                if msg_type == "error" or msg.get("success") is False:
                    print({"collector_ws_error": msg})
                continue
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
                    price = None
                    qty = None
                    if isinstance(level, dict):
                        price = level.get("price")
                        qty = level.get("qty")
                    elif isinstance(level, (list, tuple)) and len(level) >= 2:
                        price = level[0]
                        qty = level[1]
                    try:
                        bids.append((float(price), float(qty)))
                    except (TypeError, ValueError):
                        continue
                for level in item.get("asks") or []:
                    price = None
                    qty = None
                    if isinstance(level, dict):
                        price = level.get("price")
                        qty = level.get("qty")
                    elif isinstance(level, (list, tuple)) and len(level) >= 2:
                        price = level[0]
                        qty = level[1]
                    try:
                        asks.append((float(price), float(qty)))
                    except (TypeError, ValueError):
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


def _to_kraken_ws_symbol(symbol: str) -> str:
    # Kraken WS v2 uses canonical pairs like BTC/USD, while REST/ticker paths
    # still need Kraken-specific aliases such as XBT/USD or XXBTZUSD.
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


def _seed_from_snapshot_path(
    feed: LiveMarketDataFeed,
    *,
    symbol: str,
    timeframe: str,
    path: Path,
) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        frame = CandleLoader(str(path)).load()
    except Exception:
        return False
    if frame.empty:
        return False
    feed.seed_ohlc(symbol, timeframe, frame)
    return True


def _write_snapshot_frame(snapshot_dir: Path, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
    path = snapshot_dir / f"candles_{_symbol_token(symbol)}_{timeframe}.csv"
    frame.to_csv(path, index=False)


def seed_feed_from_snapshots(
    feed: LiveMarketDataFeed,
    symbols: list[str],
    snapshot_dir: Path = SNAPSHOT_DIR,
    history_dir: Path | None = HISTORY_SNAPSHOT_DIR,
) -> None:
    for symbol in symbols:
        for timeframe in ("1m", "5m", "15m", "1h", "7d", "30d"):
            if history_dir is not None:
                history_path = _snapshot_path(history_dir, symbol, timeframe)
                if _seed_from_snapshot_path(feed, symbol=symbol, timeframe=timeframe, path=history_path):
                    continue
            live_path = _snapshot_path(snapshot_dir, symbol, timeframe)
            _seed_from_snapshot_path(feed, symbol=symbol, timeframe=timeframe, path=live_path)


_TF_TO_INTERVAL_MIN: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "7d": 10080,
    "30d": 43200,
}


def seed_feed_from_rest(feed: LiveMarketDataFeed, symbols: list[str], snapshot_dir: Path = SNAPSHOT_DIR) -> None:
    """Fetch OHLC from Kraken REST for any symbol/timeframe missing a local snapshot.
    Called at startup so every symbol is instantly warm regardless of prior active set."""
    client = KrakenRestClient()
    for symbol in symbols:
        for timeframe, interval_min in _TF_TO_INTERVAL_MIN.items():
            path = _snapshot_path(snapshot_dir, symbol, timeframe)
            if path.exists() and path.stat().st_size > 50:
                continue  # Already seeded from CSV (skip empty/header-only files)
            try:
                kraken_pair = to_kraken_symbol(symbol).replace("/", "")
                resp = client.get_ohlc(kraken_pair, interval_min=interval_min)
                result = resp.get("result", {})
                rows = None
                for k, v in result.items():
                    if k == "last":
                        continue
                    if isinstance(v, list) and v:
                        rows = v
                        break
                if not rows:
                    continue
                frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
                frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype(float), unit="s", utc=True)
                for col in ("open", "high", "low", "close", "volume"):
                    frame[col] = frame[col].astype(float)
                frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].tail(200)
                if frame.empty:
                    continue
                feed.seed_ohlc(symbol, timeframe, frame)
                _write_snapshot_frame(snapshot_dir, symbol, timeframe, frame)
            except Exception:
                continue


REST_REFRESH_INTERVAL_SEC = int(os.getenv("REST_REFRESH_INTERVAL_SEC", "60"))
# Primary REST refresh: 1m/5m/15m every cycle (fast-changing signal frames)
_REST_REFRESH_TIMEFRAMES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15}
# Slow refresh: 1h/7d/30d only for symbols missing those frames (rarely needed once seeded)
_REST_SLOW_TIMEFRAMES: dict[str, int] = {"1h": 60, "7d": 10080, "30d": 43200}


async def _rest_refresh_loop(feed: LiveMarketDataFeed) -> None:
    """Data wheel: periodically re-seed the live buffer from Kraken REST API.

    Keeps features computing correctly even when the WebSocket stream stalls or
    disconnects. Refreshes 1m/5m/15m OHLC for all active symbols every
    REST_REFRESH_INTERVAL_SEC seconds, staggering requests to avoid burst rate-limits.
    """
    client = KrakenRestClient()
    while True:
        await asyncio.sleep(REST_REFRESH_INTERVAL_SEC)
        symbols = list(feed.symbols)
        if not symbols:
            continue
        stagger_delay = max(REST_REFRESH_INTERVAL_SEC / max(len(symbols), 1) / len(_REST_REFRESH_TIMEFRAMES), 0.1)
        for symbol in symbols:
            # Always refresh fast frames (1m/5m/15m)
            timeframes_to_fetch = dict(_REST_REFRESH_TIMEFRAMES)
            # Also fetch slow frames (1h/7d/30d) if the symbol is missing them
            for tf, iv in _REST_SLOW_TIMEFRAMES.items():
                if feed.get_ohlc(symbol, tf, limit=1).empty:
                    timeframes_to_fetch[tf] = iv
            for timeframe, interval_min in timeframes_to_fetch.items():
                try:
                    kraken_pair = to_kraken_symbol(symbol).replace("/", "")
                    resp = await asyncio.to_thread(client.get_ohlc, kraken_pair, interval_min=interval_min)
                    result = resp.get("result", {})
                    rows = None
                    for k, v in result.items():
                        if k == "last":
                            continue
                        if isinstance(v, list) and v:
                            rows = v
                            break
                    if not rows:
                        continue
                    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype(float), unit="s", utc=True)
                    for col in ("open", "high", "low", "close", "volume"):
                        frame[col] = frame[col].astype(float)
                    frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].tail(200)
                    if not frame.empty:
                        feed.seed_ohlc(symbol, timeframe, frame)
                        _write_snapshot_frame(SNAPSHOT_DIR, symbol, timeframe, frame)
                except Exception:
                    pass
                await asyncio.sleep(stagger_delay)


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
    refresh_task = asyncio.create_task(_rest_refresh_loop(live_feed))
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
                        _check_and_maybe_inject_surge(live_feed, tick["symbol"])
                    if UNIVERSE_REFRESH_SEC > 0 and (datetime.now(timezone.utc) - started).total_seconds() >= UNIVERSE_REFRESH_SEC:
                        refreshed_symbols = _load_active_symbols(SYMBOLS)
                        if refreshed_symbols != active_symbols:
                            break
            except Exception:
                await asyncio.sleep(1)
    finally:
        for task in (snapshot_task, refresh_task, telemetry_task):
            task.cancel()
        for task in (snapshot_task, refresh_task, telemetry_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
    return live_feed


def main() -> None:
    asyncio.run(collector_loop())


if __name__ == "__main__":
    main()
