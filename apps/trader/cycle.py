"""Per-cycle data preparation utilities for the trader loop.

Responsibilities:
- OHLC fetching (CSV and live feed)
- Symbol filtering (new-bar / intrabar-move detection)
- Lane supervision map loading and feature overlay
- Decision timing formatting
- Market sentiment fallback scoring
- Async executor wrapper for blocking calls
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from apps.trader.boot import (
    CANDLES_PATH,
    CANDLES_PATH_TEMPLATE,
    _resolve_candles_path,
)
from core.data.live_buffer import LiveMarketDataFeed
from core.data.loader import CandleLoader
from core.state.store import load_universe


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def run_blocking_decision(func, /, *args, **kwargs):
    """Run a blocking callable in a thread without blocking the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


# ---------------------------------------------------------------------------
# OHLC fetchers
# ---------------------------------------------------------------------------

async def fetch_ohlc(
    symbol: str,
    *,
    base_path: str = CANDLES_PATH,
    template: str = CANDLES_PATH_TEMPLATE,
) -> pd.DataFrame:
    loader = CandleLoader(_resolve_candles_path(symbol, base_path, template))
    if not loader.path.exists():
        raise FileNotFoundError(f"Candles CSV not found: {loader.path}")
    return loader.load().tail(200)


async def fetch_live_ohlc(
    feed: LiveMarketDataFeed, symbol: str, timeframe: str
) -> pd.DataFrame:
    return feed.get_ohlc(symbol, timeframe, limit=200)


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def frame_last_price(frame: pd.DataFrame) -> float:
    if frame.empty or "close" not in frame.columns:
        return 0.0
    return float(frame["close"].iloc[-1])


def frame_last_bar_key(frame: pd.DataFrame) -> str | None:
    if frame.empty or "timestamp" not in frame.columns:
        return None
    return pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat()


# ---------------------------------------------------------------------------
# Symbol selection
# ---------------------------------------------------------------------------

def select_symbols_for_cycle(
    symbols: list[str],
    ohlc_by_symbol: dict[str, pd.DataFrame],
    *,
    last_bar_keys: dict[str, str],
    last_prices: dict[str, float],
    intrabar_price_threshold: float,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Split symbols into active (need evaluation) and static (no change).

    Returns:
        (active_symbols, static_symbols, prices_by_symbol)
    """
    active: list[str] = []
    static: list[str] = []
    prices: dict[str, float] = {}

    for symbol in symbols:
        frame = ohlc_by_symbol.get(symbol)
        if frame is None or frame.empty:
            continue
        current_price = frame_last_price(frame)
        prices[symbol] = current_price
        current_bar_key = frame_last_bar_key(frame)
        last_bar_key = last_bar_keys.get(symbol)
        last_price = last_prices.get(symbol)

        if current_bar_key is None or last_bar_key is None or current_bar_key != last_bar_key:
            active.append(symbol)
            continue
        if last_price is None or last_price <= 0.0:
            active.append(symbol)
            continue
        change = abs(current_price - last_price) / last_price
        if change >= intrabar_price_threshold:
            active.append(symbol)
            continue
        static.append(symbol)

    return active, static, prices


# ---------------------------------------------------------------------------
# Lane supervision
# ---------------------------------------------------------------------------

def _overlay_leader_metrics(
    mapping: dict[str, dict[str, Any]], items: list[Any] | None
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        entry = mapping.setdefault(symbol, {})
        entry["leader_urgency"] = float(item.get("leader_urgency", 0.0) or 0.0)
        entry["leader_takeover"] = bool(item.get("leader_takeover", False))
        for key in (
            "expected_move_pct",
            "total_cost_pct",
            "required_edge_pct",
            "net_edge_pct",
            "tp_after_cost_valid",
            "realized_cost_penalty_pct",
            "realized_slippage_pct",
            "realized_follow_through_pct",
        ):
            if key in item:
                entry[key] = item.get(key)


def load_lane_supervision_map() -> dict[str, dict[str, Any]]:
    """Build a per-symbol supervision map from the latest universe.json."""
    universe = load_universe()
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    mapping: dict[str, dict[str, Any]] = {}

    # Universe-assigned lane from lane_shortlists (authoritative)
    lane_shortlists = meta.get("lane_shortlists", {})
    if isinstance(lane_shortlists, dict):
        for lane_name, items in lane_shortlists.items():
            if not isinstance(items, list) or lane_name == "merged":
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).strip().upper()
                if symbol:
                    mapping.setdefault(symbol, {})["universe_lane"] = lane_name.upper()

    # Phi-3 lane supervision overlay
    lane_supervision = meta.get("lane_supervision", [])
    if isinstance(lane_supervision, list):
        for item in lane_supervision:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol:
                mapping.setdefault(symbol, {}).update({
                    "lane_candidate":  item.get("lane_candidate"),
                    "lane_confidence": item.get("lane_confidence"),
                    "lane_reason":     item.get("lane_reason"),
                    "lane_conflict":   item.get("lane_conflict"),
                    "narrative_tag":   item.get("narrative_tag"),
                })

    # Leader metrics from all universe candidate lists
    _overlay_leader_metrics(mapping, meta.get("leaderboard", []))
    _overlay_leader_metrics(mapping, meta.get("top_scored", []))
    _overlay_leader_metrics(mapping, meta.get("hot_candidates", []))
    if isinstance(lane_shortlists, dict):
        for items in lane_shortlists.values():
            if isinstance(items, list):
                _overlay_leader_metrics(mapping, items)

    return mapping


def apply_lane_supervision(
    features: dict[str, Any],
    lane_supervision: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge universe lane assignment and Phi-3 advice into the features dict."""
    if not lane_supervision:
        return features
    merged = dict(features)
    universe_lane = str(
        (lane_supervision or {}).get("universe_lane") or merged.get("lane") or "L3"
    ).upper()
    merged["universe_lane"] = universe_lane
    merged["lane"] = str(merged.get("lane") or universe_lane).upper()

    for src_key, dst_key in (
        ("lane_candidate",  "phi3_lane_candidate"),
        ("lane_confidence", "phi3_lane_confidence"),
        ("lane_reason",     "phi3_lane_reason"),
    ):
        val = lane_supervision.get(src_key)
        if val is not None:
            merged[dst_key] = val

    for key in ("lane_conflict", "narrative_tag"):
        if key in lane_supervision:
            merged[key] = lane_supervision.get(key)

    if "leader_urgency" in lane_supervision:
        merged["leader_urgency"] = float(lane_supervision.get("leader_urgency", 0.0) or 0.0)
    if "leader_takeover" in lane_supervision:
        merged["leader_takeover"] = bool(lane_supervision.get("leader_takeover", False))

    # Keep universe/leader metrics as advisory context only.
    # Live per-cycle economics must come from the current feature/scoring pipeline,
    # not from cached universe overlays.
    for key in (
        "expected_move_pct",
        "total_cost_pct",
        "required_edge_pct",
        "net_edge_pct",
        "realized_cost_penalty_pct",
        "realized_slippage_pct",
        "realized_follow_through_pct",
    ):
        if key in lane_supervision:
            merged[f"universe_{key}"] = float(lane_supervision.get(key, 0.0) or 0.0)

    if "tp_after_cost_valid" in lane_supervision:
        merged["universe_tp_after_cost_valid"] = bool(lane_supervision.get("tp_after_cost_valid", False))

    return merged


# ---------------------------------------------------------------------------
# Decision timing formatter
# ---------------------------------------------------------------------------

def format_decision_timings(
    features_ms: float, timings: dict[str, Any]
) -> dict[str, float]:
    """Normalise raw timing measurements into a consistent output dict."""
    f_ms  = max(round(float(features_ms or 0.0), 2), 0.0)
    total = max(float(timings.get("total_ms", 0.0) or 0.0), 0.0)
    return {
        "features_ms":    f_ms,
        "phi3_ms":        max(float(timings.get("phi3_ms",       0.0) or 0.0), 0.0),
        "advisory_ms":    max(float(timings.get("advisory_ms",   0.0) or 0.0), 0.0),
        "nemotron_ms":    max(float(timings.get("nemotron_ms",   0.0) or 0.0), 0.0),
        "execution_ms":   max(float(timings.get("execution_ms",  0.0) or 0.0), 0.0),
        "decision_ms":    round(total, 2),
        "cycle_total_ms": round(f_ms + total, 2),
    }


# ---------------------------------------------------------------------------
# Market sentiment fallback
# ---------------------------------------------------------------------------

def market_sentiment_fallback(symbol: str, sentiment_snapshot: object) -> float:
    """Compute a simple [-1, +1] sentiment score when no dedicated feed is available."""
    fng = float(getattr(sentiment_snapshot, "fng_value", 50) or 50.0)
    mkt_chg = float(getattr(sentiment_snapshot, "market_cap_change_24h", 0.0) or 0.0)
    trending = [
        str(getattr(item, "symbol", "")).upper()
        for item in getattr(sentiment_snapshot, "trending", [])[:10]
    ]
    base = symbol.split("/")[0].upper()
    score = max(min((fng - 50.0) / 50.0, 1.0), -1.0) * 0.35
    score += max(min(mkt_chg / 4.0, 1.0), -1.0) * 0.45
    if base in trending:
        score += 0.2
    return max(min(score, 1.0), -1.0)
