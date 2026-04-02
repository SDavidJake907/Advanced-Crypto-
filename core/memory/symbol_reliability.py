"""Per-symbol reliability map — win rate, avg PnL, avg hold time.

Reads from TradeMemoryStore outcomes and returns a cached map that
final_score.py uses to add a reliability_bonus to each candidate.

Cache TTL defaults to 300 seconds so the map refreshes every 5 minutes
without re-reading the file on every single cycle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from core.memory.trade_memory import TradeMemoryStore


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SymbolReliabilityRecord:
    symbol: str
    win_rate: float
    avg_pnl_pct: float
    avg_hold_min: float
    trade_count: int
    verdict: str        # "strong" / "neutral" / "weak" / "avoid"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: dict[str, SymbolReliabilityRecord] = {}
_cache_ts: float = 0.0
_cache_ttl: float = 300.0   # seconds


def _build_map(store: TradeMemoryStore, lookback: int) -> dict[str, SymbolReliabilityRecord]:
    """Read all outcomes, group by symbol, compute rolling stats."""
    all_outcomes = store.load_outcomes()
    if not all_outcomes:
        return {}

    # Group by symbol — keep last `lookback` per symbol
    by_symbol: dict[str, list] = {}
    for o in all_outcomes:
        by_symbol.setdefault(o.symbol, []).append(o)

    result: dict[str, SymbolReliabilityRecord] = {}
    for symbol, outcomes in by_symbol.items():
        recent = outcomes[-lookback:]
        count = len(recent)
        if count == 0:
            continue

        wins = [o for o in recent if o.pnl_pct > 0.001]
        win_rate = len(wins) / count
        avg_pnl = sum(o.pnl_pct for o in recent) / count
        avg_hold = sum(o.hold_minutes for o in recent) / count

        if win_rate >= 0.65 and count >= 5:
            verdict = "strong"
        elif win_rate < 0.30 and count >= 4:
            verdict = "avoid"
        elif win_rate < 0.40 and count >= 3:
            verdict = "weak"
        else:
            verdict = "neutral"

        result[symbol] = SymbolReliabilityRecord(
            symbol=symbol,
            win_rate=round(win_rate, 3),
            avg_pnl_pct=round(avg_pnl, 4),
            avg_hold_min=round(avg_hold, 1),
            trade_count=count,
            verdict=verdict,
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_reliability_map(
    *,
    store: TradeMemoryStore | None = None,
    lookback: int = 20,
    ttl_sec: float = _cache_ttl,
    base_dir: Path | None = None,
) -> dict[str, SymbolReliabilityRecord]:
    """Return a cached per-symbol reliability map.

    The map is rebuilt from disk at most once per `ttl_sec` seconds.
    Returns an empty dict if no outcome data exists yet.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    if _cache and (now - _cache_ts) < ttl_sec:
        return _cache

    if store is None:
        store = TradeMemoryStore(base_dir) if base_dir else TradeMemoryStore()

    _cache = _build_map(store, lookback)
    _cache_ts = now
    return _cache


def reliability_map_as_dict(
    map_: dict[str, SymbolReliabilityRecord],
) -> dict[str, dict[str, Any]]:
    """Convert to the plain-dict format expected by final_score.compute_final_score."""
    return {sym: rec.to_dict() for sym, rec in map_.items()}


def get_symbol_record(
    symbol: str,
    *,
    store: TradeMemoryStore | None = None,
    lookback: int = 20,
) -> SymbolReliabilityRecord | None:
    """Quick lookup for a single symbol — uses the same cache."""
    return load_reliability_map(store=store, lookback=lookback).get(symbol)
