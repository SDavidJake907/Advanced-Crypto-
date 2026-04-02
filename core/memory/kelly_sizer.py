"""Kelly Criterion dynamic position sizer.

Reads Kraken trade history, FIFO-matches buy/sell round-trips per pair,
computes rolling half-Kelly fraction, and writes results to runtime_overrides.json.

Rolling window: last KELLY_ROLLING_N completed round-trips (default 50).
Output clamped: [KELLY_MIN_PCT, KELLY_MAX_PCT] (default 10% – 30%).
Applies half-Kelly for safety (f* / 2).
"""
from __future__ import annotations

import csv
import io
import os
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

KELLY_ENABLED: bool = os.getenv("KELLY_ENABLED", "true").lower() == "true"
KELLY_ROLLING_N: int = 50
KELLY_MIN_PCT: float = float(os.getenv("KELLY_MIN_PCT", "10.0"))
KELLY_MAX_PCT: float = float(os.getenv("KELLY_MAX_PCT", "30.0"))

# Pairs treated as meme/L4 for separate sizing
_DEFAULT_MEME_PAIRS = {
    "FARTCOIN/USD", "BONK/USD", "PEPE/USD", "FLOKI/USD", "SHIB/USD",
    "DOGE/USD", "WIF/USD", "MOODENG/USD", "POPCAT/USD",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rows_from_zip(zip_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            if not name:
                return rows
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    rows.append(dict(row))
    except Exception:
        pass
    return rows


def _load_all_rows(zip_path: str) -> list[dict[str, str]]:
    """Load the primary zip plus any extras found in the same directory."""
    primary = Path(zip_path)
    rows: list[dict[str, str]] = []
    seen_txids: set[str] = set()

    # Scan sibling zips (same dir, name contains 'kraken' and 'spot' or 'trades')
    search_dir = primary.parent
    candidates: list[Path] = sorted(search_dir.glob("kraken*.zip"))
    # Always include the primary first so its rows are preferred if duplicate
    if primary in candidates:
        candidates.remove(primary)
    candidates = [primary] + candidates

    for path in candidates:
        for row in _load_rows_from_zip(str(path)):
            txid = row.get("txid", "")
            if txid and txid not in seen_txids:
                seen_txids.add(txid)
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Round-trip matching (FIFO per pair)
# ---------------------------------------------------------------------------

def _fifo_match(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """FIFO-match buys and sells within each pair; return completed round-trips."""
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pair[row.get("pair", "")].append(row)

    round_trips: list[dict[str, Any]] = []
    for pair, trades in by_pair.items():
        trades.sort(key=lambda t: t.get("time", ""))
        buy_queue: deque[tuple[float, float, str]] = deque()  # (cost_usd, fee, time)
        for t in trades:
            try:
                cost = float(t.get("costusd") or t.get("cost") or 0)
                fee = float(t.get("fee") or 0)
            except (ValueError, TypeError):
                continue
            trade_type = t.get("type", "")
            trade_time = t.get("time", "")
            if trade_type == "buy":
                buy_queue.append((cost, fee, trade_time))
            elif trade_type == "sell" and buy_queue:
                buy_cost, buy_fee, buy_time = buy_queue.popleft()
                total_fee = buy_fee + fee
                pnl = cost - buy_cost - total_fee
                round_trips.append({
                    "pair": pair,
                    "buy_cost": buy_cost,
                    "sell_cost": cost,
                    "fee": total_fee,
                    "pnl": pnl,
                    "buy_time": buy_time,
                    "sell_time": trade_time,
                })

    round_trips.sort(key=lambda r: r.get("sell_time", ""))
    return round_trips


# ---------------------------------------------------------------------------
# Kelly calculation
# ---------------------------------------------------------------------------

def _kelly_from_trips(trips: list[dict[str, Any]], n: int = KELLY_ROLLING_N) -> float:
    """Half-Kelly fraction from last *n* round-trips. Returns pct (e.g. 15.0)."""
    recent = trips[-n:] if len(trips) >= n else trips
    if len(recent) < 5:
        return 0.0  # Not enough history; caller will use current setting

    wins = [t["pnl"] for t in recent if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in recent if t["pnl"] <= 0]

    if not wins or not losses:
        return 0.0

    W = len(wins) / len(recent)
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return 0.0

    R = avg_win / avg_loss
    kelly_f = W - (1 - W) / R  # Full Kelly fraction
    half_kelly = kelly_f / 2.0

    # Convert to percentage, clamp
    pct = half_kelly * 100.0
    return max(KELLY_MIN_PCT, min(KELLY_MAX_PCT, round(pct, 1)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_kelly_sizes(zip_path: str, meme_pairs: set[str] | None = None) -> dict[str, float]:
    """Compute half-Kelly position sizes for standard and meme trades.

    Returns dict with keys:
        ``EXEC_RISK_PER_TRADE_PCT``     – standard half-Kelly pct
        ``MEME_EXEC_RISK_PER_TRADE_PCT`` – meme half-Kelly pct
        ``kelly_standard_raw``           – raw full-Kelly before halving (diagnostic)
        ``kelly_meme_raw``               – raw full-Kelly before halving (diagnostic)
        ``standard_trips``               – number of standard round-trips used
        ``meme_trips``                   – number of meme round-trips used
        ``standard_win_rate``            – win rate for standard trades
        ``meme_win_rate``                – win rate for meme trades
    """
    meme_pairs = meme_pairs or _DEFAULT_MEME_PAIRS
    meme_pairs_upper = {p.upper() for p in meme_pairs}

    all_rows = _load_all_rows(zip_path)
    all_trips = _fifo_match(all_rows)

    standard_trips = [t for t in all_trips if t["pair"].upper() not in meme_pairs_upper]
    meme_trips = [t for t in all_trips if t["pair"].upper() in meme_pairs_upper]

    std_pct = _kelly_from_trips(standard_trips)
    meme_pct = _kelly_from_trips(meme_trips)

    # Diagnostics
    def _win_rate(trips: list[dict[str, Any]]) -> float:
        n = len(trips[-KELLY_ROLLING_N:])
        if not n:
            return 0.0
        wins = sum(1 for t in trips[-KELLY_ROLLING_N:] if t["pnl"] > 0)
        return round(wins / n, 3)

    return {
        "EXEC_RISK_PER_TRADE_PCT": std_pct if std_pct > 0 else 0.0,
        "MEME_EXEC_RISK_PER_TRADE_PCT": meme_pct if meme_pct > 0 else 0.0,
        "standard_trips": len(standard_trips),
        "meme_trips": len(meme_trips),
        "standard_win_rate": _win_rate(standard_trips),
        "meme_win_rate": _win_rate(meme_trips),
    }


def run_kelly_update(zip_path: str) -> dict[str, Any]:
    """Compute Kelly sizes and write to runtime_overrides.json.

    Returns a summary dict suitable for logging.
    """
    from core.config.runtime import update_runtime_overrides

    if not KELLY_ENABLED:
        return {"ts": datetime.now(timezone.utc).isoformat(), "source": "kelly_sizer", "applied": {}, "diagnostics": {"disabled": True}}

    result = compute_kelly_sizes(zip_path)

    updates: dict[str, float] = {}
    if result["EXEC_RISK_PER_TRADE_PCT"] > 0:
        updates["EXEC_RISK_PER_TRADE_PCT"] = result["EXEC_RISK_PER_TRADE_PCT"]
    if result["MEME_EXEC_RISK_PER_TRADE_PCT"] > 0:
        updates["MEME_EXEC_RISK_PER_TRADE_PCT"] = result["MEME_EXEC_RISK_PER_TRADE_PCT"]

    applied: dict[str, float] = {}
    if updates:
        update_runtime_overrides(updates)
        applied = updates

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "kelly_sizer",
        "applied": applied,
        "diagnostics": {
            "standard_trips": result["standard_trips"],
            "standard_win_rate": result["standard_win_rate"],
            "meme_trips": result["meme_trips"],
            "meme_win_rate": result["meme_win_rate"],
        },
    }
    return summary


if __name__ == "__main__":
    _zip = os.getenv("KRAKEN_TRADE_HISTORY_ZIP", "")
    if not _zip:
        print("[kelly_sizer] KRAKEN_TRADE_HISTORY_ZIP not set — nothing to do.")
    else:
        _summary = run_kelly_update(_zip)
        print(f"[kelly_sizer] {_summary}")
