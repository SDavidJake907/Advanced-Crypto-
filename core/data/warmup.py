from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from core.runtime.log_rotation import rotate_jsonl_if_needed

@dataclass
class WarmupInfo:
    ready: bool
    just_completed: bool
    symbols_ready: list[str]
    symbols_pending: list[str]
    timeframe_progress: dict[str, dict[str, int]]
    hard_timeframes: list[str]
    soft_timeframes: list[str]


def get_min_bars_1m() -> int:
    return int(os.getenv("WARMUP_MIN_BARS_1M", "20"))


def get_min_bars_1h() -> int:
    return int(os.getenv("WARMUP_MIN_BARS_1H", "8"))


def get_min_bars_5m() -> int:
    return int(os.getenv("WARMUP_MIN_BARS_5M", "6"))


def get_min_bars_15m() -> int:
    return int(os.getenv("WARMUP_MIN_BARS_15M", "4"))


def get_min_bars_7d() -> int:
    return int(os.getenv("WARMUP_MIN_BARS_7D", "6"))


def get_min_bars_30d() -> int:
    return int(os.getenv("WARMUP_MIN_BARS_30D", "4"))


def _minimums() -> dict[str, int]:
    return {
        "1m": get_min_bars_1m(),
        "5m": get_min_bars_5m(),
        "15m": get_min_bars_15m(),
        "1h": get_min_bars_1h(),
        "7d": get_min_bars_7d(),
        "30d": get_min_bars_30d(),
    }


def _hard_timeframes() -> list[str]:
    raw = os.getenv("WARMUP_HARD_TIMEFRAMES", "1m")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["1m"]


def _build_bar_counts(live_buffer: Any) -> dict[str, dict[str, int]]:
    symbols = getattr(live_buffer, "symbols", [])
    return {
        symbol: {
            "1m": int(live_buffer.get_bar_count(symbol, "1m")),
            "5m": int(live_buffer.get_bar_count(symbol, "5m")),
            "15m": int(live_buffer.get_bar_count(symbol, "15m")),
            "1h": int(live_buffer.get_bar_count(symbol, "1h")),
            "7d": int(live_buffer.get_bar_count(symbol, "7d")),
            "30d": int(live_buffer.get_bar_count(symbol, "30d")),
        }
        for symbol in symbols
    }


def check_status(live_buffer: Any, *, was_ready: bool = False) -> WarmupInfo:
    bar_counts = _build_bar_counts(live_buffer)
    minimums = _minimums()
    hard_timeframes = _hard_timeframes()
    soft_timeframes = [timeframe for timeframe in minimums if timeframe not in hard_timeframes]
    pending_symbols: list[str] = []

    for symbol, counts in bar_counts.items():
        if any(int(counts.get(timeframe, 0)) < minimums[timeframe] for timeframe in hard_timeframes):
            pending_symbols.append(symbol)

    symbols_ready = [symbol for symbol in bar_counts if symbol not in pending_symbols]
    active_counts = [counts for counts in bar_counts.values() if any(v > 0 for v in counts.values())]
    timeframe_progress = {
        timeframe: {
            "required": required,
            "current": max((int(counts.get(timeframe, 0)) for counts in active_counts), default=0),
            "mode": "hard" if timeframe in hard_timeframes else "soft",
        }
        for timeframe, required in minimums.items()
    }
    ready = bool(symbols_ready)
    return WarmupInfo(
        ready=ready,
        just_completed=ready and not was_ready,
        symbols_ready=symbols_ready,
        symbols_pending=pending_symbols,
        timeframe_progress=timeframe_progress,
        hard_timeframes=hard_timeframes,
        soft_timeframes=soft_timeframes,
    )


def emit_warmup_status(log_path: str | Path, status_dict: dict[str, Any]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(status_dict) + "\n")
