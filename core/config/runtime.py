from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_OVERRIDES_PATH = ROOT / "configs" / "runtime_overrides.json"

_SETTING_SPECS: dict[str, dict[str, Any]] = {
    "MEME_SYMBOLS": {"type": str, "default": ""},
    "TRADER_BASE_MOMO_THRESHOLD": {"type": float, "default": 0.0},
    "TRADER_ATR_VOLATILITY_SCALE": {"type": float, "default": 0.5},
    "TRADER_RSI_OVERBOUGHT": {"type": float, "default": 70.0},
    "TRADER_RSI_OVERSOLD": {"type": float, "default": 30.0},
    "MEME_BASE_MOMO_THRESHOLD": {"type": float, "default": -0.002},
    "MEME_ATR_VOLATILITY_SCALE": {"type": float, "default": 0.02},
    "MEME_RSI_OVERBOUGHT": {"type": float, "default": 85.0},
    "MEME_RSI_OVERSOLD": {"type": float, "default": 20.0},
    "TRADER_BB_BREAKOUT_ZSCORE": {"type": float, "default": 1.0},
    "TRADER_SMOOTHING_BARS": {"type": int, "default": 2},
    "TRADER_COOLDOWN_BARS": {"type": int, "default": 0},
    "TRADER_PROPOSED_WEIGHT": {"type": float, "default": 0.1},
    "MEME_PROPOSED_WEIGHT": {"type": float, "default": 0.03},
    "MEME_TREND_CONFLICT_SCALE": {"type": float, "default": 0.5},
    "MEME_LANE_MIN_VOLUME_RATIO": {"type": float, "default": 1.0},
    "MEME_LANE_MIN_VOLUME_SURGE": {"type": float, "default": 0.25},
    "MEME_LANE_SOFT_MAX_SPREAD_PCT": {"type": float, "default": 4.0},
    "MEME_ENTRY_SCORE_BUY_THRESHOLD": {"type": float, "default": 48.0},
    "MEME_ENTRY_SCORE_STRONG_BUY_THRESHOLD": {"type": float, "default": 62.0},
    "EXIT_ATR_STOP_MULT": {"type": float, "default": 1.5},
    "EXIT_ATR_TAKE_PROFIT_MULT": {"type": float, "default": 3.0},
    "EXIT_BREAK_EVEN_R": {"type": float, "default": 1.0},
    "EXIT_TRAIL_ARM_R": {"type": float, "default": 1.5},
    "EXIT_TRAIL_ATR_MULT": {"type": float, "default": 1.0},
    "EXIT_POSTURE_USE_PHI3": {"type": bool, "default": True},
    "EXIT_STALE_MIN_HOLD_MIN": {"type": float, "default": 180.0},
    "EXIT_STALE_MAX_ABS_PNL_PCT": {"type": float, "default": 1.5},
    "EXIT_TIGHTEN_MIN_PNL_PCT": {"type": float, "default": 1.0},
    "EXIT_POSTURE_NEG_PNL_PCT": {"type": float, "default": -2.0},
    "EXIT_POSTURE_TIGHTEN_STOP_PCT": {"type": float, "default": 0.0075},
    "MEME_EXIT_ATR_STOP_MULT": {"type": float, "default": 1.2},
    "MEME_EXIT_ATR_TAKE_PROFIT_MULT": {"type": float, "default": 2.5},
    "MEME_BREAK_EVEN_R": {"type": float, "default": 0.8},
    "MEME_TRAIL_ARM_R": {"type": float, "default": 1.2},
    "MEME_TRAIL_ATR_MULT": {"type": float, "default": 0.8},
    "MEME_EXIT_POSTURE_TIGHTEN_STOP_PCT": {"type": float, "default": 0.005},
    "ORDER_PREFERENCE": {"type": str, "default": "auto"},
    "ORDER_POST_ONLY": {"type": bool, "default": True},
    "ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 5.0},
    "MEME_ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 8.0},
    "EXEC_MAX_SPREAD_PCT": {"type": float, "default": 1.0},
    "MEME_EXEC_MAX_SPREAD_PCT": {"type": float, "default": 1.8},
    "EXEC_MIN_NOTIONAL_USD": {"type": float, "default": 10.0},
    "MEME_EXEC_MIN_NOTIONAL_USD": {"type": float, "default": 10.0},
    "EXEC_RISK_PER_TRADE_PCT": {"type": float, "default": 1.0},
    "MEME_EXEC_RISK_PER_TRADE_PCT": {"type": float, "default": 1.0},
    "EXEC_MAKER_FEE_PCT": {"type": float, "default": 0.23},
    "EXEC_TAKER_FEE_PCT": {"type": float, "default": 0.40},
    "EXEC_SLIPPAGE_ATR_MULT": {"type": float, "default": 0.10},
    "MEME_EXEC_SLIPPAGE_ATR_MULT": {"type": float, "default": 0.14},
    "RISK_MAX_POSITION_NOTIONAL": {"type": float, "default": 2000.0},
    "RISK_MAX_LEVERAGE": {"type": float, "default": 2.0},
    "PORTFOLIO_MAX_WEIGHT_PER_SYMBOL": {"type": float, "default": 0.2},
    "PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE": {"type": float, "default": 1.0},
    "PORTFOLIO_MAX_OPEN_POSITIONS": {"type": int, "default": 5},
    "PORTFOLIO_CORR_THRESHOLD": {"type": float, "default": 0.8},
    "PORTFOLIO_CORR_SCALE_DOWN": {"type": float, "default": 0.5},
    "PORTFOLIO_MAX_HIGH_CORR_SAME_DIRECTION": {"type": int, "default": 1},
    "PORTFOLIO_AVG_CORR_SCALE_THRESHOLD": {"type": float, "default": 0.7},
    "PORTFOLIO_AVG_CORR_SCALE_DOWN": {"type": float, "default": 0.6},
    "PORTFOLIO_MAX_POSITIONS_PER_SECTOR": {"type": int, "default": 2},
    "MEME_MAX_OPEN_POSITIONS": {"type": int, "default": 2},
    "NEMOTRON_TOP_CANDIDATE_COUNT": {"type": int, "default": 15},
    "NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP": {"type": bool, "default": True},
    "NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT": {"type": bool, "default": True},
    "NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP": {"type": bool, "default": True},
    "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE": {"type": float, "default": 42.0},
    "NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO": {"type": float, "default": 1.05},
    "NEMOTRON_GATE_MIN_ENTRY_SCORE": {"type": float, "default": 48.0},
    "NEMOTRON_GATE_MIN_VOLUME_RATIO": {"type": float, "default": 1.0},
    "MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE": {"type": float, "default": 38.0},
    "MEME_NEMOTRON_WATCH_MIN_VOLUME_RATIO": {"type": float, "default": 1.0},
    "MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE": {"type": float, "default": 42.0},
    "MEME_NEMOTRON_GATE_MIN_VOLUME_RATIO": {"type": float, "default": 0.95},
    "NEMOTRON_VERDICT_CACHE_TTL_SEC": {"type": float, "default": 90.0},
    "ADVISORY_ENABLE_VISUAL_CONTEXT": {"type": bool, "default": False},
    "ADVISORY_MIN_ENTRY_SCORE": {"type": float, "default": 60.0},
    "ADVISORY_MIN_VOLUME_RATIO": {"type": float, "default": 1.1},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_value(name: str, value: Any) -> Any:
    if name not in _SETTING_SPECS:
        raise KeyError(f"Unknown runtime setting: {name}")
    target_type = _SETTING_SPECS[name]["type"]
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return target_type(value)


def load_runtime_overrides() -> dict[str, Any]:
    if not RUNTIME_OVERRIDES_PATH.exists():
        return {}
    payload = json.loads(RUNTIME_OVERRIDES_PATH.read_text(encoding="utf-8"))
    overrides = payload.get("overrides", payload)
    if not isinstance(overrides, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _SETTING_SPECS:
            sanitized[key] = _coerce_value(key, value)
    return sanitized


def save_runtime_overrides(overrides: dict[str, Any]) -> None:
    RUNTIME_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now_iso(),
        "overrides": {key: _coerce_value(key, value) for key, value in overrides.items()},
    }
    RUNTIME_OVERRIDES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_runtime_overrides(updates: dict[str, Any]) -> dict[str, Any]:
    current = load_runtime_overrides()
    for key, value in updates.items():
        current[key] = _coerce_value(key, value)
    save_runtime_overrides(current)
    return current


def get_runtime_setting(name: str, default: Any | None = None) -> Any:
    if name not in _SETTING_SPECS:
        raise KeyError(f"Unknown runtime setting: {name}")
    overrides = load_runtime_overrides()
    if name in overrides:
        return overrides[name]
    if name in os.environ:
        return _coerce_value(name, os.environ[name])
    if default is not None:
        return default
    return _SETTING_SPECS[name]["default"]


def get_runtime_snapshot() -> dict[str, Any]:
    overrides = load_runtime_overrides()
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, spec in _SETTING_SPECS.items():
        if name in overrides:
            values[name] = overrides[name]
            sources[name] = "override"
        elif name in os.environ:
            values[name] = _coerce_value(name, os.environ[name])
            sources[name] = "env"
        else:
            values[name] = spec["default"]
            sources[name] = "default"
    return {
        "path": str(RUNTIME_OVERRIDES_PATH.relative_to(ROOT)),
        "values": values,
        "sources": sources,
        "overrides": overrides,
    }


def get_cooldown_bars() -> int:
    return int(get_runtime_setting("TRADER_COOLDOWN_BARS"))


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def is_meme_symbol(symbol: str) -> bool:
    raw = str(get_runtime_setting("MEME_SYMBOLS"))
    meme_symbols = {_normalize_symbol(item) for item in raw.split(",") if item.strip()}
    return _normalize_symbol(symbol) in meme_symbols


def get_symbol_lane(symbol: str) -> str:
    return "L4" if is_meme_symbol(symbol) else "L3"


def is_meme_lane(lane: str | None) -> bool:
    return str(lane or "").upper() == "L4"


def get_proposed_weight(symbol: str | None = None, lane: str | None = None) -> float:
    resolved_lane = lane or (get_symbol_lane(symbol) if symbol is not None else None)
    if symbol is not None and is_meme_lane(resolved_lane):
        return float(get_runtime_setting("MEME_PROPOSED_WEIGHT"))
    return float(get_runtime_setting("TRADER_PROPOSED_WEIGHT"))
