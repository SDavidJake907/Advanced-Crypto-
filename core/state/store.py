import json
import os
import time
from pathlib import Path

from core.symbols import normalize_symbol

UNIVERSE_PATH = Path("universe.json")
ACCOUNT_SYNC_PATH = Path("logs/account_sync.json")
RELOAD_SIGNAL_PATH = Path("logs/reload.signal")


def load_universe() -> dict:
    if UNIVERSE_PATH.exists():
        try:
            payload = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"active_pairs": [], "updated_at": 0, "reason": "invalid", "meta": {}}
        return payload if isinstance(payload, dict) else {"active_pairs": [], "updated_at": 0, "reason": "invalid", "meta": {}}
    return {"active_pairs": [], "updated_at": 0, "reason": "missing", "meta": {}}


def save_universe(active_pairs: list[str], reason: str, meta: dict | None = None) -> dict:
    normalized_pairs = [normalize_symbol(pair) for pair in active_pairs]
    payload = {
        "active_pairs": list(dict.fromkeys(normalized_pairs)),
        "updated_at": int(time.time()),
        "reason": reason,
        "meta": meta or {},
    }
    tmp_path = UNIVERSE_PATH.with_name(f"{UNIVERSE_PATH.stem}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for _ in range(5):
        try:
            tmp_path.replace(UNIVERSE_PATH)
            break
        except PermissionError:
            time.sleep(0.05)
    else:
        UNIVERSE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.unlink(missing_ok=True)
    return payload


def load_synced_position_symbols() -> list[str]:
    if not ACCOUNT_SYNC_PATH.exists():
        return []
    try:
        payload = json.loads(ACCOUNT_SYNC_PATH.read_text())
    except Exception:
        return []
    portfolio_state = payload.get("portfolio_state", {})
    positions = portfolio_state.get("positions", {}) if isinstance(portfolio_state, dict) else {}
    if not isinstance(positions, dict):
        return []
    normalized = [
        normalize_symbol(symbol)
        for symbol, qty in positions.items()
        if str(symbol).strip() and float(qty or 0.0) > 0.0
    ]
    return list(dict.fromkeys(normalized))


def load_sector_tags() -> dict[str, str]:
    universe = load_universe()
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    tags = meta.get("sector_tags", {})
    if not isinstance(tags, dict):
        return {}
    normalized: dict[str, str] = {}
    for symbol, sector in tags.items():
        name = normalize_symbol(str(symbol))
        label = str(sector).strip().lower()
        if name and label:
            normalized[name] = label
    return normalized


def read_reload_signal() -> str | None:
    if not RELOAD_SIGNAL_PATH.exists():
        return None
    try:
        return RELOAD_SIGNAL_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return None
