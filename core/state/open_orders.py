from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


OPEN_ORDERS_PATH = Path("logs/open_orders.json")
_LOGGER = logging.getLogger(__name__)


def load_open_orders() -> dict[str, Any]:
    if not OPEN_ORDERS_PATH.exists():
        return {}
    try:
        return json.loads(OPEN_ORDERS_PATH.read_text())
    except Exception:
        return {}


def save_open_orders(data: dict[str, Any]) -> None:
    OPEN_ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OPEN_ORDERS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(OPEN_ORDERS_PATH)


def upsert_open_order(client_order_id: str, payload: dict) -> None:
    data = load_open_orders()
    payload.setdefault("last_update_ts", time.time())
    data[client_order_id] = payload
    save_open_orders(data)


def update_status(client_order_id: str, status: str) -> None:
    data = load_open_orders()
    if client_order_id in data:
        # idempotency: ignore same status
        if data[client_order_id].get("status") == status:
            return
        data[client_order_id]["status"] = status
        data[client_order_id]["last_update_ts"] = time.time()
        if status in {"filled", "canceled", "rejected", "error"}:
            data[client_order_id]["final_ts"] = time.time()
        save_open_orders(data)


def has_active_order_for_symbol(symbol: str) -> bool:
    target = str(symbol).upper()
    for record in load_open_orders().values():
        if str(record.get("symbol", "")).upper() != target:
            continue
        status = str(record.get("status", "")).lower()
        if status not in {"filled", "canceled", "rejected", "error", "timeout"}:
            return True
    return False


def reconcile_with_positions(positions: dict[str, float]) -> None:
    data = load_open_orders()
    changed = False
    for order_id, record in data.items():
        status = str(record.get("status", "")).lower()
        if status in {"filled", "canceled", "rejected", "error", "timeout"}:
            continue
        symbol = str(record.get("symbol", "")).upper()
        side = str(record.get("side", "")).upper()
        qty = float(positions.get(symbol, 0.0) or 0.0)
        if side == "BUY" and qty > 0.0:
            record["status"] = "filled"
            record["last_update_ts"] = time.time()
            changed = True
        elif side == "SELL" and qty <= 0.0:
            record["status"] = "filled"
            record["last_update_ts"] = time.time()
            changed = True
    if changed:
        save_open_orders(data)


def find_stale_orders(timeout_sec: int) -> list[str]:
    data = load_open_orders()
    stale = []
    now = time.time()
    for coid, rec in data.items():
        status = rec.get("status")
        if status in {"filled", "canceled", "rejected", "error"}:
            continue
        ts = rec.get("submitted_ts")
        try:
            # submitted_ts is ISO; fall back to epoch if present
            if isinstance(ts, (int, float)):
                age = now - float(ts)
            else:
                age = now - _iso_to_epoch(ts)
            if age >= timeout_sec:
                stale.append(coid)
        except Exception as exc:
            _LOGGER.warning("failed to parse submitted_ts for open order %s: %s", coid, exc)
            continue
    return stale


def mark_timeout(client_order_id: str) -> None:
    data = load_open_orders()
    if client_order_id in data:
        data[client_order_id]["status"] = "timeout"
        data[client_order_id]["last_update_ts"] = time.time()
        save_open_orders(data)


def _iso_to_epoch(ts: str) -> float:
    # simple parse for UTC ISO timestamps
    from datetime import datetime

    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
