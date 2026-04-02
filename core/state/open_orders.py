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
    payload.setdefault("retry_count", 0)
    payload.setdefault("reprice_count", 0)
    payload.setdefault("order_role", "entry")
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


def reconcile_open_orders(
    *,
    client: Any | None = None,
    symbol: str | None = None,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    data = load_open_orders()
    if not data:
        return {}

    changed = False
    now = time.time()
    target_symbol = str(symbol or "").upper().strip()
    queryable_txids: list[str] = []
    order_ids_by_txid: dict[str, list[str]] = {}

    for order_id, record in data.items():
        record_symbol = str(record.get("symbol", "")).upper()
        if target_symbol and record_symbol != target_symbol:
            continue
        status = str(record.get("status", "")).lower()
        if status in {"filled", "canceled", "cancelled", "rejected", "error", "timeout", "expired"}:
            continue
        submitted_ts = _record_epoch(record.get("submitted_ts"))
        record_timeout_sec = int(record.get("stale_ttl_sec", timeout_sec) or timeout_sec)
        if submitted_ts is not None and (now - submitted_ts) >= record_timeout_sec:
            record["timeout_age_sec"] = now - submitted_ts
            txid = str(record.get("txid", "")).strip()
            if client is not None and txid:
                try:
                    client.cancel_order(txid)
                    _LOGGER.info("cancelled stale order %s (txid=%s)", order_id, txid)
                except Exception as exc:
                    _LOGGER.warning("failed to cancel stale order %s: %s", order_id, exc)
            record["status"] = "timeout"
            record["last_update_ts"] = now
            record["final_ts"] = now
            record["reprice_count"] = int(record.get("reprice_count", 0) or 0) + 1
            changed = True
            continue
        txid = str(record.get("txid", "")).strip()
        if txid:
            queryable_txids.append(txid)
            order_ids_by_txid.setdefault(txid, []).append(order_id)

    if client is not None and queryable_txids:
        open_txids: set[str] = set()
        if hasattr(client, "get_open_orders"):
            try:
                open_payload = client.get_open_orders().get("result", {})
                open_entries = open_payload.get("open", {}) if isinstance(open_payload, dict) else {}
                if isinstance(open_entries, dict):
                    open_txids = {str(txid).strip() for txid in open_entries.keys() if str(txid).strip()}
            except Exception as exc:
                _LOGGER.warning("failed to query Kraken open orders: %s", exc)

        for txid in open_txids:
            for order_id in order_ids_by_txid.get(txid, []):
                record = data.get(order_id)
                if not isinstance(record, dict):
                    continue
                current_status = str(record.get("status", "")).lower()
                if current_status != "open":
                    record["status"] = "open"
                    record["last_update_ts"] = now
                    changed = True

        txids_needing_detail = [txid for txid in queryable_txids if txid not in open_txids]
        if not txids_needing_detail:
            if changed:
                save_open_orders(data)
            return data

        try:
            payload = client.query_orders_info(txids_needing_detail).get("result", {})
        except Exception as exc:
            message = str(exc)
            if "EAPI:Invalid key" in message:
                _LOGGER.info("skipping Kraken QueryOrders reconcile because the current key cannot access that endpoint")
            else:
                _LOGGER.warning("failed to query Kraken order status: %s", exc)
            payload = {}
        for txid, order_payload in payload.items():
            if not isinstance(order_payload, dict):
                continue
            kraken_status = str(order_payload.get("status", "")).lower()
            normalized = _normalize_order_status(kraken_status)
            if normalized is None:
                continue
            for order_id in order_ids_by_txid.get(txid, []):
                record = data.get(order_id)
                if not isinstance(record, dict):
                    continue
                current_status = str(record.get("status", "")).lower()
                if current_status != normalized:
                    record["status"] = normalized
                    record["last_update_ts"] = now
                    if normalized in {"filled", "canceled", "rejected", "error", "timeout", "expired"}:
                        record["final_ts"] = now
                    changed = True

    if changed:
        save_open_orders(data)
    return data


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


def has_recent_buy_fill_for_symbol(symbol: str, cooldown_sec: float = 600.0) -> bool:
    """Return True if there is a filled BUY order for this symbol within the last cooldown_sec.

    Used to block rapid duplicate entries when a limit order fills but the in-memory
    portfolio/positions state hasn't synced yet from the exchange.
    """
    target = str(symbol).upper()
    now = time.time()
    for record in load_open_orders().values():
        if str(record.get("symbol", "")).upper() != target:
            continue
        if str(record.get("side", "")).upper() != "BUY":
            continue
        if str(record.get("status", "")).lower() != "filled":
            continue
        final_ts = _record_epoch(record.get("final_ts") or record.get("last_update_ts"))
        if final_ts is not None and (now - final_ts) < cooldown_sec:
            return True
    return False


def get_pending_buy_capital_usd() -> float:
    """Return total USD committed to open (unfilled) BUY limit orders."""
    data = load_open_orders()
    total = 0.0
    for record in data.values():
        status = str(record.get("status", "")).lower()
        if status in {"filled", "canceled", "cancelled", "rejected", "error", "timeout", "expired"}:
            continue
        if str(record.get("side", "")).upper() != "BUY":
            continue
        qty = float(record.get("qty", 0.0) or 0.0)
        price = float(record.get("limit_price", 0.0) or 0.0)
        if qty > 0.0 and price > 0.0:
            total += qty * price
    return total


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
            epoch = _record_epoch(ts)
            if epoch is None:
                continue
            age = now - epoch
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


def _record_epoch(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str) and ts.strip():
        return _iso_to_epoch(ts)
    return None


def _normalize_order_status(status: str) -> str | None:
    lowered = str(status or "").strip().lower()
    if lowered in {"open", "pending"}:
        return "open"
    if lowered in {"closed", "filled"}:
        return "filled"
    if lowered in {"canceled", "cancelled"}:
        return "canceled"
    if lowered in {"expired"}:
        return "expired"
    return None
