from __future__ import annotations

from typing import Any

from apps.trader.logging_sink import now_iso


def build_synced_account_sync_payload(payload: dict[str, Any]) -> dict[str, Any]:
    ts = now_iso()
    return {
        "ts": ts,
        "status": "synced",
        "last_success_ts": ts,
        **payload,
    }


def build_degraded_account_sync_payload(
    *,
    error: Exception,
    previous_payload: dict[str, Any] | None,
    fallback_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = now_iso()
    previous = previous_payload if isinstance(previous_payload, dict) else {}
    base = previous if previous else (fallback_payload if isinstance(fallback_payload, dict) else {})
    payload = dict(base)
    last_success_ts = str(payload.get("last_success_ts") or payload.get("ts") or "").strip()
    status = "degraded" if last_success_ts else "sync_failed"
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics", {}), dict) else {}
    payload["ts"] = ts
    payload["status"] = status
    payload["error"] = str(error)
    if last_success_ts:
        payload["last_success_ts"] = last_success_ts
    diagnostics = dict(diagnostics)
    diagnostics["last_sync_error"] = str(error)
    diagnostics["last_sync_error_ts"] = ts
    payload["diagnostics"] = diagnostics
    return payload
