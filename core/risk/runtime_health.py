from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_ts(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


@dataclass
class RuntimeHealthDecision:
    ok: bool
    reasons: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_runtime_health(
    *,
    now_ts: float,
    collector_telemetry: dict[str, Any] | None,
    account_sync: dict[str, Any] | None,
    watchdog_results: dict[str, Any] | None,
    telemetry_dir: Path,
    require_account_sync: bool,
    llm_required: bool,
    collector_max_age_sec: float = 45.0,
    account_sync_max_age_sec: float = 90.0,
) -> RuntimeHealthDecision:
    reasons: list[str] = []
    details: dict[str, Any] = {}

    collector_ts = _parse_ts((collector_telemetry or {}).get("ts"))
    collector_age = (now_ts - collector_ts) if collector_ts is not None else None
    details["collector_age_sec"] = collector_age
    if collector_age is None or collector_age > collector_max_age_sec:
        reasons.append("runtime_health_collector_stale")

    account_sync_ts = _parse_ts((account_sync or {}).get("ts"))
    account_sync_age = (now_ts - account_sync_ts) if account_sync_ts is not None else None
    account_sync_status = str((account_sync or {}).get("status", "") or "")
    account_sync_last_success_ts = _parse_ts((account_sync or {}).get("last_success_ts"))
    if account_sync_last_success_ts is None and account_sync_status == "synced":
        account_sync_last_success_ts = account_sync_ts
    account_sync_last_success_age = (
        (now_ts - account_sync_last_success_ts) if account_sync_last_success_ts is not None else None
    )
    details["account_sync_age_sec"] = account_sync_age
    details["account_sync_status"] = account_sync_status
    details["account_sync_last_success_age_sec"] = account_sync_last_success_age
    if require_account_sync:
        if account_sync_status not in {"synced", "degraded"}:
            reasons.append("runtime_health_account_sync_failed")
        elif account_sync_last_success_age is None:
            reasons.append("runtime_health_account_sync_failed")
        elif account_sync_last_success_age > account_sync_max_age_sec:
            reasons.append("runtime_health_account_sync_stale")

    phi3_ok = bool((watchdog_results or {}).get("phi3", {}).get("ok", True))
    nemotron_ok = bool((watchdog_results or {}).get("nemotron", {}).get("ok", True))
    details["phi3_ok"] = phi3_ok
    details["nemotron_ok"] = nemotron_ok
    if llm_required:
        if not phi3_ok:
            reasons.append("runtime_health_phi3_unhealthy")
        if not nemotron_ok:
            reasons.append("runtime_health_nemotron_unhealthy")

    telemetry_ok = telemetry_dir.exists() and telemetry_dir.is_dir()
    try:
        telemetry_ok = telemetry_ok and telemetry_dir.stat().st_mode is not None
    except Exception:
        telemetry_ok = False
    details["telemetry_dir"] = str(telemetry_dir)
    details["telemetry_ok"] = telemetry_ok
    if not telemetry_ok:
        reasons.append("runtime_health_telemetry_unavailable")

    return RuntimeHealthDecision(ok=not reasons, reasons=reasons, details=details)
