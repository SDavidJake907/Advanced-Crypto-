from __future__ import annotations

import os
import time
from typing import Any

import httpx

from core.llm.client import (
    advisory_provider_api_url,
    advisory_provider_name,
    nemotron_provider_api_url,
    nemotron_provider_name,
)
from core.runtime.alerts import emit_alert


def check_phi3_health() -> dict[str, Any]:
    advisory_provider = advisory_provider_name()
    if advisory_provider == "local_nemo":
        base_url = advisory_provider_api_url() or "http://localhost:11434"
        url = f"{base_url.rstrip('/')}/api/tags"
        try:
            response = httpx.get(url, timeout=3.0)
            return {"ok": response.status_code == 200, "status_code": response.status_code, "provider": "local_nemo"}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "provider": "local_nemo"}
    port = os.getenv("PHI3_PORT", "8084")
    url = f"http://localhost:{port}/health"
    try:
        response = httpx.get(url, timeout=5.0)
        return {"ok": response.status_code == 200, "status_code": response.status_code, "provider": "phi3"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "provider": "phi3"}


def check_nemotron_health() -> dict[str, Any]:
    provider = nemotron_provider_name()
    if provider == "nvidia":
        return {"ok": bool(os.getenv("NEMOTRON_CLOUD_API_KEY", os.getenv("NVIDIA_API_KEY", "")).strip()), "provider": "nvidia"}
    if provider == "openai":
        return {"ok": bool(os.getenv("OPENAI_API_KEY", "").strip()), "provider": "openai"}
    base_url = nemotron_provider_api_url() or "http://localhost:11434"
    url = f"{base_url}/api/tags"
    try:
        response = httpx.get(url, timeout=3.0)
        return {"ok": response.status_code == 200, "status_code": response.status_code, "provider": "local"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "provider": "local"}


def check_all_endpoints() -> dict[str, Any]:
    return {
        "phi3": check_phi3_health(),
        "nemotron": check_nemotron_health(),
        "checked_at": time.time(),
    }


def log_watchdog_status(results: dict[str, Any]) -> None:
    phi3_ok = results.get("phi3", {}).get("ok", False)
    nemotron_ok = results.get("nemotron", {}).get("ok", False)
    if not phi3_ok:
        emit_alert(
            level="error",
            source="watchdog",
            code="advisory_unhealthy",
            message="Advisory endpoint unhealthy",
            details={"endpoint": results.get("phi3", {})},
            dedupe_key="watchdog:advisory_unhealthy",
        )
        print(f"[WATCHDOG] advisory endpoint unhealthy: {results['phi3']}")
    if not nemotron_ok:
        emit_alert(
            level="error",
            source="watchdog",
            code="nemotron_unhealthy",
            message="Nemotron endpoint unhealthy",
            details={"endpoint": results.get("nemotron", {})},
            dedupe_key="watchdog:nemotron_unhealthy",
        )
        print(f"[WATCHDOG] Nemotron endpoint unhealthy: {results['nemotron']}")
