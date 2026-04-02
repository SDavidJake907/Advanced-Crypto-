from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from core.runtime.log_rotation import rotate_jsonl_if_needed

ALERTS_LOG_PATH = Path(os.getenv("ALERTS_LOG_PATH", "logs/alerts.jsonl"))
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "true").lower() == "true"
ALERT_THROTTLE_SEC = float(os.getenv("ALERT_THROTTLE_SEC", "120"))

_LAST_ALERT_TS: dict[str, float] = {}


def emit_alert(
    *,
    level: str,
    source: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    throttle_sec: float | None = None,
) -> bool:
    if not ALERTS_ENABLED:
        return False
    now = time.time()
    key = dedupe_key or f"{source}:{code}"
    throttle = ALERT_THROTTLE_SEC if throttle_sec is None else float(throttle_sec)
    last_ts = _LAST_ALERT_TS.get(key, 0.0)
    if throttle > 0.0 and now - last_ts < throttle:
        return False
    _LAST_ALERT_TS[key] = now
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "level": str(level or "error").lower(),
        "source": str(source or "unknown"),
        "code": str(code or "runtime_alert"),
        "message": str(message or "").strip(),
        "details": details or {},
    }
    ALERTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(ALERTS_LOG_PATH)
    with ALERTS_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    return True
