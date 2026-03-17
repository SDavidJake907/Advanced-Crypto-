from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_event(stage: str, reason: str, pair: str | None = None, details: dict | None = None) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason": reason,
        "pair": pair,
        "details": details or {},
    }
    Path("logs").mkdir(parents=True, exist_ok=True)
    with Path("logs/order_rejections.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_gatekeeper(snapshot: dict, response: dict) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "response": response,
    }
    Path("logs").mkdir(parents=True, exist_ok=True)
    with Path("logs/gatekeeper_decisions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
