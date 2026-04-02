from __future__ import annotations

import json
import os
import socket
import time
import ctypes
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any


DEFAULT_TRADER_LOCK_PATH = Path("logs/trader_instance.lock")
ERROR_ALREADY_EXISTS = 183


def build_trader_instance_id() -> str:
    configured = os.getenv("TRADER_INSTANCE_ID", "").strip()
    if configured:
        return configured
    return f"trader-{os.getpid()}-{int(time.time())}"


def _read_lock_metadata(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {"raw": raw}
    return payload if isinstance(payload, dict) else {"raw": raw}


@dataclass(slots=True)
class TraderAlreadyRunningError(RuntimeError):
    path: Path
    owner_metadata: dict[str, Any]

    def __str__(self) -> str:
        owner = self.owner_metadata or {}
        owner_id = owner.get("instance_id", "unknown")
        owner_pid = owner.get("pid", "unknown")
        return f"Trader singleton lock already held at {self.path} by instance_id={owner_id} pid={owner_pid}"


class TraderSingletonLock:
    def __init__(self, path: Path | str = DEFAULT_TRADER_LOCK_PATH, *, instance_id: str | None = None) -> None:
        self.path = Path(path)
        self.instance_id = instance_id or build_trader_instance_id()
        self._handle: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lock_path": str(self.path),
        }

    def acquire(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner_metadata = _read_lock_metadata(self.path)
        mutex_name = f"KrakenSKTrader_{sha1(str(self.path.resolve()).encode('utf-8')).hexdigest()}"
        handle = ctypes.windll.kernel32.CreateMutexW(None, True, mutex_name)
        if not handle:
            raise OSError("Failed to create trader singleton mutex")
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            ctypes.windll.kernel32.CloseHandle(handle)
            raise TraderAlreadyRunningError(path=self.path, owner_metadata=owner_metadata)
        self.path.write_text(json.dumps(self.metadata), encoding="utf-8")
        self._handle = handle
        return self.metadata

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            ctypes.windll.kernel32.ReleaseMutex(self._handle)
        finally:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "TraderSingletonLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
