import json
import time
from pathlib import Path


def load_age_cache(path: str, ttl_hours: int) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        now = int(time.time())
        fresh: dict = {}
        for k, v in data.items():
            checked_at = int(v.get("checked_at", 0))
            if now - checked_at <= ttl_hours * 3600:
                fresh[k] = v
        return fresh
    except Exception:
        return {}


def save_age_cache(path: str, cache: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True))
