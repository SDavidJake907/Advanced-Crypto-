from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from core.data.kraken_rest import KrakenRestClient
from core.symbols import normalize_symbol, to_kraken_symbol


@dataclass
class KrakenPairRule:
    wsname: str
    ordermin: float
    lot_decimals: int
    pair_decimals: int
    costmin: float | None = None
    tick_size: float | None = None


def _load_cache(path: Path, ttl_min: int) -> dict | None:
    if not path.exists():
        return None
    age_s = time.time() - path.stat().st_mtime
    if age_s > ttl_min * 60:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_rules(client: KrakenRestClient) -> dict[str, KrakenPairRule]:
    cache_file = Path(os.getenv("KRAKEN_RULES_CACHE_FILE", "configs/kraken_rules.cache.json"))
    ttl_min = int(os.getenv("KRAKEN_RULES_CACHE_TTL_MIN", "30"))
    resp = _load_cache(cache_file, ttl_min)
    if resp is None:
        resp = client.get_asset_pairs()
        _save_cache(cache_file, resp)

    result = resp.get("result", {})
    rules: dict[str, KrakenPairRule] = {}
    for key, v in result.items():
        wsname = v.get("wsname")
        if not wsname:
            continue
        status = (v.get("status") or "online").lower()
        if status != "online":
            continue
        try:
            ordermin = float(v.get("ordermin") or 0.0)
            lot_decimals = int(v.get("lot_decimals") or 0)
            pair_decimals = int(v.get("pair_decimals") or 0)
        except Exception:
            continue
        costmin = float(v["costmin"]) if v.get("costmin") is not None else None
        tick_size = float(v["tick_size"]) if v.get("tick_size") is not None else None
        rule = KrakenPairRule(
            wsname=wsname,
            ordermin=ordermin,
            lot_decimals=lot_decimals,
            pair_decimals=pair_decimals,
            costmin=costmin,
            tick_size=tick_size,
        )
        aliases = {
            str(wsname).upper(),
            str(v.get("altname") or "").upper(),
            str(key).upper(),
        }
        normalized_wsname = normalize_symbol(str(wsname))
        if normalized_wsname:
            aliases.add(normalized_wsname.upper())
            aliases.add(to_kraken_symbol(normalized_wsname).replace("/", "").upper())
        aliases.discard("")
        for alias in aliases:
            rules[alias] = rule
    return rules
