from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from core.symbols import normalize_symbol, to_kraken_symbol


@dataclass
class KrakenRestConfig:
    base_url: str = "https://api.kraken.com"
    timeout_s: float = 10.0


class KrakenRestClient:
    def __init__(self, cfg: KrakenRestConfig | None = None) -> None:
        self.cfg = cfg or KrakenRestConfig()
        self._client = httpx.Client(base_url=self.cfg.base_url, timeout=self.cfg.timeout_s)
        self._api_key = os.getenv("KRAKEN_API_KEY", "")
        self._api_secret = os.getenv("KRAKEN_API_SECRET", "")
        self._asset_pairs_cache_path = Path(os.getenv("ASSET_PAIRS_CACHE_FILE", "configs/kraken_asset_pairs.cache.json"))

    def resolve_order_pair(self, symbol: str) -> str:
        normalized = normalize_symbol(symbol)
        target = to_kraken_symbol(normalized)
        asset_pairs = self.get_cached_asset_pairs().get("result", {})
        for key, value in asset_pairs.items():
            wsname = str(value.get("wsname") or "").upper()
            altname = str(value.get("altname") or "").upper()
            if target == wsname or target.replace("/", "") == altname:
                return altname or target.replace("/", "")
        return target.replace("/", "")

    def _private_request(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._api_key or not self._api_secret:
            raise RuntimeError("Kraken API credentials are not configured")
        payload: dict[str, Any] = dict(data or {})
        payload["nonce"] = str(int(time.time() * 1000))
        post_data = urlencode(payload)
        encoded = (payload["nonce"] + post_data).encode("utf-8")
        message = path.encode("utf-8") + hashlib.sha256(encoded).digest()
        secret = base64.b64decode(self._api_secret)
        signature = base64.b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode("ascii")
        headers = {
            "API-Key": self._api_key,
            "API-Sign": signature,
        }
        r = self._client.post(path, data=payload, headers=headers)
        r.raise_for_status()
        response = r.json()
        errors = response.get("error") or []
        if errors:
            raise RuntimeError(f"Kraken private API error for {path}: {errors}")
        return response

    def _public_request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        response = r.json()
        errors = response.get("error") or []
        if errors:
            raise RuntimeError(f"Kraken public API error for {path}: {errors}")
        return response

    def get_ticker(self, pairs: list[str]) -> dict[str, Any]:
        pair_str = ",".join(pairs)
        return self._public_request("/0/public/Ticker", {"pair": pair_str})

    def get_ohlc(self, pair: str, interval_min: int = 60, since: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"pair": pair, "interval": interval_min}
        if since is not None:
            params["since"] = since
        return self._public_request("/0/public/OHLC", params)

    def get_asset_pairs(self) -> dict[str, Any]:
        return self._public_request("/0/public/AssetPairs")

    def get_cached_asset_pairs(self, ttl_min: int = 30) -> dict[str, Any]:
        if self._asset_pairs_cache_path.exists():
            age_s = time.time() - self._asset_pairs_cache_path.stat().st_mtime
            if age_s <= ttl_min * 60:
                try:
                    return json.loads(self._asset_pairs_cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        response = self.get_asset_pairs()
        self._asset_pairs_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._asset_pairs_cache_path.write_text(json.dumps(response), encoding="utf-8")
        return response

    def get_ticker_snapshot(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        asset_pairs = self.get_cached_asset_pairs().get("result", {})
        meta_map: dict[str, dict[str, Any]] = {}
        for key, value in asset_pairs.items():
            wsname = str(value.get("wsname") or "").upper()
            altname = str(value.get("altname") or "").upper()
            if wsname:
                meta_map[wsname] = {"key": key, **value}
            if altname:
                meta_map[altname] = {"key": key, **value}

        requested: list[tuple[str, str, dict[str, Any]]] = []
        for symbol in symbols:
            normalized = normalize_symbol(symbol)
            kraken_symbol = to_kraken_symbol(normalized)
            meta = meta_map.get(kraken_symbol)
            if meta:
                requested.append((normalized, kraken_symbol, meta))
        if not requested:
            return {}

        pair_keys = list(dict.fromkeys(str(meta.get("key") or normalized) for _, normalized, meta in requested))
        ticker = self.get_ticker(pair_keys).get("result", {})
        snapshot: dict[str, dict[str, float]] = {}
        for original, normalized, meta in requested:
            ticker_key = str(meta.get("altname") or meta.get("wsname") or meta.get("key") or normalized).upper()
            payload = ticker.get(ticker_key) or ticker.get(normalized) or ticker.get(meta.get("key"))
            if not isinstance(payload, dict):
                continue
            try:
                ask = float((payload.get("a") or [0.0])[0])
                bid = float((payload.get("b") or [0.0])[0])
                last = float((payload.get("c") or [0.0])[0])
                vol = float((payload.get("v") or [0.0, 0.0])[1])
            except Exception:
                continue
            spread_pct = 0.0
            mid = (bid + ask) / 2.0
            if mid > 0.0 and ask >= bid:
                spread_pct = ((ask - bid) / mid) * 100.0
            snapshot[original] = {
                "bid": bid,
                "ask": ask,
                "last": last,
                "spread_pct": spread_pct,
                "quote_volume": vol * last if last > 0.0 else 0.0,
            }
        return snapshot

    def get_balance(self) -> dict[str, Any]:
        return self._private_request("/0/private/Balance")

    def get_balance_ex(self) -> dict[str, Any]:
        return self._private_request("/0/private/BalanceEx")

    def get_trade_balance(self, asset: str = "ZUSD") -> dict[str, Any]:
        return self._private_request("/0/private/TradeBalance", {"asset": asset})

    def get_trades_history(self, type_filter: str = "all", trades: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": type_filter, "trades": trades}
        return self._private_request("/0/private/TradesHistory", payload)

    def add_order(
        self,
        *,
        symbol: str,
        side: str,
        ordertype: str,
        volume: float,
        price: float | None = None,
        oflags: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pair": self.resolve_order_pair(symbol),
            "type": side.lower(),
            "ordertype": ordertype.lower(),
            "volume": f"{float(volume):.8f}",
        }
        if price is not None and price > 0.0:
            payload["price"] = f"{float(price):.10f}".rstrip("0").rstrip(".")
        if oflags:
            payload["oflags"] = str(oflags)
        return self._private_request("/0/private/AddOrder", payload)
