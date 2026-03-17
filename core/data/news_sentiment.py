from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any

import httpx


@dataclass
class TrendingCoin:
    symbol: str
    change_24h: float


@dataclass
class SentimentSnapshot:
    fng_value: int = 50
    fng_label: str = "unknown"
    total_market_cap_usd: float = 0.0
    btc_dominance: float = 0.0
    market_cap_change_24h: float = 0.0
    trending: list[TrendingCoin] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trending"] = [asdict(item) for item in self.trending]
        return payload


class NewsSentimentFeed:
    def __init__(self, fetch_interval_sec: float = 300.0) -> None:
        self.client = httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "KrakenSK/1.0"})
        self.snapshot = SentimentSnapshot()
        self.last_fetch_ts = 0.0
        self.fetch_interval_sec = fetch_interval_sec

    async def maybe_update(self, now: float | None = None) -> None:
        now_ts = now if now is not None else time.time()
        if now_ts - self.last_fetch_ts < self.fetch_interval_sec:
            return
        self.last_fetch_ts = now_ts
        await self._update()

    async def _update(self) -> None:
        fng, global_data, trending = await self._fetch_all()
        if fng is not None:
            self.snapshot.fng_value = int(fng.get("value", 50))
            self.snapshot.fng_label = str(fng.get("value_classification", "unknown"))
        if global_data is not None:
            self.snapshot.total_market_cap_usd = float(global_data.get("market_cap_usd", 0.0))
            self.snapshot.btc_dominance = float(global_data.get("btc_dominance", 0.0))
            self.snapshot.market_cap_change_24h = float(global_data.get("market_cap_change_24h", 0.0))
        if trending is not None:
            self.snapshot.trending = trending

    async def _fetch_all(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[TrendingCoin] | None]:
        fng = await self._fetch_fear_greed()
        global_data = await self._fetch_global()
        trending = await self._fetch_trending()
        return fng, global_data, trending

    async def _fetch_fear_greed(self) -> dict[str, Any] | None:
        try:
            response = await self.client.get("https://api.alternative.me/fng/?limit=1")
            response.raise_for_status()
            body = response.json()
            data = (body.get("data") or [{}])[0]
            return {
                "value": int(data.get("value", 50)),
                "value_classification": str(data.get("value_classification", "unknown")),
            }
        except Exception:
            return None

    async def _fetch_global(self) -> dict[str, Any] | None:
        try:
            response = await self.client.get("https://api.coingecko.com/api/v3/global")
            response.raise_for_status()
            body = response.json().get("data", {})
            return {
                "market_cap_usd": float((body.get("total_market_cap") or {}).get("usd", 0.0)),
                "btc_dominance": float((body.get("market_cap_percentage") or {}).get("btc", 0.0)),
                "market_cap_change_24h": float(body.get("market_cap_change_percentage_24h_usd", 0.0)),
            }
        except Exception:
            return None

    async def _fetch_trending(self) -> list[TrendingCoin] | None:
        try:
            response = await self.client.get("https://api.coingecko.com/api/v3/search/trending")
            response.raise_for_status()
            coins = response.json().get("coins") or []
            results: list[TrendingCoin] = []
            for coin in coins[:10]:
                item = coin.get("item") or {}
                data = item.get("data") or {}
                changes = data.get("price_change_percentage_24h") or {}
                results.append(
                    TrendingCoin(
                        symbol=str(item.get("symbol", "")).upper(),
                        change_24h=float(changes.get("usd", 0.0)),
                    )
                )
            return results
        except Exception:
            return None

    def build_news_block(self) -> str:
        snapshot = self.snapshot
        if snapshot.fng_label == "unknown" and snapshot.total_market_cap_usd == 0.0:
            return ""
        trending = " ".join(
            f"{coin.symbol}({coin.change_24h:+.1f}%)" for coin in snapshot.trending[:7]
        )
        return (
            "=== NEWS & SENTIMENT ===\n"
            f"Fear/Greed: {snapshot.fng_value} ({snapshot.fng_label})\n"
            f"Market: ${snapshot.total_market_cap_usd / 1e9:.0f}B | BTC dom {snapshot.btc_dominance:.1f}% | 24h {snapshot.market_cap_change_24h:+.2f}%\n"
            f"Trending: {trending}\n"
        )
