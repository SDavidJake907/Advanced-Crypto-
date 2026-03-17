from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)


@dataclass
class DexPairSummary:
    symbol: str
    chain_id: str
    dex_id: str
    pair_address: str
    price_usd: float
    volume_24h: float
    liquidity_usd: float
    price_change_24h: float
    pair_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DexScreenerFeed:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "KrakenSK/1.0"})

    async def fetch_symbol_summary(self, symbol: str) -> DexPairSummary | None:
        token = str(symbol).split("/")[0].strip().upper()
        if not token:
            return None
        try:
            response = await self.client.get(
                "https://api.dexscreener.com/latest/dex/search",
                params={"q": token},
            )
            response.raise_for_status()
            body = response.json()
            pairs = body.get("pairs") or []
            if not isinstance(pairs, list) or not pairs:
                return None

            best = None
            best_liquidity = -1.0
            for pair in pairs[:20]:
                base = pair.get("baseToken") or {}
                quote = pair.get("quoteToken") or {}
                if str(base.get("symbol", "")).upper() != token and str(quote.get("symbol", "")).upper() != token:
                    continue
                liquidity_usd = float(((pair.get("liquidity") or {}).get("usd")) or 0.0)
                if liquidity_usd > best_liquidity:
                    best = pair
                    best_liquidity = liquidity_usd
            if best is None:
                return None

            volume_24h = float(((best.get("volume") or {}).get("h24")) or 0.0)
            price_change_24h = float(((best.get("priceChange") or {}).get("h24")) or 0.0)
            return DexPairSummary(
                symbol=symbol,
                chain_id=str(best.get("chainId", "")),
                dex_id=str(best.get("dexId", "")),
                pair_address=str(best.get("pairAddress", "")),
                price_usd=float(best.get("priceUsd") or 0.0),
                volume_24h=volume_24h,
                liquidity_usd=float(((best.get("liquidity") or {}).get("usd")) or 0.0),
                price_change_24h=price_change_24h,
                pair_url=str(best.get("url", "")),
            )
        except Exception as exc:
            _LOGGER.warning("dexscreener lookup failed for %s: %s", symbol, exc)
            return None

    async def fetch_market_summary(self, symbols: list[str]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for symbol in symbols[:15]:
            summary = await self.fetch_symbol_summary(symbol)
            if summary is not None:
                results[symbol] = summary.to_dict()
        return results
