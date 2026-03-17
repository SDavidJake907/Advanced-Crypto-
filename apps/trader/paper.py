from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PaperFill:
    symbol: str
    side: str
    price: float
    size_base: float
    notional_usd: float
    ts: str
    reason: str


def simulate_limit_fill(
    *,
    symbol: str,
    side: str,
    limit_price: float,
    size_base: float,
    next_high: float,
    next_low: float,
) -> PaperFill | None:
    if side == "buy" and next_low <= limit_price:
        return PaperFill(
            symbol=symbol,
            side=side,
            price=limit_price,
            size_base=size_base,
            notional_usd=limit_price * size_base,
            ts=datetime.now(timezone.utc).isoformat(),
            reason="limit_fill",
        )
    if side == "sell" and next_high >= limit_price:
        return PaperFill(
            symbol=symbol,
            side=side,
            price=limit_price,
            size_base=size_base,
            notional_usd=limit_price * size_base,
            ts=datetime.now(timezone.utc).isoformat(),
            reason="limit_fill",
        )
    return None


def log_fill(fill: PaperFill) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    with Path("logs/paper_fills.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(fill.__dict__) + "\n")
