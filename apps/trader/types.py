from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CandidateOrder:
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["limit"]
    price: float
    size_usd: float
    stop_atr_mult: float
    reason: str
