from dataclasses import dataclass
from typing import Iterable

from core.symbols import normalize_symbol


@dataclass
class UniversePolicy:
    active_min: int
    active_max: int
    max_adds: int
    max_removes: int
    cooldown_minutes: int
    min_volume_usd: float
    max_spread_bps: float
    min_price: float
    churn_threshold: float


def normalize_pair(pair: str) -> str:
    return normalize_symbol(pair)


def base_symbol(pair: str) -> str:
    return normalize_pair(pair).split("/")[0]


def apply_churn_threshold(candidate_score: float, worst_active_score: float, threshold: float) -> bool:
    # Only swap in if it's meaningfully better than the worst active
    return candidate_score > worst_active_score * threshold


def clamp_active_size(pairs: Iterable[str], active_max: int) -> list[str]:
    unique = list(dict.fromkeys(pairs))
    return unique[:active_max]
