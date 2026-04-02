from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PatternName = Literal[
    "double_bottom",
    "double_top",
    "inverse_head_and_shoulders",
    "head_and_shoulders",
    "bullish_rectangle",
    "bearish_rectangle",
    "bullish_flag",
    "bearish_flag",
    "ascending_triangle",
    "descending_triangle",
    "symmetrical_triangle",
    "falling_wedge",
    "rising_wedge",
    "none",
]

Bias = Literal["bullish", "bearish", "neutral"]
Validity = Literal["valid", "invalid", "unclear"]


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["high", "low"]


@dataclass
class PatternEvidence:
    pattern: PatternName
    bias: Bias
    timeframe: str
    symbol: str

    confidence_raw: float = 0.0
    symmetry_score: float = 0.0
    breakout_score: float = 0.0
    retest_score: float = 0.0
    volume_confirmation_score: float = 0.0
    trend_alignment_score: float = 0.0

    neckline_level: float | None = None
    support_level: float | None = None
    resistance_level: float | None = None
    breakout_level: float | None = None
    stop_level_hint: float | None = None
    target_level_hint: float | None = None

    breakout_confirmed: bool = False
    retest_seen: bool = False
    retest_holding: bool = False

    swing_points: list[SwingPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["swing_points"] = [asdict(item) for item in self.swing_points]
        return payload
