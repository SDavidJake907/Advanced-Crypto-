from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypedDict

from core.config.runtime import get_runtime_setting, get_symbol_lane
from core.state.store import load_sector_tags


DecisionType = Literal["allow", "block", "scale_down"]


class PortfolioDecision(TypedDict):
    decision: DecisionType
    size_factor: float
    reasons: List[str]


@dataclass
class PortfolioConfig:
    max_weight_per_symbol: float = 0.2
    max_total_gross_exposure: float = 1.0
    max_open_positions: int = 5
    corr_threshold: float = 0.8
    corr_scale_down: float = 0.5
    max_high_corr_same_direction: int = 1
    avg_corr_scale_threshold: float = 0.7
    avg_corr_scale_down: float = 0.6
    max_positions_per_sector: int = 2

    @classmethod
    def from_runtime(cls) -> "PortfolioConfig":
        return cls(
            max_weight_per_symbol=float(get_runtime_setting("PORTFOLIO_MAX_WEIGHT_PER_SYMBOL")),
            max_total_gross_exposure=float(get_runtime_setting("PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE")),
            max_open_positions=int(get_runtime_setting("PORTFOLIO_MAX_OPEN_POSITIONS")),
            corr_threshold=float(get_runtime_setting("PORTFOLIO_CORR_THRESHOLD")),
            corr_scale_down=float(get_runtime_setting("PORTFOLIO_CORR_SCALE_DOWN")),
            max_high_corr_same_direction=int(get_runtime_setting("PORTFOLIO_MAX_HIGH_CORR_SAME_DIRECTION")),
            avg_corr_scale_threshold=float(get_runtime_setting("PORTFOLIO_AVG_CORR_SCALE_THRESHOLD")),
            avg_corr_scale_down=float(get_runtime_setting("PORTFOLIO_AVG_CORR_SCALE_DOWN")),
            max_positions_per_sector=int(get_runtime_setting("PORTFOLIO_MAX_POSITIONS_PER_SECTOR")),
        )


@dataclass
class Position:
    symbol: str
    side: Literal["LONG", "SHORT"]
    weight: float
    lane: str = "L3"
    entry_price: float | None = None
    entry_bar_ts: str | None = None
    entry_bar_idx: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_r: float | None = None
    break_even_armed: bool = False
    trailing_armed: bool = False
    trail_stop: float | None = None
    entry_reasons: list[str] = field(default_factory=list)
    entry_thesis: str = ""
    expected_hold_style: str = ""
    invalidate_on: str = ""
    exit_posture: str = "RUN"
    exit_posture_reason: str = ""
    exit_posture_confidence: float = 0.0


class PositionState:
    def __init__(self) -> None:
        self.positions: Dict[str, Position] = {}

    def add_or_update(self, pos: Position) -> None:
        self.positions[pos.symbol] = pos

    def remove(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def get(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def all(self) -> List[Position]:
        return list(self.positions.values())

    def total_gross_exposure(self) -> float:
        return sum(abs(p.weight) for p in self.positions.values())

    def count(self) -> int:
        return len(self.positions)


def _sector_for_symbol(symbol: str, lane: str | None = None) -> str:
    sector_tags = load_sector_tags()
    normalized = str(symbol).strip().upper()
    if normalized in sector_tags:
        return sector_tags[normalized]
    resolved_lane = str(lane or get_symbol_lane(symbol)).upper()
    return f"lane:{resolved_lane}"


def evaluate_trade(
    *,
    config: PortfolioConfig,
    positions: PositionState,
    symbol: str,
    side: Literal["LONG", "SHORT"],
    proposed_weight: float,
    correlation_row,
    symbols: List[str],
    lane: str | None = None,
    trend_conflict: bool = False,
) -> PortfolioDecision:
    reasons: List[str] = []
    lane = lane or get_symbol_lane(symbol)

    if proposed_weight > config.max_weight_per_symbol:
        reasons.append("proposed_weight_exceeds_per_symbol_cap")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons}

    if lane == "L4":
        max_meme_positions = int(get_runtime_setting("MEME_MAX_OPEN_POSITIONS"))
        meme_positions = sum(1 for position in positions.all() if position.lane == "L4")
        if meme_positions >= max_meme_positions and symbol not in positions.positions:
            reasons.append("max_meme_positions_reached")
            return {"decision": "block", "size_factor": 0.0, "reasons": reasons}

    if positions.count() >= config.max_open_positions and symbol not in positions.positions:
        reasons.append("max_open_positions_reached")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons}

    current_sector = _sector_for_symbol(symbol, lane)
    sector_positions = sum(
        1
        for position in positions.all()
        if position.symbol != symbol and _sector_for_symbol(position.symbol, position.lane) == current_sector
    )
    if sector_positions >= config.max_positions_per_sector:
        reasons.append(f"sector_limit_reached:{current_sector}")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons}

    total_gross = positions.total_gross_exposure()
    if total_gross + abs(proposed_weight) > config.max_total_gross_exposure:
        reasons.append("total_gross_exposure_limit")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons}

    size_factor = 1.0
    if lane == "L4" and trend_conflict:
        reasons.append("meme_trend_conflict_scale_down")
        size_factor = min(size_factor, float(get_runtime_setting("MEME_TREND_CONFLICT_SCALE")))
    high_corr_count = 0
    active_correlations: list[float] = []
    for idx, other_symbol in enumerate(symbols):
        if other_symbol == symbol:
            continue

        existing = positions.get(other_symbol)
        if existing is None:
            continue
        corr = float(correlation_row[idx])
        if existing.side == side:
            active_correlations.append(abs(corr))
        if corr < config.corr_threshold:
            continue

        if existing.side == side:
            reasons.append(f"high_corr_with_{other_symbol}_same_direction_corr={corr:.2f}")
            high_corr_count += 1
            size_factor = min(size_factor, config.corr_scale_down)

    if active_correlations:
        avg_corr = sum(active_correlations) / len(active_correlations)
        if avg_corr >= config.avg_corr_scale_threshold:
            reasons.append(f"avg_corr_scale_down={avg_corr:.2f}")
            size_factor = min(size_factor, config.avg_corr_scale_down)

    if high_corr_count > config.max_high_corr_same_direction:
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons}

    if size_factor < 1.0:
        return {"decision": "scale_down", "size_factor": size_factor, "reasons": reasons}

    return {"decision": "allow", "size_factor": 1.0, "reasons": reasons}
