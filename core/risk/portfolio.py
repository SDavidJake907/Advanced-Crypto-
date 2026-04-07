from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypedDict

from core.config.runtime import get_runtime_setting, get_symbol_lane
from core.state.store import load_sector_tags


DecisionType = Literal["allow", "block", "scale_down", "replace"]


class PortfolioDecision(TypedDict):
    decision: DecisionType
    size_factor: float
    reasons: List[str]
    replace_symbol: str | None
    replace_reason: str | None


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
    monitor_state: str = "RUN"
    monitor_reason: str = ""
    monitor_confidence: float = 0.0
    exit_posture: str = "RUN"
    exit_posture_reason: str = ""
    exit_posture_confidence: float = 0.0
    structure_state: str = "intact"
    max_price_seen: float | None = None
    min_price_seen: float | None = None
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    mfe_ts: str = ""
    mae_ts: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0
    etd_pct: float = 0.0
    etd_r: float = 0.0
    expected_edge_pct: float = 0.0
    risk_reward_ratio: float = 0.0


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
    resolved_lane = str(lane or "").strip().upper()
    if resolved_lane:
        return f"lane:{resolved_lane}"
    sector_tags = load_sector_tags()
    normalized = str(symbol).strip().upper()
    if normalized in sector_tags:
        return sector_tags[normalized]
    resolved_lane = str(get_symbol_lane(symbol)).upper()
    return f"lane:{resolved_lane}"


def _position_weakness_score(position: Position) -> float:
    monitor_score = {
        "RUN": 0.0,
        "STALL": 2.0,
        "WEAKEN": 3.0,
        "FAIL": 4.0,
        "ROTATE": 5.0,
    }.get(str(position.monitor_state or "RUN").upper(), 0.0)
    posture_score = {
        "RUN": 0.0,
        "STALE": 2.0,
        "TIGHTEN": 3.0,
        "EXIT": 4.0,
    }.get(str(position.exit_posture or "RUN").upper(), 0.0)
    structure_score = {
        "intact": 0.0,
        "fragile": 2.0,
        "broken": 4.0,
    }.get(str(position.structure_state or "intact").lower(), 0.0)
    return monitor_score + posture_score + structure_score


def _candidate_is_replacement_quality(features: dict | None) -> bool:
    if not isinstance(features, dict):
        return False
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    entry_recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
    trend_confirmed = bool(features.get("trend_confirmed", False))
    tp_after_cost_valid = bool(features.get("tp_after_cost_valid", False))
    point_breakdown = features.get("point_breakdown") if isinstance(features.get("point_breakdown"), dict) else {}
    net_edge_pct = float(
        features.get(
            "net_edge_pct",
            point_breakdown.get("net_edge_pct", 0.0) if isinstance(point_breakdown, dict) else 0.0,
        )
        or 0.0
    )
    return bool(
        entry_score >= 72.0
        and tp_after_cost_valid
        and net_edge_pct > 0.0
        and (trend_confirmed or entry_recommendation in {"BUY", "STRONG_BUY"})
    )


def build_opportunity_snapshot(
    *,
    positions: PositionState,
    candidate_symbol: str,
    features: dict | None,
) -> dict[str, object]:
    weakest_overall: Position | None = None
    weakest_overall_score = -1.0
    for position in positions.all():
        if position.symbol == candidate_symbol:
            continue
        weakness = _position_weakness_score(position)
        if weakness > weakest_overall_score:
            weakest_overall = position
            weakest_overall_score = weakness

    replacement = _select_replacement_candidate(
        positions=positions,
        incoming_symbol=candidate_symbol,
        features=features,
    )
    point_breakdown = features.get("point_breakdown") if isinstance(features, dict) else {}
    candidate_net_edge_pct = 0.0
    if isinstance(features, dict):
        candidate_net_edge_pct = float(
            features.get(
                "net_edge_pct",
                point_breakdown.get("net_edge_pct", 0.0) if isinstance(point_breakdown, dict) else 0.0,
            )
            or 0.0
        )
    return {
        "candidate_symbol": candidate_symbol,
        "candidate_entry_score": float(features.get("entry_score", 0.0) or 0.0) if isinstance(features, dict) else 0.0,
        "candidate_net_edge_pct": candidate_net_edge_pct,
        "candidate_replacement_quality": _candidate_is_replacement_quality(features),
        "weakest_held_symbol": weakest_overall.symbol if weakest_overall is not None else None,
        "weakest_held_monitor_state": getattr(weakest_overall, "monitor_state", None) if weakest_overall is not None else None,
        "weakest_held_exit_posture": getattr(weakest_overall, "exit_posture", None) if weakest_overall is not None else None,
        "weakest_held_structure_state": getattr(weakest_overall, "structure_state", None) if weakest_overall is not None else None,
        "weakest_held_weakness_score": weakest_overall_score if weakest_overall is not None else 0.0,
        "replace_ready_symbol": replacement.symbol if replacement is not None else None,
        "replace_ready": replacement is not None,
    }


def _select_replacement_candidate(
    *,
    positions: PositionState,
    incoming_symbol: str,
    features: dict | None,
) -> Position | None:
    if not _candidate_is_replacement_quality(features):
        return None
    weakest: Position | None = None
    weakest_score = 0.0
    soft_replaceable: Position | None = None
    soft_replace_weight = 1.0
    for position in positions.all():
        if position.symbol == incoming_symbol:
            continue
        score = _position_weakness_score(position)
        if score < 4.0:
            monitor_reason = str(position.monitor_reason or "").lower()
            exit_reason = str(position.exit_posture_reason or "").lower()
            if (
                abs(float(position.weight or 0.0)) <= 0.08
                and monitor_reason == "default_hold_state"
                and exit_reason == "default_hold_state"
            ):
                if soft_replaceable is None or abs(float(position.weight or 0.0)) < soft_replace_weight:
                    soft_replaceable = position
                    soft_replace_weight = abs(float(position.weight or 0.0))
            continue
        if weakest is None or score > weakest_score:
            weakest = position
            weakest_score = score
    return weakest if weakest is not None else soft_replaceable


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
    features: dict | None = None,
) -> PortfolioDecision:
    reasons: List[str] = []
    lane = lane or get_symbol_lane(symbol)
    size_factor = 1.0

    if proposed_weight > config.max_weight_per_symbol:
        reasons.append("proposed_weight_exceeds_per_symbol_cap")
        if proposed_weight > 0.0:
            size_factor = min(size_factor, config.max_weight_per_symbol / proposed_weight)

    if lane == "L4":
        max_meme_positions = int(get_runtime_setting("MEME_MAX_OPEN_POSITIONS"))
        meme_positions = sum(1 for position in positions.all() if position.lane == "L4")
        if meme_positions >= max_meme_positions and symbol not in positions.positions:
            reasons.append("max_meme_positions_reached")
            return {"decision": "block", "size_factor": 0.0, "reasons": reasons, "replace_symbol": None, "replace_reason": None}

    if positions.count() >= config.max_open_positions and symbol not in positions.positions:
        replacement = _select_replacement_candidate(
            positions=positions,
            incoming_symbol=symbol,
            features=features,
        )
        if replacement is not None:
            reasons.extend(["replace_weakest_position", f"replace_symbol:{replacement.symbol}"])
            return {
                "decision": "replace",
                "size_factor": 1.0,
                "reasons": reasons,
                "replace_symbol": replacement.symbol,
                "replace_reason": "candidate_stronger_than_weak_hold",
            }
        reasons.append("max_open_positions_reached")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons, "replace_symbol": None, "replace_reason": None}

    current_sector = _sector_for_symbol(symbol, lane)
    sector_positions = sum(
        1
        for position in positions.all()
        if position.symbol != symbol and _sector_for_symbol(position.symbol, position.lane) == current_sector
    )
    if sector_positions >= config.max_positions_per_sector:
        reasons.append(f"sector_limit_reached:{current_sector}")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons, "replace_symbol": None, "replace_reason": None}

    total_gross = positions.total_gross_exposure()
    effective_proposed_weight = abs(proposed_weight) * size_factor
    if total_gross + effective_proposed_weight > config.max_total_gross_exposure:
        replacement = _select_replacement_candidate(
            positions=positions,
            incoming_symbol=symbol,
            features=features,
        )
        if replacement is not None:
            reasons.extend(["replace_weakest_position", f"replace_symbol:{replacement.symbol}"])
            return {
                "decision": "replace",
                "size_factor": 1.0,
                "reasons": reasons,
                "replace_symbol": replacement.symbol,
                "replace_reason": "gross_exposure_replacement",
            }
        reasons.append("total_gross_exposure_limit")
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons, "replace_symbol": None, "replace_reason": None}

    if lane == "L4" and trend_conflict:
        reasons.append("meme_trend_conflict_scale_down")
        size_factor = min(size_factor, float(get_runtime_setting("MEME_TREND_CONFLICT_SCALE")))
    is_main_leader = symbol in {"BTC/USD", "XBTUSD", "XXBTZUSD", "XBT/USD"}
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
            if not is_main_leader:
                size_factor = min(size_factor, config.corr_scale_down)

    if active_correlations:
        avg_corr = sum(active_correlations) / len(active_correlations)
        if avg_corr >= config.avg_corr_scale_threshold:
            reasons.append(f"avg_corr_scale_down={avg_corr:.2f}")
            if not is_main_leader:
                size_factor = min(size_factor, config.avg_corr_scale_down)

    if high_corr_count > config.max_high_corr_same_direction:
        return {"decision": "block", "size_factor": 0.0, "reasons": reasons, "replace_symbol": None, "replace_reason": None}

    if size_factor < 1.0:
        return {"decision": "scale_down", "size_factor": size_factor, "reasons": reasons, "replace_symbol": None, "replace_reason": None}

    return {"decision": "allow", "size_factor": 1.0, "reasons": reasons, "replace_symbol": None, "replace_reason": None}
