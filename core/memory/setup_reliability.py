"""Setup-level reliability summaries built from persisted trade outcomes.

This extends the existing symbol-only memory with setup-oriented views so we
can evaluate which lanes, patterns, and entry conditions actually work live.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from core.memory.trade_memory import OutcomeRecord, TradeMemoryStore, build_setup_key


@dataclass
class SetupReliabilityRecord:
    key: str
    dimension: str
    trade_count: int
    win_rate: float
    avg_pnl_pct: float
    avg_hold_min: float
    avg_expected_edge_pct: float
    avg_realized_edge_pct: float
    avg_edge_capture_ratio: float
    expectancy_pct: float
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_primary_reason(entry_reasons: list[str]) -> str:
    if not entry_reasons:
        return "none"
    return str(entry_reasons[0] or "").strip().lower() or "none"


def _group_outcomes(
    outcomes: list[OutcomeRecord],
    key_fn: Callable[[OutcomeRecord], str],
    *,
    dimension: str,
    min_trades: int,
) -> dict[str, SetupReliabilityRecord]:
    grouped: dict[str, list[OutcomeRecord]] = {}
    for outcome in outcomes:
        key = key_fn(outcome)
        if not key:
            continue
        grouped.setdefault(key, []).append(outcome)

    result: dict[str, SetupReliabilityRecord] = {}
    for key, records in grouped.items():
        if len(records) < min_trades:
            continue
        trade_count = len(records)
        win_rate = sum(1 for o in records if o.pnl_pct > 0.001) / trade_count
        avg_pnl_pct = sum(o.pnl_pct for o in records) / trade_count
        avg_hold_min = sum(o.hold_minutes for o in records) / trade_count
        avg_expected_edge_pct = sum(float(o.expected_edge_pct or 0.0) for o in records) / trade_count
        avg_realized_edge_pct = sum(float(o.realized_edge_pct or 0.0) for o in records) / trade_count
        avg_edge_capture_ratio = sum(float(o.edge_capture_ratio or 0.0) for o in records) / trade_count

        if win_rate >= 0.65 and avg_pnl_pct > 0.003:
            verdict = "strong"
        elif win_rate < 0.35 and avg_pnl_pct <= 0.0:
            verdict = "avoid"
        elif win_rate < 0.45:
            verdict = "weak"
        else:
            verdict = "neutral"

        result[key] = SetupReliabilityRecord(
            key=key,
            dimension=dimension,
            trade_count=trade_count,
            win_rate=round(win_rate, 3),
            avg_pnl_pct=round(avg_pnl_pct, 4),
            avg_hold_min=round(avg_hold_min, 1),
            avg_expected_edge_pct=round(avg_expected_edge_pct, 4),
            avg_realized_edge_pct=round(avg_realized_edge_pct, 4),
            avg_edge_capture_ratio=round(avg_edge_capture_ratio, 4),
            expectancy_pct=round(avg_pnl_pct, 4),
            verdict=verdict,
        )
    return result


def build_setup_reliability_map(
    *,
    store: TradeMemoryStore | None = None,
    lookback: int = 200,
    min_trades: int = 3,
) -> dict[str, dict[str, SetupReliabilityRecord]]:
    store = store or TradeMemoryStore()
    outcomes = store.load_recent_outcomes(lookback)
    if not outcomes:
        return {
            "by_lane": {},
            "by_pattern": {},
            "by_entry_reason": {},
            "by_symbol_regime": {},
            "by_setup_key": {},
        }

    return {
        "by_lane": _group_outcomes(
            outcomes,
            lambda o: str(o.lane or "").strip().upper() or "",
            dimension="lane",
            min_trades=min_trades,
        ),
        "by_pattern": _group_outcomes(
            outcomes,
            lambda o: str(o.pattern_name or "").strip().lower() or "",
            dimension="pattern",
            min_trades=min_trades,
        ),
        "by_entry_reason": _group_outcomes(
            outcomes,
            lambda o: _safe_primary_reason(o.entry_reasons),
            dimension="entry_reason",
            min_trades=min_trades,
        ),
        "by_symbol_regime": _group_outcomes(
            outcomes,
            lambda o: f"{o.symbol}|{str(o.regime_label or '').strip().lower() or 'unknown'}",
            dimension="symbol_regime",
            min_trades=min_trades,
        ),
        "by_setup_key": _group_outcomes(
            outcomes,
            lambda o: str(o.setup_key or "").strip()
            or build_setup_key(
                lane=o.lane,
                pattern_name=o.pattern_name,
                entry_recommendation=o.entry_recommendation,
                entry_reasons=o.entry_reasons,
            ),
            dimension="setup_key",
            min_trades=min_trades,
        ),
    }


def setup_reliability_as_dict(
    records: dict[str, dict[str, SetupReliabilityRecord]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        group: {key: record.to_dict() for key, record in group_records.items()}
        for group, group_records in records.items()
    }


def _strength_sort_key(record: SetupReliabilityRecord) -> tuple[float, float, int]:
    return (record.expectancy_pct, record.win_rate, record.trade_count)


def _weakness_sort_key(record: SetupReliabilityRecord) -> tuple[float, float, int]:
    return (record.expectancy_pct, record.win_rate, -record.trade_count)


def summarize_setup_reliability(
    records: dict[str, dict[str, SetupReliabilityRecord]],
    *,
    limit: int = 5,
) -> dict[str, dict[str, Any]]:
    limit = max(1, int(limit))
    summary: dict[str, dict[str, Any]] = {}
    for group, group_records in records.items():
        values = list(group_records.values())
        strongest = sorted(values, key=_strength_sort_key, reverse=True)[:limit]
        weakest = sorted(values, key=_weakness_sort_key)[:limit]
        summary[group] = {
            "count": len(values),
            "strongest": [record.to_dict() for record in strongest],
            "weakest": [record.to_dict() for record in weakest],
        }
    return summary


def build_setup_reliability_summary(
    *,
    store: TradeMemoryStore | None = None,
    lookback: int = 200,
    min_trades: int = 3,
    limit: int = 5,
) -> dict[str, dict[str, Any]]:
    grouped = build_setup_reliability_map(
        store=store,
        lookback=lookback,
        min_trades=min_trades,
    )
    return summarize_setup_reliability(grouped, limit=limit)
