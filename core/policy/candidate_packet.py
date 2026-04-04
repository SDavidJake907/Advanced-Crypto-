from __future__ import annotations

from typing import Any
import numpy as np

from core.memory.symbol_reliability import load_reliability_map, reliability_map_as_dict
from core.policy.final_score import FinalTradeScore, compute_final_score

LOCAL_NEMOTRON_CANDIDATE_FIELDS = (
    "symbol",
    "lane",
    "price",
    "final_score",
    "net_edge_pct",
    "avg_held_correlation",
    "max_held_correlation",
    "entry_score",
    "entry_recommendation",
    "reversal_risk",
    "promotion_reason",
    "momentum_5",
    "rotation_score",
    "volume_ratio",
    "ema9_above_ema20",
    "range_breakout_1h",
    "pullback_hold",
    "structure_quality",
    "behavior_score",
    "lesson_summary",
)


def _build_held_correlation_map(features: dict[str, Any], positions_state: Any) -> dict[str, float]:
    corr_row = features.get("correlation_row")
    corr_syms = list(features.get("correlation_symbols") or [])
    held_set = set(getattr(positions_state, "positions", {}).keys()) if positions_state is not None else set()

    held_correlation_map: dict[str, float] = {}
    if corr_row is None or not corr_syms:
        return held_correlation_map

    row = np.asarray(corr_row, dtype=np.float64).ravel()
    for idx, sym in enumerate(corr_syms):
        if sym in held_set and idx < len(row) and np.isfinite(row[idx]):
            held_correlation_map[sym] = float(row[idx])
    return held_correlation_map


def _held_correlation_stats(held_correlation_map: dict[str, float]) -> tuple[float, float]:
    if not held_correlation_map:
        return 0.0, 0.0
    values = [abs(float(v)) for v in held_correlation_map.values() if np.isfinite(v)]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(max(values))


def _load_reliability_dict() -> dict[str, Any]:
    try:
        return reliability_map_as_dict(load_reliability_map())
    except Exception:
        return {}


def _normalize_lesson_summary(lesson_summary: Any) -> list[str]:
    if isinstance(lesson_summary, list):
        return [str(item).strip() for item in lesson_summary if str(item).strip()][:6]
    if isinstance(lesson_summary, str):
        items: list[str] = []
        for line in lesson_summary.splitlines():
            text = line.strip()
            if not text or text.startswith("==="):
                continue
            if text.startswith("- "):
                text = text[2:].strip()
            items.append(text)
            if len(items) >= 6:
                break
        return items
    return []


def _compact_behavior_score(behavior_score: Any) -> Any:
    if isinstance(behavior_score, dict):
        compact: dict[str, Any] = {}
        for key in ("score", "threshold_advice", "verdict", "confidence"):
            if key in behavior_score:
                compact[key] = behavior_score[key]
        return compact or behavior_score
    if isinstance(behavior_score, str):
        lines = [line.strip() for line in behavior_score.splitlines() if line.strip()]
        return " | ".join(lines[:3])[:240]
    return behavior_score


def compute_candidate_economics(
    features: dict[str, Any],
    *,
    reliability_map: dict[str, Any] | None = None,
    held_correlation_map: dict[str, float] | None = None,
) -> FinalTradeScore:
    """Compute the shared deterministic economics object for one candidate."""
    return compute_final_score(
        features,
        reliability_map=reliability_map or {},
        held_correlation_map=held_correlation_map or None,
    )


def build_candidate_packet(
    *,
    features: dict[str, Any],
    positions_state: Any = None,
    portfolio_state: Any = None,
    lesson_summary: list[str] | None = None,
    behavior_score: Any = None,
    reliability_map: dict[str, Any] | None = None,
    held_correlation_map: dict[str, float] | None = None,
    economics: FinalTradeScore | None = None,
) -> dict[str, Any]:
    """Build the canonical candidate packet consumed by downstream adapters."""
    resolved_reliability_map = reliability_map or _load_reliability_dict()
    resolved_held_correlation_map = held_correlation_map if held_correlation_map is not None else _build_held_correlation_map(features, positions_state)
    final = economics or compute_candidate_economics(
        features,
        reliability_map=resolved_reliability_map,
        held_correlation_map=resolved_held_correlation_map,
    )
    avg_held_correlation, max_held_correlation = _held_correlation_stats(resolved_held_correlation_map)
    return {
        "symbol": features.get("symbol"),
        "lane": features.get("lane"),
        "price": features.get("price"),
        "final_score": final.final_score,
        "net_edge_pct": final.net_edge_pct,
        "score_breakdown": final.score_breakdown,
        "breakdown_notes": final.breakdown_notes,
        "fear_greed_bonus": final.fear_greed_bonus,
        "btc_dominance_bonus": final.btc_dominance_bonus,
        "reliability_bonus": final.reliability_bonus,
        "basket_fit_bonus": final.basket_fit_bonus,
        "spread_penalty": final.spread_penalty,
        "cost_penalty": final.cost_penalty,
        "correlation_penalty": final.correlation_penalty,
        "avg_held_correlation": round(avg_held_correlation, 4),
        "max_held_correlation": round(max_held_correlation, 4),
        "entry_score": features.get("entry_score"),
        "entry_recommendation": features.get("entry_recommendation"),
        "reversal_risk": features.get("reversal_risk"),
        "promotion_tier": features.get("promotion_tier"),
        "promotion_reason": features.get("promotion_reason"),
        "momentum_5": features.get("momentum_5"),
        "momentum_14": features.get("momentum_14"),
        "rotation_score": features.get("rotation_score"),
        "volume_ratio": features.get("volume_ratio"),
        "volume_surge": features.get("volume_surge"),
        "rsi": features.get("rsi"),
        "spread_pct": features.get("spread_pct"),
        "price_zscore": features.get("price_zscore"),
        "trend_1h": features.get("trend_1h"),
        "regime_7d": features.get("regime_7d"),
        "macro_30d": features.get("macro_30d"),
        "atr_pct": features.get("atr_pct"),
        "volatility_percentile": features.get("volatility_percentile"),
        "compression_score": features.get("compression_score"),
        "expansion_score": features.get("expansion_score"),
        "volatility_state": features.get("volatility_state"),
        "trend_confirmed": features.get("trend_confirmed"),
        "ranging_market": features.get("ranging_market"),
        "short_tf_ready_5m": features.get("short_tf_ready_5m"),
        "short_tf_ready_15m": features.get("short_tf_ready_15m"),
        "ema9_above_ema20": features.get("ema9_above_ema20"),
        "range_breakout_1h": features.get("range_breakout_1h"),
        "pullback_hold": features.get("pullback_hold"),
        "higher_low_count": features.get("higher_low_count"),
        "bullish_divergence": features.get("bullish_divergence"),
        "bearish_divergence": features.get("bearish_divergence"),
        "divergence_strength": features.get("divergence_strength"),
        "divergence_age_bars": features.get("divergence_age_bars"),
        "overextended": features.get("overextended"),
        "range_pos_1h": features.get("range_pos_1h"),
        "book_imbalance": features.get("book_imbalance"),
        "book_wall_pressure": features.get("book_wall_pressure"),
        "structure_quality": features.get("structure_quality"),
        "pattern_candidate": features.get("pattern_candidate"),
        "pattern_verification": features.get("pattern_verification"),
        "phi3_veto_flag": features.get("phi3_veto_flag"),
        "continuation_quality": features.get("continuation_quality"),
        "risk_quality": features.get("risk_quality"),
        "trade_quality": features.get("trade_quality"),
        "xgb_score": features.get("xgb_score"),
        "behavior_score": behavior_score,
        "lesson_summary": _normalize_lesson_summary(lesson_summary),
        "held_correlation_map": resolved_held_correlation_map,
    }


def build_local_nemotron_candidate_packet(candidate_packet: dict[str, Any]) -> dict[str, Any]:
    compact = {field: candidate_packet.get(field) for field in LOCAL_NEMOTRON_CANDIDATE_FIELDS}
    compact["lesson_summary"] = _normalize_lesson_summary(candidate_packet.get("lesson_summary"))[:3]
    compact["behavior_score"] = _compact_behavior_score(candidate_packet.get("behavior_score"))
    return compact
