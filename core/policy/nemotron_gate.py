from __future__ import annotations

from typing import Any

from core.config.runtime import get_runtime_setting
from core.risk.portfolio import PositionState
from core.state.store import load_universe


def load_universe_candidate_context(symbol: str) -> dict[str, Any]:
    universe = load_universe()
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    top_n = int(get_runtime_setting("NEMOTRON_TOP_CANDIDATE_COUNT"))
    top_scored = meta.get("top_scored", []) if isinstance(meta.get("top_scored", []), list) else []
    hot_candidates = meta.get("hot_candidates", []) if isinstance(meta.get("hot_candidates", []), list) else []
    avoid_candidates = meta.get("avoid_candidates", []) if isinstance(meta.get("avoid_candidates", []), list) else []
    top_ranked = meta.get("top_ranked", []) if isinstance(meta.get("top_ranked", []), list) else []
    lane_supervision = meta.get("lane_supervision", []) if isinstance(meta.get("lane_supervision", []), list) else []
    return {
        "top_scored": top_scored[:top_n],
        "hot_candidates": hot_candidates[:top_n],
        "avoid_candidates": avoid_candidates[:top_n],
        "top_ranked": top_ranked[:top_n],
        "lane_supervision": lane_supervision[:top_n],
        "current_symbol_is_top_candidate": symbol in top_ranked[:top_n]
        or any(str(item.get("symbol")) == symbol for item in top_scored[:top_n]),
    }


def symbol_in_top_candidates(
    symbol: str,
    positions_state: PositionState,
    features: dict[str, Any],
    market_state_review: dict[str, Any] | None = None,
    posture_review: dict[str, Any] | None = None,
) -> bool:
    if positions_state.get(symbol) is not None:
        return True
    context = load_universe_candidate_context(symbol)
    if context.get("current_symbol_is_top_candidate"):
        return True
    if any(str(item.get("symbol")) == symbol for item in context.get("hot_candidates", [])):
        return True

    entry_recommendation = str(features.get("entry_recommendation", "")).upper()
    lane = str(features.get("lane", "L3") or "L3").upper()
    reversal_risk = str(features.get("reversal_risk", "")).upper()
    lane_conflict = bool(features.get("lane_conflict"))
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    min_watch_score = float(get_runtime_setting("NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE"))
    min_watch_volume_ratio = float(get_runtime_setting("NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO"))
    if lane == "L4":
        min_watch_score = float(get_runtime_setting("MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE"))
        min_watch_volume_ratio = float(get_runtime_setting("MEME_NEMOTRON_WATCH_MIN_VOLUME_RATIO"))
    market_state = str((market_state_review or {}).get("market_state", "transition")).lower()
    posture_promotion_bias = str((posture_review or {}).get("promotion_bias", "normal")).lower()

    if market_state == "ranging" and features.get("lane") in {"L1", "L3"}:
        return False

    if posture_promotion_bias == "wider":
        min_watch_score = max(min_watch_score - 4.0, 35.0)
        min_watch_volume_ratio = max(min_watch_volume_ratio - 0.05, 1.0)
    elif posture_promotion_bias == "tighter":
        min_watch_score += 4.0
        min_watch_volume_ratio += 0.05

    if bool(get_runtime_setting("NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP")):
        if entry_recommendation in {"BUY", "STRONG_BUY"} and reversal_risk == "LOW":
            return True

    if bool(get_runtime_setting("NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP")):
        if (
            entry_recommendation in {"BUY", "STRONG_BUY"}
            and reversal_risk == "MEDIUM"
            and rotation_score > 0.0
            and entry_score >= 55.0
        ):
            return True

    if bool(get_runtime_setting("NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT")):
        if (
            entry_recommendation == "WATCH"
            and reversal_risk == "LOW"
            and (
                lane_conflict
                or entry_score >= min_watch_score
                or volume_ratio >= min_watch_volume_ratio
            )
            and (
                rotation_score > 0.0
                or momentum_5 > 0.0
                or volume_ratio >= min_watch_volume_ratio
            )
        ):
            return True

    if lane == "L4":
        if (
            entry_recommendation in {"WATCH", "BUY", "STRONG_BUY"}
            and reversal_risk != "HIGH"
            and (
                entry_score >= min_watch_score
                or volume_ratio >= min_watch_volume_ratio
                or volume_surge >= 0.35
                or symbol_trending
            )
            and (momentum_5 > 0.0 or rotation_score > 0.0 or volume_surge >= 0.35)
        ):
            return True

    return False


def passes_deterministic_candidate_gate(
    *,
    symbol: str,
    positions_state: PositionState,
    features: dict[str, Any],
    universe_context: dict[str, Any],
) -> tuple[bool, str]:
    if positions_state.get(symbol) is not None:
        return True, "existing_position"
    if not bool(features.get("indicators_ready", True)):
        return False, "indicator_warmup"

    entry_recommendation = str(features.get("entry_recommendation", "")).upper()
    lane = str(features.get("lane", "L3") or "L3").upper()
    reversal_risk = str(features.get("reversal_risk", "")).upper()
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    gate_min_entry_score = float(get_runtime_setting("NEMOTRON_GATE_MIN_ENTRY_SCORE"))
    gate_min_volume_ratio = float(get_runtime_setting("NEMOTRON_GATE_MIN_VOLUME_RATIO"))
    if lane == "L4":
        gate_min_entry_score = float(get_runtime_setting("MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE"))
        gate_min_volume_ratio = float(get_runtime_setting("MEME_NEMOTRON_GATE_MIN_VOLUME_RATIO"))
    trend_confirmed = bool(features.get("trend_confirmed"))

    if entry_recommendation == "AVOID":
        return False, "entry_recommendation_avoid"
    if reversal_risk == "HIGH":
        return False, "reversal_risk_high"
    if (
        entry_score < gate_min_entry_score
        and volume_ratio < gate_min_volume_ratio
        and rotation_score <= 0.0
        and momentum_5 <= 0.0
        and volume_surge < 0.35
        and not symbol_trending
    ):
        return False, "weak_pre_filter"
    if not symbol_in_top_candidates(symbol, positions_state, features):
        return False, "outside_top_candidate_set"
    if (
        not universe_context.get("current_symbol_is_top_candidate")
        and entry_recommendation == "BUY"
        and reversal_risk != "LOW"
        and entry_score < max(gate_min_entry_score + (4.0 if lane == "L4" else 8.0), 50.0 if lane == "L4" else 58.0)
        and volume_ratio < max(gate_min_volume_ratio + 0.05, 1.0 if lane == "L4" else 1.1)
        and volume_surge < 0.35
    ):
        return False, "buy_not_promoted"
    if (
        not universe_context.get("current_symbol_is_top_candidate")
        and entry_recommendation == "WATCH"
        and reversal_risk != "LOW"
        and (
            entry_score < max(gate_min_entry_score + (2.0 if lane == "L4" else 10.0), 44.0 if lane == "L4" else 60.0)
            or volume_ratio < max(gate_min_volume_ratio + (0.0 if lane == "L4" else 0.1), 1.0 if lane == "L4" else 1.1)
            or (rotation_score <= 0.0 and momentum_5 <= 0.0 and not trend_confirmed and volume_surge < 0.35 and not symbol_trending)
        )
    ):
        return False, "watch_not_promoted"
    return True, "passed"


def should_run_nemotron(
    *,
    symbol: str,
    features: dict[str, Any],
    positions_state: PositionState,
    universe_context: dict[str, Any],
    candidate_review: dict[str, Any],
) -> bool:
    if positions_state.get(symbol) is not None:
        return True
    if universe_context.get("current_symbol_is_top_candidate"):
        return True
    entry_recommendation = str(features.get("entry_recommendation", "")).upper()
    lane = str(features.get("lane", "L3") or "L3").upper()
    reversal_risk = str(features.get("reversal_risk", "")).upper()
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    symbol_trending = bool(features.get("sentiment_symbol_trending", False))
    trend_confirmed = bool(features.get("trend_confirmed"))
    promotion_decision = str(candidate_review.get("promotion_decision", "")).strip().lower()
    priority = float(candidate_review.get("priority", 0.0) or 0.0)
    if entry_recommendation == "STRONG_BUY" and reversal_risk != "HIGH":
        return True
    if (
        entry_recommendation == "BUY"
        and reversal_risk != "HIGH"
        and (
            promotion_decision == "promote"
            or priority >= 0.50
            or (entry_score >= 55.0 and rotation_score > 0.0)
        )
    ):
        return True
    if (
        entry_recommendation == "WATCH"
        and reversal_risk != "HIGH"
        and promotion_decision == "promote"
        and priority >= 0.65
        and rotation_score > 0.0
        and (momentum_5 > 0.0 or trend_confirmed)
    ):
        return True
    if (
        entry_recommendation == "WATCH"
        and reversal_risk not in {"HIGH"}
        and entry_score >= 62.0
        and rotation_score > 0.1
        and momentum_5 > 0.0
    ):
        return True
    if (
        lane == "L4"
        and entry_recommendation in {"WATCH", "BUY"}
        and reversal_risk != "HIGH"
        and (
            promotion_decision == "promote"
            or priority >= 0.45
            or entry_score >= 44.0
        )
        and (momentum_5 > 0.0 or volume_surge >= 0.35 or symbol_trending or rotation_score > 0.0)
    ):
        return True
    return False
