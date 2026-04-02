from __future__ import annotations

from typing import Any

from core.config.runtime import get_runtime_setting
from core.risk.portfolio import PositionState
from core.state.store import load_universe


def _get_leader_metrics(symbol: str, universe_context: dict[str, Any]) -> tuple[float, bool]:
    """Extract leader metrics for a symbol from cached candidate lists."""
    for cand_list in (
        universe_context.get("top_scored") or [],
        universe_context.get("hot_candidates") or [],
        universe_context.get("lane_shortlists") or {},
    ):
        if isinstance(cand_list, dict):
            iterable = cand_list.values()
        else:
            iterable = cand_list
        for item in iterable:
            if not isinstance(item, dict):
                continue
            if str(item.get("symbol", "")).upper() == str(symbol).upper():
                return float(item.get("leader_urgency", 0.0) or 0.0), bool(item.get("leader_takeover", False))
    return 0.0, False


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


def _is_strong_mover_candidate(features: dict[str, Any]) -> bool:
    """Quick heuristic to recognize aggression-worthy leaders."""
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    leader_urgency = float(features.get("leader_urgency", 0.0) or 0.0)
    return (
        entry_score >= 60.0
        or rotation_score > 0.2
        or momentum_5 > 0.008
        or volume_surge >= 0.4
        or leader_urgency >= 6.0
    )


def _effective_net_edge_pct(features: dict[str, Any]) -> float:
    if "net_edge_pct" in features:
        return float(features.get("net_edge_pct", 0.0) or 0.0)
    point_breakdown = features.get("point_breakdown") or {}
    if isinstance(point_breakdown, dict):
        return float(point_breakdown.get("net_edge_pct", 0.0) or 0.0)
    return 0.0


def _is_range_safe_candidate(features: dict[str, Any]) -> bool:
    lane = str(features.get("lane", "L3") or "L3").upper()
    structure_quality = float(features.get("structure_quality", 0.0) or 0.0)
    continuation_quality = float(features.get("continuation_quality", 0.0) or 0.0)
    trade_quality = float(features.get("trade_quality", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    volume_ratio = float(features.get("volume_ratio", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    short_tf_ready_15m = bool(features.get("short_tf_ready_15m", False))
    range_breakout_1h = bool(features.get("range_breakout_1h", False))
    pullback_hold = bool(features.get("pullback_hold", False))
    tp_after_cost_valid = bool(features.get("tp_after_cost_valid", False))
    net_edge_pct = _effective_net_edge_pct(features)

    if lane == "L2":
        return bool(
            short_tf_ready_15m
            and tp_after_cost_valid
            and net_edge_pct > 0.0
            and momentum_5 > 0.0
            and volume_ratio >= 0.95
            and volume_surge >= 0.15
            and trade_quality >= 55.0
            and structure_quality >= 60.0
            and continuation_quality >= 60.0
            and (range_breakout_1h or pullback_hold or continuation_quality >= 66.0)
        )

    if lane == "L3":
        return bool(
            short_tf_ready_15m
            and tp_after_cost_valid
            and net_edge_pct > 0.0
            and momentum_5 > 0.0
            and volume_ratio >= 1.0
            and trade_quality >= 58.0
            and structure_quality >= 64.0
            and continuation_quality >= 64.0
            and (range_breakout_1h or pullback_hold)
        )

    return True


def _passes_market_state_entry_gate(features: dict[str, Any]) -> tuple[bool, str]:
    lane = str(features.get("lane", "L3") or "L3").upper()
    if lane not in {"L2", "L3"}:
        return True, "market_state_not_applicable"

    trend_1h = int(features.get("trend_1h", 0) or 0)
    trend_confirmed = bool(features.get("trend_confirmed", False))
    ranging_market = bool(features.get("ranging_market", False))
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    range_breakout_1h = bool(features.get("range_breakout_1h", False))
    pullback_hold = bool(features.get("pullback_hold", False))

    if trend_1h < 0 and momentum_14 <= 0.0:
        if not (_is_strong_mover_candidate(features) and _is_range_safe_candidate(features)):
            return False, "market_state_downtrend_block"
        return True, "market_state_downtrend_override"

    if not trend_confirmed:
        if ranging_market:
            if not (_is_strong_mover_candidate(features) and _is_range_safe_candidate(features)):
                return False, "market_state_ranging_unconfirmed_block"
            return True, "market_state_ranging_override"

        if momentum_5 <= 0.0 and not (range_breakout_1h or pullback_hold):
            return False, "market_state_transition_weak_block"

    return True, "market_state_ok"


def _base_promotion_tier(features: dict[str, Any]) -> str:
    tier = str(features.get("promotion_tier", "") or "").lower()
    if tier in {"skip", "probe", "promote"}:
        return tier
    entry_recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
    reversal_risk = str(features.get("reversal_risk", "MEDIUM") or "MEDIUM").upper()
    if entry_recommendation in {"BUY", "STRONG_BUY"} and reversal_risk != "HIGH":
        return "promote"
    return "skip"


def _strict_stabilization_gate(features: dict[str, Any]) -> tuple[bool, str]:
    if not bool(get_runtime_setting("STABILIZATION_STRICT_ENTRY_ENABLED")):
        return True, "disabled"

    lane = str(features.get("lane", "L3") or "L3").upper()
    allowed_lanes = {
        chunk.strip().upper()
        for chunk in str(get_runtime_setting("STABILIZATION_ALLOWED_LANES") or "").split(",")
        if chunk.strip()
    }
    if allowed_lanes and lane not in allowed_lanes:
        return False, f"stabilization_lane_block({lane})"

    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    min_entry_score = float(get_runtime_setting("STABILIZATION_MIN_ENTRY_SCORE"))
    if entry_score < min_entry_score:
        return False, f"stabilization_entry_score({entry_score:.1f}<{min_entry_score:.1f})"

    if bool(get_runtime_setting("STABILIZATION_REQUIRE_BUY_RECOMMENDATION")):
        entry_recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
        if entry_recommendation not in {"BUY", "STRONG_BUY"}:
            return False, f"stabilization_recommendation_block({entry_recommendation})"

    if bool(get_runtime_setting("STABILIZATION_REQUIRE_TREND_CONFIRMED")) and not bool(features.get("trend_confirmed", False)):
        return False, "stabilization_trend_unconfirmed"

    if bool(get_runtime_setting("STABILIZATION_REQUIRE_SHORT_TF_READY_15M")) and not bool(features.get("short_tf_ready_15m", False)):
        return False, "stabilization_15m_not_ready"

    if bool(get_runtime_setting("STABILIZATION_BLOCK_RANGING_MARKET")) and bool(features.get("ranging_market", False)):
        return False, "stabilization_ranging_market"

    if bool(get_runtime_setting("STABILIZATION_REQUIRE_TP_AFTER_COST_VALID")) and not bool(features.get("tp_after_cost_valid", False)):
        return False, "stabilization_tp_cost_invalid"

    net_edge_pct = float(features.get("net_edge_pct", 0.0) or 0.0)
    min_net_edge_pct = float(get_runtime_setting("STABILIZATION_MIN_NET_EDGE_PCT"))
    if net_edge_pct < min_net_edge_pct:
        return False, f"stabilization_net_edge({net_edge_pct:.2f}<{min_net_edge_pct:.2f})"

    return True, "passed"


def symbol_in_top_candidates(
    symbol: str,
    positions_state: PositionState,
    features: dict[str, Any],
    market_state_review: dict[str, Any] | None = None,
) -> bool:
    if positions_state.get(symbol) is not None:
        return True
    if str(features.get("promotion_reason", "") or "") == "falling_short_structure":
        return False
    context = load_universe_candidate_context(symbol)
    leader_urgency = float(features.get("leader_urgency", 0.0) or 0.0)
    leader_takeover = bool(features.get("leader_takeover", False))
    leader_urgency_override = max(float(get_runtime_setting("LEADER_URGENCY_OVERRIDE_THRESHOLD")), 6.0)
    if leader_takeover or leader_urgency >= leader_urgency_override:
        return True
    if context.get("current_symbol_is_top_candidate"):
        return True

    # Elite structure bypass — channel_breakout or channel_retest with EMA aligned
    # always gets through regardless of rank. These are the exact setups we built
    # Wave 2 to catch — don't let the top-N gate kill them.
    _promotion_reason = str(features.get("promotion_reason", "") or "")
    _ema_aligned = bool(features.get("ema9_above_ema20", False))
    _breakout = bool(features.get("range_breakout_1h", False))
    _pullback = bool(features.get("pullback_hold", False))
    if _ema_aligned and (_breakout or _pullback):
        return True
    if _promotion_reason in {"channel_breakout", "channel_retest"}:
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

    if bool(get_runtime_setting("NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP")):
        if entry_recommendation in {"BUY", "STRONG_BUY"} and reversal_risk == "LOW":
            return True

    if bool(get_runtime_setting("NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP")):
        if (
            entry_recommendation in {"BUY", "STRONG_BUY"}
            and reversal_risk == "MEDIUM"
            and rotation_score > 0.0
            and entry_score >= 58.0
            and volume_ratio >= 1.0
        ):
            return True

    if bool(get_runtime_setting("NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT")):
        if (
            entry_recommendation == "WATCH"
            and reversal_risk == "LOW"
            and _base_promotion_tier(features) == "probe"
            and entry_score >= max(min_watch_score, 48.0)
            and volume_ratio >= max(min_watch_volume_ratio, 0.9)
            and (
                lane_conflict
                or leader_takeover
                or leader_urgency >= leader_urgency_override
            )
            and (
                rotation_score > 0.0
                or momentum_5 > 0.0
                or volume_ratio >= max(min_watch_volume_ratio, 0.95)
            )
        ):
            return True

    if lane == "L4":
        if (
            entry_recommendation in {"WATCH", "BUY", "STRONG_BUY"}
            and reversal_risk != "HIGH"
            and (
                entry_score >= max(min_watch_score, 50.0)
                or volume_ratio >= max(min_watch_volume_ratio, 0.95)
                or volume_surge >= 0.50   # 50% above baseline = volume_surge_flag threshold
                or symbol_trending
            )
            and (momentum_5 > 0.0 or rotation_score > 0.0 or volume_surge >= 0.50)
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
        return True, "held_position"

    if not bool(features.get("indicators_ready", True)):
        return False, "feature_warmup"

    # Hard entry-score and volume-ratio gate — tunable via runtime overrides.
    # Held positions bypass above; everything else must clear this floor before
    # promotion-tier logic is even evaluated.
    _lane = str(features.get("lane", "L3") or "L3").upper()
    if _lane == "L4":
        _min_score = float(get_runtime_setting("MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE"))
        _min_vol   = float(get_runtime_setting("MEME_NEMOTRON_GATE_MIN_VOLUME_RATIO"))
    else:
        _min_score = float(get_runtime_setting("NEMOTRON_GATE_MIN_ENTRY_SCORE"))
        _min_vol   = float(get_runtime_setting("NEMOTRON_GATE_MIN_VOLUME_RATIO"))

    _entry_score  = float(features.get("entry_score",  0.0) or 0.0)
    _volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)

    if _entry_score < _min_score:
        return False, f"entry_score_below_gate({_entry_score:.1f}<{_min_score:.1f})"
    if _volume_ratio < _min_vol:
        return False, f"volume_ratio_below_gate({_volume_ratio:.2f}<{_min_vol:.2f})"
    _net_edge_pct = _effective_net_edge_pct(features)
    if _net_edge_pct <= 0.0:
        return False, f"net_edge_non_positive({_net_edge_pct:.2f})"

    market_state_ok, market_state_reason = _passes_market_state_entry_gate(features)
    if not market_state_ok:
        return False, market_state_reason

    stabilization_ok, stabilization_reason = _strict_stabilization_gate(features)
    if not stabilization_ok:
        return False, stabilization_reason

    promotion_tier = _base_promotion_tier(features)
    promotion_reason = str(features.get("promotion_reason", "not_qualified") or "not_qualified")
    lane = str(features.get("lane", "L3") or "L3").upper()
    ranging_market = bool(features.get("ranging_market", False))
    trend_confirmed = bool(features.get("trend_confirmed", False))

    if promotion_reason == "falling_short_structure":
        return False, promotion_reason

    if ranging_market and not trend_confirmed and lane in {"L2", "L3"}:
        if not (_is_strong_mover_candidate(features) and _is_range_safe_candidate(features)):
            return False, "ranging_unconfirmed_structure_weak"

    if promotion_tier == "promote":
        return True, "passed"

    if promotion_tier == "probe":
        if ranging_market and lane in {"L1", "L3"} and not _is_strong_mover_candidate(features):
            return False, "ranging_market_weak"
        return True, "passed"

    if ranging_market and not _is_strong_mover_candidate(features):
        return False, "ranging_market_weak"

    return False, promotion_reason


def should_run_nemotron(*, symbol: str, features: dict[str, Any], positions_state: PositionState, universe_context: dict[str, Any]) -> bool:
    if str(features.get("promotion_reason", "") or "") == "falling_short_structure":
        return False
    passed, _ = passes_deterministic_candidate_gate(
        symbol=symbol,
        positions_state=positions_state,
        features=features,
        universe_context=universe_context,
    )
    if passed:
        return True

    if positions_state.get(symbol) is not None:
        return True

    promotion_tier = _base_promotion_tier(features)
    return promotion_tier == "probe"
