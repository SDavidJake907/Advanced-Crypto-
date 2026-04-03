from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from core.state.system_record import upsert_runtime_override_proposal


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_OVERRIDES_PATH = ROOT / "configs" / "runtime_overrides.json"
RUNTIME_OVERRIDE_PROPOSALS_PATH = ROOT / "configs" / "runtime_override_proposals.json"

_SETTING_SPECS: dict[str, dict[str, Any]] = {
    "AGGRESSION_MODE": {"type": str, "default": "NORMAL"},
    "MEME_UNIVERSE_ENABLED": {"type": bool, "default": True},
    "MEME_ACTIVE_UNIVERSE_MAX": {"type": int, "default": 1},
    "MEME_SYMBOLS": {"type": str, "default": ""},
    "TRADER_BASE_MOMO_THRESHOLD": {"type": float, "default": 0.0},
    "TRADER_ATR_VOLATILITY_SCALE": {"type": float, "default": 0.5},
    "TRADER_RSI_OVERBOUGHT": {"type": float, "default": 70.0},
    "TRADER_RSI_OVERSOLD": {"type": float, "default": 30.0},
    "MEME_BASE_MOMO_THRESHOLD": {"type": float, "default": -0.002},
    "MEME_ATR_VOLATILITY_SCALE": {"type": float, "default": 0.02},
    "MEME_RSI_OVERBOUGHT": {"type": float, "default": 85.0},
    "MEME_RSI_OVERSOLD": {"type": float, "default": 20.0},
    "TRADER_BB_BREAKOUT_ZSCORE": {"type": float, "default": 1.0},
    "TRADER_SMOOTHING_BARS": {"type": int, "default": 2},
    "TRADER_COOLDOWN_BARS": {"type": int, "default": 3},
    "STOP_MIN_HOLD_MIN": {"type": float, "default": 0.0},
    "REENTRY_COOLDOWN_MIN": {"type": float, "default": 240.0},
    "L1_MIN_HOLD_MIN": {"type": float, "default": 0.0},
    "L2_MIN_HOLD_MIN": {"type": float, "default": 0.0},
    "L3_MIN_HOLD_MIN": {"type": float, "default": 0.0},
    "L4_MIN_HOLD_MIN": {"type": float, "default": 0.0},
    "TRADER_PROPOSED_WEIGHT": {"type": float, "default": 0.18},
    "MEME_PROPOSED_WEIGHT": {"type": float, "default": 0.03},
    "MEME_TREND_CONFLICT_SCALE": {"type": float, "default": 0.5},
    "MEME_LANE_MIN_VOLUME_RATIO": {"type": float, "default": 1.0},
    "MEME_LANE_MIN_VOLUME_SURGE": {"type": float, "default": 0.25},
    "MEME_LANE_SOFT_MAX_SPREAD_PCT": {"type": float, "default": 4.0},
    "MEME_ENTRY_SCORE_BUY_THRESHOLD": {"type": float, "default": 48.0},
    "MEME_ENTRY_SCORE_STRONG_BUY_THRESHOLD": {"type": float, "default": 62.0},
    "EXIT_ATR_STOP_MULT": {"type": float, "default": 22.5},
    "EXIT_ATR_TAKE_PROFIT_MULT": {"type": float, "default": 5.0},
    "EXIT_PRIMARY_TP_ATR_MULT": {"type": float, "default": 1.2},
    "EXIT_MIN_STOP_PCT": {"type": float, "default": 1.5},
    "L1_EXIT_ATR_STOP_MULT": {"type": float, "default": 18.0},
    "L1_EXIT_ATR_TAKE_PROFIT_MULT": {"type": float, "default": 8.0},
    "L1_EXIT_PRIMARY_TP_ATR_MULT": {"type": float, "default": 1.8},
    "L1_EXIT_MIN_STOP_PCT": {"type": float, "default": 1.2},
    "L2_EXIT_ATR_STOP_MULT": {"type": float, "default": 16.2},
    "L2_EXIT_ATR_TAKE_PROFIT_MULT": {"type": float, "default": 6.0},
    "L2_EXIT_PRIMARY_TP_ATR_MULT": {"type": float, "default": 1.35},
    "L2_EXIT_MIN_STOP_PCT": {"type": float, "default": 1.5},
    "MEME_EXIT_MIN_STOP_PCT": {"type": float, "default": 3.0},
    "EXIT_BREAK_EVEN_R": {"type": float, "default": 3.0},
    "EXIT_TRAIL_ARM_R": {"type": float, "default": 2.0},
    "EXIT_TRAIL_ATR_MULT": {"type": float, "default": 1.5},
    "L1_BREAK_EVEN_R": {"type": float, "default": 5.0},
    "L1_TRAIL_ARM_R": {"type": float, "default": 3.0},
    "L1_TRAIL_ATR_MULT": {"type": float, "default": 2.0},
    "L2_BREAK_EVEN_R": {"type": float, "default": 3.5},
    "L2_TRAIL_ARM_R": {"type": float, "default": 2.5},
    "L2_TRAIL_ATR_MULT": {"type": float, "default": 1.5},
    "EXIT_POSTURE_USE_PHI3": {"type": bool, "default": True},
    "EXIT_LIVE_STATE_ENABLED": {"type": bool, "default": True},
    "EXIT_STALE_MIN_HOLD_MIN": {"type": float, "default": 99999.0},
    "EXIT_STALE_MAX_ABS_PNL_PCT": {"type": float, "default": 99999.0},
    "EXIT_STALE_GREEN_BLOCK": {"type": bool, "default": True},
    "EXIT_NEVER_PROFITED_MIN_HOLD_MIN": {"type": float, "default": 99999.0},
    "EXIT_FEE_AWARE_GREEN_BLOCK": {"type": bool, "default": True},
    "EXIT_GREEN_ONLY_STOP_OR_TRAIL": {"type": bool, "default": True},
    "EXIT_MIN_PROFIT_AFTER_COST_PCT": {"type": float, "default": 1.0},
    "EXIT_GREEN_EMA_RSI_ATR_HOLD": {"type": bool, "default": True},
    "EXIT_GREEN_HOLD_MIN_RSI": {"type": float, "default": 52.0},
    "EXIT_GREEN_HOLD_MIN_STOP_ATR_BUFFER": {"type": float, "default": 0.25},
    "EXIT_TIGHTEN_MIN_PNL_PCT": {"type": float, "default": 1.0},
    "EXIT_POSTURE_NEG_PNL_PCT": {"type": float, "default": -2.0},
    "EXIT_POSTURE_TIGHTEN_STOP_PCT": {"type": float, "default": 0.0075},
    "EXIT_LIVE_STALL_MIN_HOLD_MIN": {"type": float, "default": 99999.0},
    "EXIT_LIVE_STALL_MAX_PNL_PCT": {"type": float, "default": 99999.0},
    "EXIT_LIVE_SPREAD_TIGHTEN_PCT": {"type": float, "default": 0.75},
    "EXIT_LIVE_SPREAD_EXIT_PCT": {"type": float, "default": 1.1},
    "EXIT_LIVE_RANK_DECAY_MIN_PNL_PCT": {"type": float, "default": 0.25},
    "EXIT_L3_BROKEN_WEAKEN_MIN_HOLD_MIN": {"type": float, "default": 99999.0},
    "EXIT_L3_BROKEN_WEAKEN_NEG_PNL_PCT": {"type": float, "default": -99999.0},
    "EXIT_L3_BROKEN_FAIL_MIN_HOLD_MIN": {"type": float, "default": 99999.0},
    "EXIT_L3_BROKEN_FAIL_NEG_PNL_PCT": {"type": float, "default": -99999.0},
    "EXIT_L3_FRAGILE_WEAKEN_MIN_HOLD_MIN": {"type": float, "default": 99999.0},
    "EXIT_L3_FRAGILE_WEAKEN_NEG_PNL_PCT": {"type": float, "default": -99999.0},
    "L1_EXIT_TIGHTEN_MIN_PNL_PCT": {"type": float, "default": 4.0},
    "L2_EXIT_TIGHTEN_MIN_PNL_PCT": {"type": float, "default": 3.0},
    "L3_EXIT_TIGHTEN_MIN_PNL_PCT": {"type": float, "default": 2.5},
    "L4_EXIT_TIGHTEN_MIN_PNL_PCT": {"type": float, "default": 1.5},
    "MEME_EXIT_ATR_STOP_MULT": {"type": float, "default": 25.2},
    "MEME_EXIT_ATR_TAKE_PROFIT_MULT": {"type": float, "default": 7.0},
    "MEME_EXIT_PRIMARY_TP_ATR_MULT": {"type": float, "default": 1.4},
    "MEME_BREAK_EVEN_R": {"type": float, "default": 2.0},
    "MEME_TRAIL_ARM_R": {"type": float, "default": 1.2},
    "MEME_TRAIL_ATR_MULT": {"type": float, "default": 0.8},
    "MEME_EXIT_POSTURE_TIGHTEN_STOP_PCT": {"type": float, "default": 0.005},
    "ORDER_PREFERENCE": {"type": str, "default": "auto"},
    "ORDER_POST_ONLY": {"type": bool, "default": True},
    "ENTRY_AGGRESSIVE_TAKER_ENABLED": {"type": bool, "default": True},
    "ENTRY_AGGRESSIVE_TAKER_MAX_SPREAD_PCT": {"type": float, "default": 0.20},
    "ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 5.0},
    "L1_ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 6.0},
    "L2_ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 3.0},
    "L3_ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 4.0},
    "MEME_ORDER_LIMIT_OFFSET_BPS": {"type": float, "default": 8.0},
    "ORDER_OPEN_TTL_SEC": {"type": int, "default": 900},
    "ORDER_REPRICE_MAX_RETRIES": {"type": int, "default": 2},
    "ORDER_REPRICE_TTL_SEC": {"type": int, "default": 45},
    "EXIT_ORDER_REPRICE_MAX_RETRIES": {"type": int, "default": 3},
    "EXIT_ORDER_REPRICE_TTL_SEC": {"type": int, "default": 90},
    "EXIT_PROTECTIVE_ORDER_TTL_SEC": {"type": int, "default": 20},
    "EXIT_TAKE_PROFIT_ORDER_TTL_SEC": {"type": int, "default": 45},
    "ORDER_REPRICE_MAX_CHASE_BPS": {"type": float, "default": 8.0},
    "EXIT_ORDER_REPRICE_MAX_CHASE_BPS": {"type": float, "default": 12.0},
    "UNIVERSE_LOCKED": {"type": bool, "default": False},
    "CORE_ACTIVE_UNIVERSE": {"type": str, "default": "BTC/USD,ETH/USD,SOL/USD,LINK/USD"},
    "CONDITIONAL_UNIVERSE": {"type": str, "default": "XRP/USD"},
    "ENABLE_CONDITIONAL_UNIVERSE": {"type": bool, "default": True},
    "XRP_MIN_STRUCTURE_QUALITY": {"type": float, "default": 0.60},
    "XRP_MIN_NET_EDGE_PCT": {"type": float, "default": 1.0},
    "XRP_MAX_SPREAD_PCT": {"type": float, "default": 0.35},
    "XRP_MAX_DUPLICATE_CORR": {"type": float, "default": 0.85},
    "ROTATION_MARGIN_SCORE": {"type": float, "default": 6.0},
    "ROTATION_PERSIST_MIN_RANK_DELTA": {"type": int, "default": 3},
    "ROTATION_PERSIST_MIN_LANE_RANK_DELTA": {"type": int, "default": 2},
    "ROTATION_PERSIST_MIN_LEADER_URGENCY": {"type": float, "default": 6.0},
    "ROTATION_WEAK_HOLD_MIN_HOLD_MIN": {"type": float, "default": 45.0},
    "TRADE_COST_MIN_EDGE_MULT": {"type": float, "default": 2.5},
    "TRADE_COST_MIN_EXPECTED_EDGE_PCT": {"type": float, "default": 0.75},
    "TRADE_COST_SAFETY_BUFFER_PCT": {"type": float, "default": 0.15},
    "TRADE_COST_ASSUME_AGGRESSIVE_ENTRY_TAKER": {"type": bool, "default": True},
    "EXEC_MAX_SPREAD_PCT": {"type": float, "default": 1.0},
    "MEME_EXEC_MAX_SPREAD_PCT": {"type": float, "default": 1.8},
    "EXEC_MIN_NOTIONAL_USD": {"type": float, "default": 15.0},
    "MEME_EXEC_MIN_NOTIONAL_USD": {"type": float, "default": 15.0},
    "EXEC_MIN_TRADE_RISK_BUDGET_MULT": {"type": float, "default": 1.5},
    "EXEC_RISK_PER_TRADE_PCT": {"type": float, "default": 20.0},
    "L1_EXEC_RISK_PER_TRADE_PCT": {"type": float, "default": 30.0},
    "L2_EXEC_RISK_PER_TRADE_PCT": {"type": float, "default": 25.0},
    "MEME_EXEC_RISK_PER_TRADE_PCT": {"type": float, "default": 15.0},
    "EXEC_MAKER_FEE_PCT": {"type": float, "default": 0.23},
    "EXEC_TAKER_FEE_PCT": {"type": float, "default": 0.40},
    "EXEC_SLIPPAGE_ATR_MULT": {"type": float, "default": 0.10},
    "MEME_EXEC_SLIPPAGE_ATR_MULT": {"type": float, "default": 0.14},
    "RISK_MAX_POSITION_NOTIONAL": {"type": float, "default": 2000.0},
    "RISK_MAX_LEVERAGE": {"type": float, "default": 2.0},
    "PORTFOLIO_MAX_WEIGHT_PER_SYMBOL": {"type": float, "default": 0.2},
    "PORTFOLIO_MAX_TOTAL_GROSS_EXPOSURE": {"type": float, "default": 0.95},
    "PORTFOLIO_MAX_OPEN_POSITIONS": {"type": int, "default": 5},
    "PORTFOLIO_CORR_THRESHOLD": {"type": float, "default": 0.8},
    "PORTFOLIO_CORR_SCALE_DOWN": {"type": float, "default": 0.5},
    "PORTFOLIO_MAX_HIGH_CORR_SAME_DIRECTION": {"type": int, "default": 1},
    "PORTFOLIO_AVG_CORR_SCALE_THRESHOLD": {"type": float, "default": 0.7},
    "PORTFOLIO_AVG_CORR_SCALE_DOWN": {"type": float, "default": 0.6},
    "PORTFOLIO_MAX_POSITIONS_PER_SECTOR": {"type": int, "default": 2},
    "MEME_MAX_OPEN_POSITIONS": {"type": int, "default": 1},
    "NEMOTRON_TOP_CANDIDATE_COUNT": {"type": int, "default": 15},
    "NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP": {"type": bool, "default": True},
    "NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT": {"type": bool, "default": True},
    "NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP": {"type": bool, "default": True},
    "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE": {"type": float, "default": 38.0},
    "NEMOTRON_WATCH_LOW_MIN_VOLUME_RATIO": {"type": float, "default": 1.05},
    "NEMOTRON_GATE_MIN_ENTRY_SCORE": {"type": float, "default": 42.0},
    "NEMOTRON_GATE_MIN_VOLUME_RATIO": {"type": float, "default": 1.0},
    "MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE": {"type": float, "default": 38.0},
    "MEME_NEMOTRON_WATCH_MIN_VOLUME_RATIO": {"type": float, "default": 1.0},
    "MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE": {"type": float, "default": 42.0},
    "MEME_NEMOTRON_GATE_MIN_VOLUME_RATIO": {"type": float, "default": 0.95},
    "LEADER_URGENCY_OVERRIDE_THRESHOLD": {"type": float, "default": 0.35},
    "NEMOTRON_VERDICT_CACHE_TTL_SEC": {"type": float, "default": 90.0},
    "ADVISORY_MIN_ENTRY_SCORE": {"type": float, "default": 45.0},
    "MARKET_TREND_BEAR_SCORE_PENALTY": {"type": float, "default": 8.0},
    "MARKET_TREND_BULL_SCORE_BOOST": {"type": float, "default": 4.0},
    "ADVISORY_MIN_VOLUME_RATIO": {"type": float, "default": 1.1},
    "STABILIZATION_STRICT_ENTRY_ENABLED": {"type": bool, "default": False},
    "STABILIZATION_ALLOWED_LANES": {"type": str, "default": "L2,L3"},
    "STABILIZATION_MIN_ENTRY_SCORE": {"type": float, "default": 70.0},
    "STABILIZATION_MIN_NET_EDGE_PCT": {"type": float, "default": 0.0},
    "STABILIZATION_REQUIRE_TP_AFTER_COST_VALID": {"type": bool, "default": True},
    "STABILIZATION_REQUIRE_TREND_CONFIRMED": {"type": bool, "default": True},
    "STABILIZATION_REQUIRE_SHORT_TF_READY_15M": {"type": bool, "default": True},
    "STABILIZATION_BLOCK_RANGING_MARKET": {"type": bool, "default": True},
    "STABILIZATION_REQUIRE_BUY_RECOMMENDATION": {"type": bool, "default": True},
}

_AGGRESSION_ALLOWED_KEYS = {
    "NEMOTRON_TOP_CANDIDATE_COUNT",
    "ADVISORY_MIN_ENTRY_SCORE",
    "ADVISORY_MIN_VOLUME_RATIO",
    "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE",
    "NEMOTRON_GATE_MIN_ENTRY_SCORE",
    "MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE",
    "MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE",
    "TRADER_COOLDOWN_BARS",
    "TRADER_PROPOSED_WEIGHT",
    "MEME_PROPOSED_WEIGHT",
}

_AGGRESSION_PROFILES: dict[str, dict[str, Any]] = {
    "DEFENSIVE": {
        "NEMOTRON_TOP_CANDIDATE_COUNT": ("add", -4, 5, 40),
        "ADVISORY_MIN_ENTRY_SCORE": ("add", 4.0, 40.0, 95.0),
        "ADVISORY_MIN_VOLUME_RATIO": ("add", 0.05, 1.0, 2.0),
        "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE": ("add", 4.0, 30.0, 95.0),
        "NEMOTRON_GATE_MIN_ENTRY_SCORE": ("add", 4.0, 30.0, 95.0),
        "MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE": ("add", 4.0, 25.0, 95.0),
        "MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE": ("add", 4.0, 25.0, 95.0),
        "TRADER_COOLDOWN_BARS": ("add", 1, 0, 20),
        "TRADER_PROPOSED_WEIGHT": ("scale", 0.9, 0.01, 0.15),
        "MEME_PROPOSED_WEIGHT": ("scale", 0.9, 0.005, 0.05),
    },
    "NORMAL": {},
    "OFFENSIVE": {
        "NEMOTRON_TOP_CANDIDATE_COUNT": ("add", 4, 5, 40),
        "ADVISORY_MIN_ENTRY_SCORE": ("add", -3.0, 40.0, 95.0),
        "ADVISORY_MIN_VOLUME_RATIO": ("add", -0.05, 1.0, 2.0),
        "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE": ("add", -3.0, 30.0, 95.0),
        "NEMOTRON_GATE_MIN_ENTRY_SCORE": ("add", -2.0, 30.0, 95.0),
        "MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE": ("add", -3.0, 25.0, 95.0),
        "MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE": ("add", -2.0, 25.0, 95.0),
        "TRADER_COOLDOWN_BARS": ("add", -1, 0, 20),
        "TRADER_PROPOSED_WEIGHT": ("scale", 1.1, 0.01, 0.12),
        "MEME_PROPOSED_WEIGHT": ("scale", 1.1, 0.005, 0.04),
    },
    "HIGH_OFFENSIVE": {
        "NEMOTRON_TOP_CANDIDATE_COUNT": ("add", 8, 5, 40),
        "ADVISORY_MIN_ENTRY_SCORE": ("add", -5.0, 40.0, 95.0),
        "ADVISORY_MIN_VOLUME_RATIO": ("add", -0.1, 1.0, 2.0),
        "NEMOTRON_WATCH_LOW_MIN_ENTRY_SCORE": ("add", -5.0, 30.0, 95.0),
        "NEMOTRON_GATE_MIN_ENTRY_SCORE": ("add", -4.0, 30.0, 95.0),
        "MEME_NEMOTRON_WATCH_MIN_ENTRY_SCORE": ("add", -4.0, 25.0, 95.0),
        "MEME_NEMOTRON_GATE_MIN_ENTRY_SCORE": ("add", -4.0, 25.0, 95.0),
        "TRADER_COOLDOWN_BARS": ("add", -2, 0, 20),
        "TRADER_PROPOSED_WEIGHT": ("scale", 1.2, 0.01, 0.14),
        "MEME_PROPOSED_WEIGHT": ("scale", 1.2, 0.005, 0.045),
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_value(name: str, value: Any) -> Any:
    if name not in _SETTING_SPECS:
        raise KeyError(f"Unknown runtime setting: {name}")
    target_type = _SETTING_SPECS[name]["type"]
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if name == "AGGRESSION_MODE":
        mode = str(value).strip().upper()
        if mode not in _AGGRESSION_PROFILES:
            raise ValueError(f"Invalid AGGRESSION_MODE: {value}")
        return mode
    return target_type(value)


def get_aggression_mode() -> str:
    return str(get_runtime_setting("AGGRESSION_MODE")).strip().upper()


def _apply_aggression_profile(name: str, value: Any) -> Any:
    mode = get_aggression_mode()
    if mode == "NORMAL" or name not in _AGGRESSION_ALLOWED_KEYS:
        return value
    profile = _AGGRESSION_PROFILES.get(mode, {})
    if name not in profile:
        return value
    operation, delta, low, high = profile[name]
    if operation == "add":
        adjusted = value + delta
    elif operation == "scale":
        adjusted = value * delta
    else:
        return value
    if isinstance(value, int):
        return int(max(low, min(high, round(adjusted))))
    return max(low, min(high, float(adjusted)))


def load_runtime_overrides() -> dict[str, Any]:
    if not RUNTIME_OVERRIDES_PATH.exists():
        return {}
    payload = json.loads(RUNTIME_OVERRIDES_PATH.read_text(encoding="utf-8"))
    overrides = payload.get("overrides", payload)
    if not isinstance(overrides, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _SETTING_SPECS:
            sanitized[key] = _coerce_value(key, value)
    return sanitized


def save_runtime_overrides(overrides: dict[str, Any]) -> None:
    RUNTIME_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now_iso(),
        "overrides": {key: _coerce_value(key, value) for key, value in overrides.items()},
    }
    RUNTIME_OVERRIDES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_runtime_overrides(updates: dict[str, Any]) -> dict[str, Any]:
    current = load_runtime_overrides()
    for key, value in updates.items():
        current[key] = _coerce_value(key, value)
    save_runtime_overrides(current)
    return current


def load_runtime_override_proposals() -> list[dict[str, Any]]:
    if not RUNTIME_OVERRIDE_PROPOSALS_PATH.exists():
        return []
    payload = json.loads(RUNTIME_OVERRIDE_PROPOSALS_PATH.read_text(encoding="utf-8"))
    proposals = payload.get("proposals", payload)
    if not isinstance(proposals, list):
        return []
    items: list[dict[str, Any]] = []
    for proposal in proposals:
        if isinstance(proposal, dict):
            items.append(proposal)
    return items


def save_runtime_override_proposals(proposals: list[dict[str, Any]]) -> None:
    RUNTIME_OVERRIDE_PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now_iso(),
        "proposals": proposals,
    }
    RUNTIME_OVERRIDE_PROPOSALS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stage_runtime_override_proposal(
    updates: dict[str, Any],
    *,
    source: str,
    summary: str = "",
    validation: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sanitized = {key: _coerce_value(key, value) for key, value in updates.items() if key in _SETTING_SPECS}
    proposal_validation = {
        "replay_passed": bool((validation or {}).get("replay_passed", False)),
        "shadow_passed": bool((validation or {}).get("shadow_passed", False)),
        "human_approved": bool((validation or {}).get("human_approved", False)),
    }
    proposal = {
        "id": f"rtovr-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "created_at": _now_iso(),
        "source": str(source).strip() or "unknown",
        "summary": str(summary or "").strip(),
        "updates": sanitized,
        "validation": proposal_validation,
        "context": context or {},
        "status": "pending",
    }
    proposals = load_runtime_override_proposals()
    proposals.append(proposal)
    save_runtime_override_proposals(proposals)
    upsert_runtime_override_proposal(proposal)
    return proposal


def _proposal_ready_for_apply(proposal: dict[str, Any]) -> bool:
    validation = proposal.get("validation", {}) if isinstance(proposal.get("validation", {}), dict) else {}
    return bool(validation.get("replay_passed")) and bool(validation.get("shadow_passed")) and bool(validation.get("human_approved"))


def apply_runtime_override_proposal(
    proposal_id: str,
    *,
    approved_by: str,
    approval_note: str = "",
) -> dict[str, Any]:
    proposals = load_runtime_override_proposals()
    for proposal in proposals:
        if proposal.get("id") != proposal_id:
            continue
        if proposal.get("status") == "applied":
            return proposal
        if not _proposal_ready_for_apply(proposal):
            raise ValueError("runtime override proposal missing replay/shadow/human approval")
        if not str(approved_by).strip():
            raise ValueError("approved_by is required to apply runtime override proposal")
        applied = update_runtime_overrides(proposal.get("updates", {}))
        proposal["status"] = "applied"
        proposal["applied_at"] = _now_iso()
        proposal["applied_by"] = str(approved_by).strip()
        if approval_note:
            proposal["approval_note"] = str(approval_note)
        proposal["applied_snapshot"] = applied
        save_runtime_override_proposals(proposals)
        upsert_runtime_override_proposal(proposal)
        return proposal
    raise KeyError(f"Unknown runtime override proposal: {proposal_id}")


def get_runtime_setting(name: str, default: Any | None = None) -> Any:
    if name not in _SETTING_SPECS:
        raise KeyError(f"Unknown runtime setting: {name}")
    overrides = load_runtime_overrides()
    if name in overrides:
        value = overrides[name]
    elif name in os.environ:
        value = _coerce_value(name, os.environ[name])
    elif default is not None:
        value = default
    else:
        value = _SETTING_SPECS[name]["default"]
    if name == "AGGRESSION_MODE":
        return value
    return _apply_aggression_profile(name, value)


def get_runtime_snapshot() -> dict[str, Any]:
    overrides = load_runtime_overrides()
    values: dict[str, Any] = {}
    base_values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, spec in _SETTING_SPECS.items():
        if name in overrides:
            base_values[name] = overrides[name]
            sources[name] = "override"
        elif name in os.environ:
            base_values[name] = _coerce_value(name, os.environ[name])
            sources[name] = "env"
        else:
            base_values[name] = spec["default"]
            sources[name] = "default"
        values[name] = _apply_aggression_profile(name, base_values[name]) if name != "AGGRESSION_MODE" else base_values[name]
    return {
        "path": str(RUNTIME_OVERRIDES_PATH.relative_to(ROOT)),
        "values": values,
        "base_values": base_values,
        "sources": sources,
        "overrides": overrides,
        "aggression_mode": get_aggression_mode(),
    }


def get_cooldown_bars() -> int:
    return int(get_runtime_setting("TRADER_COOLDOWN_BARS"))


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def is_meme_symbol(symbol: str) -> bool:
    raw = str(get_runtime_setting("MEME_SYMBOLS"))
    meme_symbols = {_normalize_symbol(item) for item in raw.split(",") if item.strip()}
    return _normalize_symbol(symbol) in meme_symbols


def get_symbol_lane(symbol: str) -> str:
    return "L4" if is_meme_symbol(symbol) else "L3"


def is_meme_lane(lane: str | None) -> bool:
    return str(lane or "").upper() == "L4"


def get_proposed_weight(symbol: str | None = None, lane: str | None = None) -> float:
    resolved_lane = lane or (get_symbol_lane(symbol) if symbol is not None else None)
    if symbol is not None and is_meme_lane(resolved_lane):
        return float(get_runtime_setting("MEME_PROPOSED_WEIGHT"))
    return float(get_runtime_setting("TRADER_PROPOSED_WEIGHT"))
