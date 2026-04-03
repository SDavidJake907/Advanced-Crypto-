from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.data.kraken_rest import KrakenRestClient
from core.data.dexscreener import DexScreenerFeed
from core.data.news_sentiment import NewsSentimentFeed
from core.features.momentum import risk_adjusted_momentum
from core.llm.phi3_scan import ScanCandidate, phi3_scan_market, phi3_supervise_lanes
from core.policy.pipeline import apply_policy_pipeline
from core.policy.universe_policy import UniversePolicy, apply_churn_threshold, clamp_active_size
from core.risk.fee_filter import evaluate_trade_cost
from core.symbols import normalize_symbol, to_kraken_symbol
from core.state.age_cache import load_age_cache, save_age_cache
from core.config.runtime import get_runtime_setting, is_meme_symbol
from core.state.store import load_synced_position_symbols, load_universe, save_universe
from core.state.system_record import SYSTEM_RECORD_DB_PATH


@dataclass
class Candidate:
    pair: str
    score: float
    volume_usd: float
    last: float
    momentum_5: float = 0.0
    momentum_14: float = 0.0
    momentum_30: float = 0.0
    # Extended multi-timeframe momentum profile
    m5m: float = 0.0    # 5-minute
    m30m: float = 0.0   # 30-minute
    m4h: float = 0.0    # 4-hour
    m8h: float = 0.0    # 8-hour
    m24h: float = 0.0   # 24-hour (daily)
    m7d: float = 0.0    # 7-day
    m30d: float = 0.0   # 30-day
    momentum_alignment: int = 0       # 0-7: timeframes pointing up (confirmation count)
    momentum_acceleration: float = 0.0  # short minus long divergence (breakout signal)
    trend_1h: int = 0
    volume_ratio: float = 1.0
    price_zscore: float = 0.0
    rsi: float = 50.0
    spread_bps: float = 0.0
    structure_quality: float = 50.0
    momentum_quality: float = 50.0
    volume_quality: float = 50.0
    trade_quality: float = 50.0
    market_support: float = 50.0
    continuation_quality: float = 50.0
    risk_quality: float = 50.0
    lane: str = "L3"
    candidate_score: float = 0.0
    candidate_recommendation: str = "WATCH"
    candidate_risk: str = "MEDIUM"
    candidate_reasons: list[str] | None = None
    vwrs: float = 0.0  # Volume-weighted relative strength vs BTC on 5h window
    rank_score: float = 0.0
    previous_rank: int = 999
    rank_delta: int = 0
    previous_lane_rank: int = 999
    lane_rank_delta: int = 0
    momentum_delta: float = 0.0
    volume_ratio_delta: float = 0.0
    leader_urgency: float = 0.0
    leader_takeover: bool = False
    atr_pct: float = 0.0
    expected_move_pct: float = 1.5
    total_cost_pct: float = 0.7
    required_edge_pct: float = 1.0
    net_edge_pct: float = 0.8
    tp_after_cost_valid: bool = True
    realized_cost_penalty_pct: float = 0.0
    realized_slippage_pct: float = 0.0
    realized_follow_through_pct: float = 0.0
    golden_profile_score: float = 0.0
    golden_profile_tag: str = "neutral"
    bullish_divergence: bool = False
    bearish_divergence: bool = False
    divergence_strength: float = 0.0
    divergence_age_bars: int = 99
    final_score: float = 0.0
    reliability_bonus: float = 0.0
    basket_fit_bonus: float = 0.0
    score_breakdown: dict[str, Any] | None = None


def _scan_quality_ok(candidate: Candidate) -> bool:
    recommendation = str(candidate.candidate_recommendation or "WATCH").upper()
    risk = str(candidate.candidate_risk or "MEDIUM").upper()
    if recommendation == "AVOID" or risk == "HIGH":
        return False
    if candidate.candidate_score < 52.0:
        return False
    if candidate.volume_ratio < 0.75:
        return False
    if candidate.spread_bps > 20.0:
        return False
    if candidate.trade_quality < 52.0 or candidate.risk_quality < 50.0:
        return False
    if not candidate.tp_after_cost_valid:
        return False
    if candidate.net_edge_pct < 0.35:
        return False
    return True


def _golden_profile_score(candidate: Candidate) -> float:
    score = 0.0
    if candidate.tp_after_cost_valid:
        score += 4.0
    else:
        score -= 10.0

    score += min(max(candidate.net_edge_pct - 0.35, 0.0) * 7.0, 8.0)
    score -= max(candidate.total_cost_pct - 1.25, 0.0) * 4.0

    if candidate.spread_bps <= 8.0:
        score += 5.0
    elif candidate.spread_bps <= 12.0:
        score += 2.5
    elif candidate.spread_bps > 18.0:
        score -= 4.0

    if candidate.volume_ratio >= 1.0:
        score += min((candidate.volume_ratio - 1.0) * 8.0, 5.0)
    elif candidate.volume_ratio >= 0.85:
        score += 1.5
    else:
        score -= 3.0

    score += max(candidate.trade_quality - 70.0, 0.0) * 0.18
    score += max(candidate.continuation_quality - 68.0, 0.0) * 0.22
    score += max(candidate.risk_quality - 68.0, 0.0) * 0.14
    score += max(candidate.structure_quality - 60.0, 0.0) * 0.12

    if candidate.trend_1h > 0:
        score += 3.5
    if candidate.momentum_alignment >= 3:
        score += 4.0
    if candidate.momentum_5 > 0.0 and candidate.momentum_14 > 0.0:
        score += 2.0
    if candidate.momentum_acceleration > 0.0:
        score += min(candidate.momentum_acceleration * 250.0, 3.0)

    if 48.0 <= candidate.rsi <= 64.0:
        score += 1.5
    elif candidate.rsi >= 72.0:
        score -= 2.5

    if candidate.realized_follow_through_pct > 0.0:
        score += min(candidate.realized_follow_through_pct * 2.5, 4.0)
    if candidate.realized_cost_penalty_pct > 0.0:
        score -= min(candidate.realized_cost_penalty_pct * 2.0, 4.0)

    if candidate.lane == "L2":
        score += 2.0
    elif candidate.lane == "L1":
        score += 1.5
    elif candidate.lane == "L4":
        score -= 2.0

    return round(score, 3)


def _golden_profile_tag(candidate: Candidate) -> str:
    if (
        candidate.golden_profile_score >= 18.0
        and candidate.net_edge_pct >= 0.5
        and candidate.spread_bps <= 12.0
        and candidate.trade_quality >= 75.0
        and candidate.continuation_quality >= 70.0
        and candidate.risk_quality >= 68.0
    ):
        return "golden"
    if (
        candidate.golden_profile_score >= 10.0
        and candidate.net_edge_pct >= 0.35
        and candidate.spread_bps <= 16.0
        and candidate.trade_quality >= 65.0
    ):
        return "near_golden"
    if candidate.golden_profile_score <= 2.0:
        return "fragile"
    return "neutral"


def _simple_atr_pct(rows: list[list[str | float]], lookback: int = 14) -> float:
    if len(rows) < lookback + 1:
        return 0.0
    recent = rows[-(lookback + 1) :]
    trs: list[float] = []
    prev_close = float(recent[0][4])
    for row in recent[1:]:
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(max(tr, 0.0))
        prev_close = close
    if not trs:
        return 0.0
    last_close = float(recent[-1][4])
    if last_close <= 0.0:
        return 0.0
    return (sum(trs) / len(trs)) / last_close * 100.0


def _cost_penalty_points_from_net_edge(net_edge_pct: float) -> float:
    if net_edge_pct >= 1.0:
        return -5.0
    if net_edge_pct >= 0.0:
        return 0.0
    if net_edge_pct >= -0.5:
        return 5.0
    if net_edge_pct >= -1.0:
        return 10.0
    return 15.0


def _candidate_final_score_features(candidate: Candidate) -> dict[str, Any]:
    return {
        "symbol": candidate.pair,
        "lane": candidate.lane,
        "entry_score": float(candidate.candidate_score),
        "spread_pct": float(candidate.spread_bps / 100.0),
        "bullish_divergence": bool(candidate.bullish_divergence),
        "bearish_divergence": bool(candidate.bearish_divergence),
        "divergence_strength": float(candidate.divergence_strength),
        "divergence_age_bars": int(candidate.divergence_age_bars),
        "point_breakdown": {
            "cost_penalty_pts": _cost_penalty_points_from_net_edge(float(candidate.net_edge_pct)),
            "net_edge_pct": float(candidate.net_edge_pct),
        },
    }


def _apply_candidate_final_score(
    candidate: Candidate,
    *,
    reliability_map: dict[str, Any] | None = None,
) -> Candidate:
    from core.policy.candidate_packet import compute_candidate_economics

    final = compute_candidate_economics(
        _candidate_final_score_features(candidate),
        reliability_map=reliability_map or {},
    )
    candidate.final_score = float(final.final_score)
    candidate.reliability_bonus = float(final.reliability_bonus)
    candidate.basket_fit_bonus = float(final.basket_fit_bonus)
    candidate.score_breakdown = dict(final.score_breakdown)
    return candidate


def _load_recent_execution_feedback(symbols: list[str], *, fill_limit: int = 400, outcome_limit: int = 400) -> dict[str, dict[str, float]]:
    symbol_set = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    if not symbol_set or not SYSTEM_RECORD_DB_PATH.exists():
        return {}
    feedback: dict[str, dict[str, float]] = {}
    try:
        conn = sqlite3.connect(SYSTEM_RECORD_DB_PATH)
    except Exception:
        return {}
    try:
        rows = conn.execute(
            "SELECT symbol, payload_json FROM fill_events ORDER BY id DESC LIMIT ?",
            (fill_limit,),
        ).fetchall()
        for symbol, payload_json in rows:
            symbol_key = str(symbol or "").strip().upper()
            if symbol_key not in symbol_set:
                continue
            try:
                payload = json.loads(payload_json)
            except Exception:
                continue
            cost = payload.get("cost", {})
            if not isinstance(cost, dict):
                continue
            stats = feedback.setdefault(symbol_key, {"cost_sum": 0.0, "slippage_sum": 0.0, "fill_count": 0.0, "follow_sum": 0.0, "follow_count": 0.0})
            stats["cost_sum"] += float(cost.get("total_cost_pct", 0.0) or 0.0)
            stats["slippage_sum"] += float(cost.get("slippage_pct", 0.0) or 0.0)
            stats["fill_count"] += 1.0

        rows = conn.execute(
            "SELECT symbol, pnl_pct FROM outcome_reviews ORDER BY id DESC LIMIT ?",
            (outcome_limit,),
        ).fetchall()
        for symbol, pnl_pct in rows:
            symbol_key = str(symbol or "").strip().upper()
            if symbol_key not in symbol_set:
                continue
            stats = feedback.setdefault(symbol_key, {"cost_sum": 0.0, "slippage_sum": 0.0, "fill_count": 0.0, "follow_sum": 0.0, "follow_count": 0.0})
            stats["follow_sum"] += float(pnl_pct or 0.0) * 100.0
            stats["follow_count"] += 1.0
    except Exception:
        return {}
    finally:
        conn.close()

    reduced: dict[str, dict[str, float]] = {}
    for symbol, stats in feedback.items():
        fill_count = max(stats.get("fill_count", 0.0), 0.0)
        follow_count = max(stats.get("follow_count", 0.0), 0.0)
        reduced[symbol] = {
            "avg_total_cost_pct": stats["cost_sum"] / fill_count if fill_count else 0.0,
            "avg_slippage_pct": stats["slippage_sum"] / fill_count if fill_count else 0.0,
            "avg_follow_through_pct": stats["follow_sum"] / follow_count if follow_count else 0.0,
        }
    return reduced


def _previous_rank_maps() -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, float]]]:
    universe = load_universe()
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    prev_top_scored = meta.get("top_scored", []) if isinstance(meta.get("top_scored", []), list) else []
    prev_lane_shortlists = meta.get("lane_shortlists", {}) if isinstance(meta.get("lane_shortlists", {}), dict) else {}

    overall_rank: dict[str, int] = {}
    previous_stats: dict[str, dict[str, float]] = {}
    for idx, item in enumerate(prev_top_scored, start=1):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        if not symbol:
            continue
        overall_rank[symbol] = idx
        previous_stats[symbol] = {
            "momentum_5": float(item.get("momentum_5", 0.0) or 0.0),
            "volume_ratio": float(item.get("volume_ratio", 1.0) or 1.0),
            "rank_score": float(item.get("rank_score", 0.0) or 0.0),
        }

    lane_rank: dict[str, int] = {}
    for lane, items in prev_lane_shortlists.items():
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip()
            if not symbol:
                continue
            lane_rank[f"{str(lane).upper()}::{symbol}"] = idx
            stats = previous_stats.setdefault(symbol, {})
            stats.setdefault("momentum_5", float(item.get("momentum_5", 0.0) or 0.0))
            stats.setdefault("volume_ratio", float(item.get("volume_ratio", 1.0) or 1.0))
            stats.setdefault("rank_score", float(item.get("rank_score", 0.0) or 0.0))

    return overall_rank, lane_rank, previous_stats


def _annotate_leader_deltas(candidates: list[Candidate]) -> list[Candidate]:
    prev_overall_rank, prev_lane_rank, prev_stats = _previous_rank_maps()
    overall_sorted = sorted(candidates, key=lambda item: item.candidate_score, reverse=True)
    for idx, candidate in enumerate(overall_sorted, start=1):
        candidate.previous_rank = int(prev_overall_rank.get(candidate.pair, 999))
        candidate.rank_delta = candidate.previous_rank - idx if candidate.previous_rank < 999 else 0
        lane_key = f"{candidate.lane.upper()}::{candidate.pair}"
        candidate.previous_lane_rank = int(prev_lane_rank.get(lane_key, 999))
        lane_sorted = 0  # filled later after per-lane sort
        prev = prev_stats.get(candidate.pair, {})
        prev_momo = float(prev.get("momentum_5", 0.0) or 0.0)
        prev_volume_ratio = float(prev.get("volume_ratio", 1.0) or 1.0)
        candidate.momentum_delta = candidate.momentum_5 - prev_momo
        candidate.volume_ratio_delta = candidate.volume_ratio - prev_volume_ratio
        candidate.leader_takeover = bool(candidate.previous_rank >= 999 and candidate.candidate_score >= 55.0)
        # Hard trigger: decisive surge pattern — jump the queue unconditionally
        hard_trigger = (
            candidate.rank_delta >= 5
            and candidate.momentum_5 > 0.005
            and candidate.momentum_delta > 0.0
            and candidate.volume_ratio_delta > 0.0
            and candidate.candidate_recommendation != "AVOID"
            and candidate.spread_bps <= 25.0
        )
        if hard_trigger:
            candidate.leader_takeover = True
        candidate.leader_urgency = (
            max(candidate.rank_delta, 0) * 1.8
            + max(candidate.momentum_delta, 0.0) * 1200.0
            + max(candidate.volume_ratio_delta, 0.0) * 6.0
            + (4.0 if candidate.leader_takeover else 0.0)
            + (8.0 if hard_trigger else 0.0)
        )
        candidate.rank_score = (
            candidate.candidate_score
            + candidate.leader_urgency
            + (candidate.net_edge_pct * 4.0)
            - max(candidate.total_cost_pct - candidate.expected_move_pct, 0.0) * 2.0
            + (candidate.realized_follow_through_pct * 2.0)
        )
    return candidates


def _lane_rank_score(candidate: Candidate) -> float:
    # Confirmed alignment bonus — requires 3+ timeframes agreeing (no single-bar spikes)
    confirmed = candidate.momentum_alignment >= 3
    align_bonus = candidate.momentum_alignment * 3.5 if confirmed else 0.0
    # Acceleration bonus — short-term outpacing long-term (fresh breakout)
    accel_bonus = max(candidate.momentum_acceleration, 0.0) * 400.0 if confirmed else 0.0
    structure_bonus = (candidate.structure_quality - 50.0) * 0.18
    momentum_bonus = (candidate.momentum_quality - 50.0) * 0.22
    volume_bonus = (candidate.volume_quality - 50.0) * 0.12
    trade_bonus = (candidate.trade_quality - 50.0) * 0.16
    market_bonus = (candidate.market_support - 50.0) * 0.14
    continuation_bonus = (candidate.continuation_quality - 50.0) * 0.20
    risk_bonus = (candidate.risk_quality - 50.0) * 0.14
    edge_bonus = candidate.net_edge_pct * 8.0
    cost_penalty = candidate.total_cost_pct * 2.0
    realized_cost_penalty = candidate.realized_cost_penalty_pct * 2.0
    realized_follow_bonus = candidate.realized_follow_through_pct * 4.0
    tp_cost_penalty = 12.0 if not candidate.tp_after_cost_valid else 0.0

    if candidate.lane == "L1":
        # Trend lane: reward broad alignment across all timeframes
        return (
            candidate.candidate_score
            + structure_bonus * 1.4
            + trade_bonus
            + market_bonus
            + continuation_bonus * 1.5
            + risk_bonus
            + (candidate.momentum_14 * 450.0)
            + (candidate.momentum_30 * 250.0)
            + max(candidate.m8h, 0.0) * 300.0
            + max(candidate.m24h, 0.0) * 200.0
            + max(candidate.vwrs, 0.0) * 600.0
            + max(candidate.volume_ratio - 1.0, 0.0) * 8.0
            + max(candidate.rank_delta, 0) * 1.4
            + max(candidate.momentum_delta, 0.0) * 300.0
            + align_bonus * 1.5  # alignment matters most here
            + edge_bonus
            + realized_follow_bonus
            - candidate.spread_bps * 0.08
            - cost_penalty
            - realized_cost_penalty
            - tp_cost_penalty
        )
    if candidate.lane == "L2":
        # Mean-reversion lane: use m8h/m24h for context
        return (
            candidate.candidate_score
            + structure_bonus
            + momentum_bonus * 1.2
            + volume_bonus * 0.8
            + trade_bonus
            + market_bonus * 0.8
            + continuation_bonus
            + risk_bonus
            + abs(candidate.price_zscore) * 8.0
            + max(candidate.momentum_5, 0.0) * 500.0
            + max(candidate.m8h, 0.0) * 200.0
            + max(candidate.score, 0.0) * 60.0
            + max(candidate.rank_delta, 0) * 2.2
            + max(candidate.momentum_delta, 0.0) * 700.0
            + max(candidate.volume_ratio_delta, 0.0) * 8.0
            + align_bonus
            + edge_bonus
            + realized_follow_bonus
            - candidate.spread_bps * 0.05
            - cost_penalty
            - realized_cost_penalty
            - tp_cost_penalty
        )
    if candidate.lane == "L4":
        # Meme/breakout lane: weight short timeframes heavily, require confirmation
        sub_hour_bonus = (
            max(candidate.m5m, 0.0) * 1800.0
            + max(candidate.m30m, 0.0) * 1000.0
        ) if confirmed else 0.0
        return (
            candidate.candidate_score
            + structure_bonus * 0.8
            + momentum_bonus * 1.5
            + volume_bonus * 1.3
            + trade_bonus
            + market_bonus * 0.8
            + continuation_bonus * 1.1
            + risk_bonus * 0.8
            + max(candidate.momentum_5, 0.0) * 1100.0
            + max(candidate.m4h, 0.0) * 600.0
            + max(candidate.volume_ratio - 1.0, 0.0) * 12.0
            + max(candidate.vwrs, 0.0) * 450.0
            + max(candidate.rank_delta, 0) * 2.5
            + max(candidate.momentum_delta, 0.0) * 900.0
            + max(candidate.volume_ratio_delta, 0.0) * 10.0
            + sub_hour_bonus   # 5m and 30m confirmation
            + align_bonus * 2.0
            + accel_bonus * 1.5
            + edge_bonus
            + realized_follow_bonus
            - candidate.spread_bps * 0.03
            - cost_penalty
            - realized_cost_penalty
            - tp_cost_penalty
        )
    # L3 default: medium-term focus, penalize lone spikes without confirmation
    return (
        candidate.candidate_score
        + structure_bonus * 1.1
        + momentum_bonus
        + volume_bonus * 0.8
        + trade_bonus * 1.2
        + market_bonus
        + continuation_bonus
        + risk_bonus * 1.1
        + max(candidate.momentum_14, 0.0) * 260.0
        + max(candidate.momentum_30, 0.0) * 180.0
        + max(candidate.m8h, 0.0) * 180.0
        + max(candidate.m24h, 0.0) * 120.0
        + max(candidate.volume_ratio - 0.8, 0.0) * 4.0
        + (6.0 if 46.0 <= candidate.rsi <= 64.0 else -4.0)
        + max(candidate.rank_delta, 0) * 1.2
        + max(candidate.momentum_delta, 0.0) * 180.0
        + align_bonus
        + edge_bonus
        + realized_follow_bonus
        - max(candidate.momentum_5 - 0.006, 0.0) * 420.0
        - candidate.spread_bps * 0.06
        - cost_penalty
        - realized_cost_penalty
        - tp_cost_penalty
    )


def _lane_sort(candidates: list[Candidate]) -> list[Candidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (_lane_rank_score(item), item.rank_score, item.candidate_score, item.volume_usd),
        reverse=True,
    )
    for idx, candidate in enumerate(sorted_candidates, start=1):
        candidate.lane_rank_delta = candidate.previous_lane_rank - idx if candidate.previous_lane_rank < 999 else 0
    return sorted_candidates


def _lane_shortlist_limits(shortlist_size: int) -> dict[str, int]:
    # Default percentage allocation per lane (must sum to ~100)
    default_pct = {"L1": 25, "L2": 30, "L3": 30, "L4": 15}

    # Allow override via LANE_SHORTLIST_PCT=L1:25,L2:30,L3:30,L4:15
    raw_pct = os.getenv("LANE_SHORTLIST_PCT", "").strip()
    if raw_pct:
        parsed_pct = dict(default_pct)
        for chunk in raw_pct.split(","):
            lane, _, value = chunk.partition(":")
            lane = lane.strip().upper()
            try:
                parsed_pct[lane] = max(0, int(value.strip()))
            except Exception:
                continue
        pct = parsed_pct
    else:
        pct = default_pct

    total_pct = sum(pct.values())
    if total_pct <= 0:
        pct = default_pct
        total_pct = sum(pct.values())

    # Derive per-lane slot counts from percentages
    limits = {
        lane: max(1 if v > 0 else 0, round(v / total_pct * shortlist_size))
        for lane, v in pct.items()
    }

    # Allow fixed-count override via LANE_SHORTLIST_LIMITS=L1:3,L2:4,L3:3,L4:2 (takes precedence)
    raw_fixed = os.getenv("LANE_SHORTLIST_LIMITS", "").strip()
    if raw_fixed:
        for chunk in raw_fixed.split(","):
            lane, _, value = chunk.partition(":")
            lane = lane.strip().upper()
            try:
                limits[lane] = max(0, int(value.strip()))
            except Exception:
                continue

    return limits


def scan_l1_candidates(candidates: list[Candidate]) -> list[Candidate]:
    scoped = [c for c in candidates if c.lane == "L1" and c.candidate_recommendation != "AVOID"]
    return _lane_sort(scoped)


def scan_l2_candidates(candidates: list[Candidate]) -> list[Candidate]:
    # L2 rotation names + near-L2 L3 names rated BUY/STRONG_BUY (borderline classification)
    scoped = [
        c for c in candidates
        if c.candidate_recommendation != "AVOID"
        and (
            c.lane == "L2"
            or (c.lane == "L3" and c.candidate_recommendation in {"BUY", "STRONG_BUY"})
        )
    ]
    return _lane_sort(scoped)


def scan_l3_candidates(candidates: list[Candidate]) -> list[Candidate]:
    scoped = [
        c for c in candidates
        if (
            c.lane == "L3"
            and _scan_quality_ok(c)
            and c.volume_ratio >= 0.75
            and c.spread_bps <= 20.0
            and c.trade_quality >= 55.0
            and c.risk_quality >= 52.0
            and c.momentum_5 <= 0.010  # exclude hot movers — they belong in L2 or L4
        )
    ]
    return _lane_sort(scoped)


def scan_l4_candidates(candidates: list[Candidate]) -> list[Candidate]:
    scoped = [c for c in candidates if c.lane == "L4" and c.candidate_recommendation != "AVOID"]
    return _lane_sort(scoped)


def build_lane_shortlists(
    candidates: list[Candidate],
    *,
    shortlist_size: int,
) -> tuple[dict[str, list[Candidate]], list[Candidate], dict[str, list[str]]]:
    limits = _lane_shortlist_limits(shortlist_size)
    lane_shortlists = {
        "L1": scan_l1_candidates(candidates)[: limits.get("L1", 0)],
        "L2": scan_l2_candidates(candidates)[: limits.get("L2", 0)],
        "L3": scan_l3_candidates(candidates)[: limits.get("L3", 0)],
        "L4": scan_l4_candidates(candidates)[: limits.get("L4", 0)],
    }

    merged: list[Candidate] = []
    seen: set[str] = set()
    lane_order = ["L1", "L2", "L3", "L4"]
    max_depth = max((len(items) for items in lane_shortlists.values()), default=0)
    for idx in range(max_depth):
        for lane in lane_order:
            items = lane_shortlists.get(lane, [])
            if idx >= len(items):
                continue
            candidate = items[idx]
            if candidate.pair in seen:
                continue
            merged.append(candidate)
            seen.add(candidate.pair)
            if len(merged) >= shortlist_size:
                break
        if len(merged) >= shortlist_size:
            break

    if len(merged) < shortlist_size:
        fallback = _lane_sort(candidates)
        for candidate in fallback:
            if candidate.pair in seen:
                continue
            merged.append(candidate)
            seen.add(candidate.pair)
            if len(merged) >= shortlist_size:
                break

    # Hard trigger injection: decisive surge candidates bypass scan filters and jump to front
    force_candidates = sorted(
        [c for c in candidates if c.leader_takeover and c.leader_urgency >= 8.0 and c.pair not in seen],
        key=lambda c: c.leader_urgency,
        reverse=True,
    )
    for candidate in force_candidates:
        if len(merged) >= shortlist_size:
            merged.pop()  # displace the weakest tail entry to make room
        merged.insert(0, candidate)
        seen.add(candidate.pair)
        lane_shortlists.setdefault(candidate.lane, []).insert(0, candidate)

    lane_meta = {
        lane: [candidate.pair for candidate in items]
        for lane, items in lane_shortlists.items()
    }
    lane_meta["merged"] = [candidate.pair for candidate in merged]
    return lane_shortlists, merged, lane_meta


def _simple_momentum(closes: list[float], lookback: int) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    prev = float(closes[-(lookback + 1)])
    last = float(closes[-1])
    if prev <= 0.0:
        return 0.0
    return (last / prev) - 1.0


def _simple_rsi(closes: list[float], lookback: int = 14) -> float:
    if len(closes) < lookback + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    window = closes[-(lookback + 1) :]
    for idx in range(1, len(window)):
        delta = float(window[idx]) - float(window[idx - 1])
        if delta > 0:
            gains += delta
        elif delta < 0:
            losses += abs(delta)
    if losses == 0.0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _price_zscore(closes: list[float], lookback: int = 20) -> float:
    if len(closes) < lookback:
        return 0.0
    window = closes[-lookback:]
    mean = sum(window) / len(window)
    variance = sum((value - mean) ** 2 for value in window) / len(window)
    if variance <= 0.0:
        return 0.0
    std = variance ** 0.5
    if std <= 0.0:
        return 0.0
    return (closes[-1] - mean) / std


def _volume_ratio(rows: list[list[str | float]], lookback: int = 20) -> float:
    if len(rows) < lookback + 1:
        return 1.0
    volumes = [float(r[6]) for r in rows[-(lookback + 1) :]]
    current = volumes[-1]
    baseline = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
    if baseline <= 0.0:
        return 1.0
    return current / baseline


def load_pair_pool() -> list[str]:
    pool_file = os.getenv("PAIR_POOL_FILE", "pair_pool_usd.txt")
    path = Path(pool_file)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def get_policy() -> UniversePolicy:
    return UniversePolicy(
        active_min=int(os.getenv("ACTIVE_MIN", "10")),
        active_max=int(os.getenv("ACTIVE_MAX", "15")),
        max_adds=int(os.getenv("MAX_ADDS_PER_REBALANCE", "5")),
        max_removes=int(os.getenv("MAX_REMOVES_PER_REBALANCE", "5")),
        cooldown_minutes=int(os.getenv("PAIR_COOLDOWN_MINUTES", "180")),
        min_volume_usd=float(os.getenv("MIN_QUOTE_VOLUME_USD", "200000")),
        max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", "12")),
        min_price=float(os.getenv("MIN_PRICE", "0.05")),
        churn_threshold=float(os.getenv("CHURN_THRESHOLD", "1.20")),
    )


def normalize_pair(pair: str) -> str:
    return normalize_symbol(pair)


def _env_symbol_set(name: str, default: str = "") -> set[str]:
    raw = str(os.getenv(name, default) or "").strip()
    if not raw:
        return set()
    return {normalize_pair(chunk) for chunk in raw.split(",") if normalize_pair(chunk)}


def _symbol_base(symbol: str) -> str:
    normalized = normalize_pair(symbol)
    base, _, _quote = normalized.partition("/")
    return base.strip().upper()


def _meme_universe_enabled() -> bool:
    return bool(get_runtime_setting("MEME_UNIVERSE_ENABLED"))


def _meme_universe_cap() -> int:
    default_cap = int(get_runtime_setting("MEME_MAX_OPEN_POSITIONS"))
    configured_cap = int(get_runtime_setting("MEME_ACTIVE_UNIVERSE_MAX", default_cap))
    return max(0, configured_cap)


def _retention_buffer_mult() -> float:
    return max(1.0, float(os.getenv("UNIVERSE_RETAIN_BUFFER_MULT", "1.5")))


def _refresh_interval_sec() -> int:
    configured_min = max(1, int(os.getenv("UNIVERSE_REFRESH_INTERVAL_MIN", "5")))
    floor_min = max(1, int(os.getenv("UNIVERSE_MIN_REFRESH_INTERVAL_MIN", "5")))
    return max(configured_min, floor_min) * 60


def _excluded_bases() -> set[str]:
    default = "USDT,USDC,DAI,PYUSD,TUSD,FDUSD,USDE,USDS,USDP"
    raw = str(os.getenv("UNIVERSE_EXCLUDED_BASES", default) or "").strip()
    return {chunk.strip().upper() for chunk in raw.split(",") if chunk.strip()}


def _excluded_symbols() -> set[str]:
    return _env_symbol_set("UNIVERSE_EXCLUDED_SYMBOLS")


def _runtime_symbol_set(name: str) -> set[str]:
    raw = str(get_runtime_setting(name) or "").strip()
    if not raw:
        return set()
    return {normalize_pair(chunk) for chunk in raw.split(",") if normalize_pair(chunk)}


def _core_active_universe() -> set[str]:
    return _runtime_symbol_set("CORE_ACTIVE_UNIVERSE")


def _conditional_universe() -> set[str]:
    if not bool(get_runtime_setting("ENABLE_CONDITIONAL_UNIVERSE")):
        return set()
    return _runtime_symbol_set("CONDITIONAL_UNIVERSE")


def _strict_seed_only_enabled() -> bool:
    return str(os.getenv("UNIVERSE_STRICT_SEED_ONLY", "false") or "").strip().lower() == "true"


def _stabilization_universe() -> set[str]:
    return _core_active_universe() | _conditional_universe()


def _xrp_min_structure_quality() -> float:
    raw = float(get_runtime_setting("XRP_MIN_STRUCTURE_QUALITY"))
    return raw * 100.0 if raw <= 1.0 else raw


def _xrp_cleanliness_score(candidate: Candidate) -> tuple[int, float, float]:
    return (
        int(round(float(candidate.structure_quality))),
        float(candidate.net_edge_pct),
        -float(candidate.spread_bps / 100.0),
    )


def _duplicate_exposure_proxy(candidate: Candidate, other: Candidate) -> float:
    score = 0.0
    if str(candidate.lane).upper() == str(other.lane).upper():
        score += 0.30
    if int(candidate.trend_1h) != 0 and int(candidate.trend_1h) == int(other.trend_1h):
        score += 0.20
    if (candidate.momentum_14 > 0.0) == (other.momentum_14 > 0.0):
        score += 0.20
    if abs(float(candidate.price_zscore) - float(other.price_zscore)) <= 0.75:
        score += 0.15
    if abs(float(candidate.rsi) - float(other.rsi)) <= 8.0:
        score += 0.15
    return min(score, 1.0)


def _cleaner_than_core_candidate(candidate: Candidate, core_candidate: Candidate) -> bool:
    wins = 0
    if float(candidate.spread_bps / 100.0) <= float(core_candidate.spread_bps / 100.0):
        wins += 1
    if float(candidate.net_edge_pct) >= float(core_candidate.net_edge_pct):
        wins += 1
    if float(candidate.structure_quality) >= float(core_candidate.structure_quality):
        wins += 1
    return wins >= 2


def _xrp_candidate_allowed(candidate: Candidate, core_candidates: list[Candidate]) -> bool:
    if float(candidate.spread_bps / 100.0) > float(get_runtime_setting("XRP_MAX_SPREAD_PCT")):
        return False
    if float(candidate.net_edge_pct) < float(get_runtime_setting("XRP_MIN_NET_EDGE_PCT")):
        return False
    if float(candidate.structure_quality) < _xrp_min_structure_quality():
        return False
    if not bool(candidate.tp_after_cost_valid):
        return False

    duplicate_limit = float(get_runtime_setting("XRP_MAX_DUPLICATE_CORR"))
    for core_candidate in core_candidates:
        if _duplicate_exposure_proxy(candidate, core_candidate) >= duplicate_limit and not _cleaner_than_core_candidate(candidate, core_candidate):
            return False
    return True


def _conditional_candidate_allowed(candidate: Candidate, core_candidates: list[Candidate]) -> bool:
    normalized = normalize_pair(candidate.pair)
    if normalized in _core_active_universe():
        return True
    if normalized not in _conditional_universe():
        return False
    if normalized == "XRP/USD":
        return _xrp_candidate_allowed(candidate, core_candidates)
    return False


def _broad_candidate_allowed(candidate: Candidate) -> bool:
    score_floor = float(os.getenv("UNIVERSE_EXPANSION_MIN_ENTRY_SCORE", "50"))
    rank_floor = float(os.getenv("UNIVERSE_EXPANSION_MIN_RANK_SCORE", "55"))
    volume_ratio_floor = float(os.getenv("UNIVERSE_EXPANSION_MIN_VOLUME_RATIO", "0.85"))
    trade_quality_floor = float(os.getenv("UNIVERSE_EXPANSION_MIN_TRADE_QUALITY", "50"))
    recommendation = str(candidate.candidate_recommendation or "WATCH").upper()
    risk = str(candidate.candidate_risk or "MEDIUM").upper()
    if recommendation == "AVOID" or risk == "HIGH":
        return False
    if candidate.volume_ratio < volume_ratio_floor:
        return False
    if candidate.trade_quality < trade_quality_floor:
        return False
    if candidate.candidate_score < score_floor and candidate.rank_score < rank_floor:
        return False
    return True


def _major_bases() -> set[str]:
    default = "BTC,XBT,ETH,SOL,XRP,ADA,DOGE,TRX,AVAX,LINK,DOT,LTC"
    raw = str(os.getenv("UNIVERSE_MAJOR_BASES", default) or "").strip()
    return {chunk.strip().upper() for chunk in raw.split(",") if chunk.strip()}


def _metal_bases() -> set[str]:
    raw = str(os.getenv("UNIVERSE_METAL_BASES", "PAXG,XAUT") or "").strip()
    return {chunk.strip().upper() for chunk in raw.split(",") if chunk.strip()}


def _segment_for_symbol(symbol: str, lane: str | None = None) -> str:
    base = _symbol_base(symbol)
    lane_upper = str(lane or "").strip().upper()
    if is_meme_symbol(symbol) or lane_upper == "L4":
        return "meme"
    if base in _metal_bases():
        return "store_of_value"
    if base in _major_bases():
        return "major"
    return "core_alt"


def _layer_for_candidate(candidate: Candidate) -> str:
    base_segment = _segment_for_symbol(candidate.pair, candidate.lane)
    normalized = normalize_pair(candidate.pair)
    if base_segment == "meme":
        return "meme"
    if normalized in _core_active_universe() or base_segment in {"major", "store_of_value"}:
        return "core"

    recommendation = str(candidate.candidate_recommendation or "WATCH").upper()
    if (
        recommendation in {"BUY", "STRONG_BUY"}
        and candidate.momentum_5 > 0.0
        and candidate.volume_ratio >= 0.95
        and candidate.trade_quality >= 55.0
    ):
        return "momentum"

    if (
        recommendation in {"WATCH", "BUY", "STRONG_BUY"}
        and candidate.momentum_14 >= 0.0
        and candidate.structure_quality >= 55.0
        and candidate.continuation_quality >= 55.0
    ):
        return "recovery"

    return "core"


def _universe_symbol_allowed(symbol: str) -> bool:
    normalized = normalize_pair(symbol)
    if not normalized:
        return False
    if normalized in _excluded_symbols():
        return False
    if _symbol_base(normalized) in _excluded_bases():
        return False
    if is_meme_symbol(normalized) and not _meme_universe_enabled():
        return False
    stabilization_universe = _stabilization_universe()
    if _strict_seed_only_enabled() and stabilization_universe and normalized not in stabilization_universe:
        return False
    return True


def _ranked_symbol_rows(candidates: list[Candidate]) -> list[tuple[int, Candidate]]:
    return list(enumerate(candidates, start=1))


def _select_segment_active(
    *,
    segment_candidates: list[Candidate],
    current_active: list[str],
    held_symbols: set[str],
    target_size: int,
    policy: UniversePolicy,
) -> tuple[list[str], list[str], list[str]]:
    if target_size <= 0 or not segment_candidates:
        return [], [], [pair for pair in current_active if pair not in held_symbols]

    ranked = _ranked_symbol_rows(segment_candidates)
    score_map = {candidate.pair: max(candidate.rank_score, candidate.candidate_score) for _, candidate in ranked}
    rank_map = {candidate.pair: idx for idx, candidate in ranked}
    retain_rank_limit = max(target_size, int(round(target_size * _retention_buffer_mult())))

    retained: list[str] = []
    for pair in current_active:
        if pair not in score_map:
            continue
        if pair in held_symbols or rank_map.get(pair, 10_000) <= retain_rank_limit:
            retained.append(pair)

    retained = sorted(
        dict.fromkeys(retained),
        key=lambda pair: (0 if pair in held_symbols else 1, -score_map.get(pair, float("-inf"))),
    )[:target_size]

    additions: list[str] = []
    current_scores = [score_map.get(pair, float("-inf")) for pair in retained]
    worst_kept = min(current_scores) if current_scores else float("-inf")

    for idx, candidate in ranked:
        if len(retained) + len(additions) >= target_size or len(additions) >= policy.max_adds:
            break
        if candidate.pair in retained or candidate.pair in additions:
            continue
        if worst_kept != float("-inf") and idx > retain_rank_limit:
            if not apply_churn_threshold(
                score_map.get(candidate.pair, float("-inf")),
                worst_kept,
                policy.churn_threshold,
            ):
                continue
        additions.append(candidate.pair)
        worst_kept = min(worst_kept, score_map.get(candidate.pair, float("-inf"))) if worst_kept != float("-inf") else score_map.get(candidate.pair, float("-inf"))

    next_active = retained + additions
    if len(next_active) < target_size:
        for _idx, candidate in ranked:
            if candidate.pair in next_active:
                continue
            next_active.append(candidate.pair)
            if len(next_active) >= target_size:
                break

    next_active = next_active[:target_size]
    removed = [pair for pair in current_active if pair not in next_active and pair not in held_symbols]
    return next_active, additions, removed


def load_cached_asset_pairs(cache_path: Path, ttl_min: int) -> dict | None:
    if not cache_path.exists():
        return None
    age_s = time.time() - cache_path.stat().st_mtime
    if age_s > ttl_min * 60:
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def save_cached_asset_pairs(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))


def filter_pairs_with_kraken(client: KrakenRestClient, pairs: list[str]) -> tuple[list[str], list[str], dict]:
    # Use Kraken AssetPairs to validate available USD pairs
    cache_file = Path(os.getenv("ASSET_PAIRS_CACHE_FILE", "configs/kraken_asset_pairs.cache.json"))
    ttl_min = int(os.getenv("ASSET_PAIRS_CACHE_TTL_MIN", "30"))
    resp = load_cached_asset_pairs(cache_file, ttl_min)
    if resp is None:
        try:
            resp = client.get_asset_pairs()
            save_cached_asset_pairs(cache_file, resp)
        except Exception:
            # If REST fails and cache exists, try using stale cache
            if cache_file.exists():
                try:
                    resp = json.loads(cache_file.read_text())
                except Exception:
                    raise
            else:
                raise

    result = resp.get("result", {})
    meta_map: dict[str, dict] = {}
    for k, v in result.items():
        wsname = v.get("wsname")
        altname = v.get("altname")
        status = (v.get("status") or "online").lower()
        if status != "online":
            continue
        if wsname:
            meta_map[wsname.upper()] = {"key": k, **v}
        if altname:
            meta_map[altname.upper()] = {"key": k, **v}

    normalized = [normalize_pair(p) for p in pairs]
    valid = [p for p in normalized if to_kraken_symbol(p) in meta_map]
    invalid = [p for p in normalized if to_kraken_symbol(p) not in meta_map]
    return valid, invalid, meta_map


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def calc_spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 1e9
    return ((ask - bid) / mid) * 10_000.0


def first_ohlc_timestamp(client: KrakenRestClient, pair: str, interval_min: int) -> int | None:
    try:
        res = client.get_ohlc(pair, interval_min=interval_min, since=0)
        result = res.get("result", {})
        for k, v in result.items():
            if k == "last":
                continue
            if isinstance(v, list) and v:
                return int(v[0][0])
    except Exception:
        return None
    return None


def validate_and_filter_bench(
    client: KrakenRestClient, bench_pairs: list[str]
) -> tuple[list[str], list[str], dict]:
    require_marginable = os.getenv("REQUIRE_MARGINABLE", "false").lower() == "true"
    min_notional = float(os.getenv("MIN_NOTIONAL_USD", "25"))
    max_ordermin_notional = float(os.getenv("MAX_ORDERMIN_NOTIONAL_USD", "25"))
    min_vol_usd = float(os.getenv("MIN_QUOTE_VOLUME_USD", "200000"))
    max_spread_bps = float(os.getenv("MAX_SPREAD_BPS", "12"))
    min_price = float(os.getenv("MIN_PRICE", "0.05"))

    min_age_days = int(os.getenv("MIN_AGE_DAYS", "30"))
    age_interval = int(os.getenv("AGE_CHECK_INTERVAL_MINUTES", "60"))
    age_cache_file = os.getenv("AGE_CACHE_FILE", "configs/pair_age_cache.json")
    age_cache_ttl = int(os.getenv("AGE_CACHE_TTL_HOURS", "24"))

    scoped_pairs = [pair for pair in bench_pairs if _universe_symbol_allowed(pair)]
    valid_pool, invalid_pool, meta_map = filter_pairs_with_kraken(client, scoped_pairs)

    # Marginable filter (optional)
    if require_marginable:
        valid_pool = [
            p for p in valid_pool if bool(meta_map[p].get("marginable") or meta_map[p].get("margin"))
        ]

    # Batch ticker lookup
    eligible: list[str] = []
    for batch in chunked(valid_pool, 20):
        ticker_pairs = [
            str(meta_map[to_kraken_symbol(p)].get("key") or to_kraken_symbol(p))
            for p in batch
            if to_kraken_symbol(p) in meta_map
        ]
        tick = client.get_ticker(ticker_pairs).get("result", {}) if ticker_pairs else {}
        for p in batch:
            meta = meta_map.get(to_kraken_symbol(p), {})
            ticker_key = (meta.get("altname") or meta.get("wsname") or meta.get("key") or p).upper()
            t = tick.get(ticker_key) or tick.get(p) or tick.get(meta.get("key"))
            if not t:
                continue

            try:
                ask = float(t["a"][0])
                bid = float(t["b"][0])
                last = float(t["c"][0])
                vol_base = float(t["v"][1])
            except Exception:
                continue

            if last < min_price:
                continue
            quote_vol = vol_base * last
            if quote_vol < min_vol_usd:
                continue
            if calc_spread_bps(bid, ask) > max_spread_bps:
                continue

            ordermin = float(meta.get("ordermin") or 0.0)
            ordermin_notional = ordermin * last if ordermin > 0 else 0.0
            if ordermin_notional > max_ordermin_notional:
                continue
            if min_notional < ordermin_notional:
                # This pair is still ok; execution should clamp upward.
                pass

            if not _universe_symbol_allowed(p):
                continue
            eligible.append(p)

    # Age filter with caching
    age_cache = load_age_cache(age_cache_file, ttl_hours=age_cache_ttl)
    aged: list[str] = []
    now = int(time.time())
    for p in eligible:
        rec = age_cache.get(p)
        first_ts = int(rec["first_ts"]) if rec and "first_ts" in rec else None
        if not first_ts:
            first_ts = first_ohlc_timestamp(client, p, interval_min=age_interval)
            if first_ts:
                age_cache[p] = {"first_ts": first_ts, "checked_at": now}
        if not first_ts:
            aged.append(p)
            continue
        age_days = (now - first_ts) / 86400.0
        if age_days < min_age_days:
            continue
        aged.append(p)

    save_age_cache(age_cache_file, age_cache)
    return aged, invalid_pool, meta_map


def rank_pairs(client: KrakenRestClient, pairs: list[str], meta_map: dict[str, dict]) -> list[Candidate]:
    # Basic ranking using OHLC 1h candles over 24h
    candidates: list[Candidate] = []
    ticker_pairs = [
        str(meta_map[to_kraken_symbol(pair)].get("key") or to_kraken_symbol(pair))
        for pair in pairs
        if to_kraken_symbol(pair) in meta_map
    ]
    ticker_result = client.get_ticker(ticker_pairs).get("result", {}) if ticker_pairs else {}
    for pair in pairs:
        try:
            meta = meta_map.get(to_kraken_symbol(pair), {})
            pair_key = str(meta.get("key") or pair)
            ohlc = client.get_ohlc(pair_key, interval_min=60)
            result = ohlc.get("result", {})
            rows = None
            for key, value in result.items():
                if key == "last":
                    continue
                if key in {pair, pair_key, meta.get("key"), meta.get("altname"), meta.get("wsname"), to_kraken_symbol(pair)}:
                    rows = value
                    break
            if rows is None:
                continue
            # Use all available hourly bars (up to 720 = 30 days) for full momentum profile
            all_closes = [float(r[4]) for r in rows]
            closes = all_closes[-25:]  # keep 25 for rsi/zscore/volume calcs
            score = risk_adjusted_momentum(closes)
            momentum_5 = _simple_momentum(all_closes, 5)
            momentum_14 = _simple_momentum(all_closes, 14)
            momentum_30 = _simple_momentum(all_closes, min(30, len(all_closes) - 1))
            # Extended hourly momentum (no extra API calls — use bars already fetched)
            m4h = _simple_momentum(all_closes, 4)
            m8h = _simple_momentum(all_closes, 8)
            m24h = _simple_momentum(all_closes, 24)
            m7d = _simple_momentum(all_closes, min(168, len(all_closes) - 1))
            m30d = _simple_momentum(all_closes, min(720, len(all_closes) - 1))
            # Breakout bonus: fast movers get penalized by Sharpe-like scoring.
            # Require at least 3 hourly timeframes confirming (alignment) before boosting.
            hourly_alignment = sum([m4h > 0, m8h > 0, m24h > 0, m7d > 0, momentum_5 > 0])
            if momentum_5 > 0.02 and hourly_alignment >= 3:
                score = max(score if score > float("-inf") else 0.0, 0.0) + min(momentum_5 * 40.0, 25.0)
            last = closes[-1]
            volume_ratio = _volume_ratio(rows)
            price_zscore = _price_zscore(closes)
            rsi = _simple_rsi(closes)
            atr_pct = _simple_atr_pct(rows)
            # Volume is base volume; convert to quote using last
            vol_base = sum(float(r[6]) for r in rows[-24:])
            vol_quote = vol_base * last
            trend_1h = 1 if momentum_14 > 0 else (-1 if momentum_14 < 0 else 0)
            ticker_key = (meta.get("altname") or meta.get("wsname") or meta.get("key") or pair).upper()
            ticker = (
                ticker_result.get(ticker_key)
                or ticker_result.get(pair_key)
                or ticker_result.get(meta.get("key"))
                or {}
            )
            try:
                ask = float(ticker["a"][0])
                bid = float(ticker["b"][0])
                spread_bps = calc_spread_bps(bid, ask)
            except Exception:
                spread_bps = 0.0
            policy = apply_policy_pipeline(
                pair,
                {
                    "symbol": pair,
                    "momentum": momentum_14,
                    "momentum_5": momentum_5,
                    "momentum_14": momentum_14,
                    "momentum_30": momentum_30,
                    "trend_1h": trend_1h,
                    "volume_ratio": volume_ratio,
                    "price_zscore": price_zscore,
                    "rsi": rsi,
                    "regime_7d": "unknown",
                    "macro_30d": "sideways",
                    "price": last,
                    "atr": last * (atr_pct / 100.0),
                    "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
                    "bb_bandwidth": 0.02,
                    "spread_pct": spread_bps / 100.0,
                    "rotation_score": score,
                    "correlation_row": [],
                    "hurst": 0.5,
                    "autocorr": 0.0,
                    "entropy": 0.5,
                },
            )
            lane = str(policy.get("lane", "L3"))
            candidate_reasons = list(policy.get("entry_reasons", []))
            lane_filter_reason = str(policy.get("lane_filter_reason", "") or "")
            if lane_filter_reason and lane_filter_reason not in candidate_reasons:
                candidate_reasons.append(lane_filter_reason)
            regime_state_value = str(policy.get("regime_state", "") or "")
            if regime_state_value:
                candidate_reasons.append(regime_state_value)
            candidates.append(
                Candidate(
                    pair=pair,
                    score=score,
                    volume_usd=vol_quote,
                    last=last,
                    momentum_5=momentum_5,
                    momentum_14=momentum_14,
                    momentum_30=momentum_30,
                    m4h=m4h,
                    m8h=m8h,
                    m24h=m24h,
                    m7d=m7d,
                    m30d=m30d,
                    trend_1h=trend_1h,
                    volume_ratio=volume_ratio,
                    price_zscore=price_zscore,
                    rsi=rsi,
                    spread_bps=spread_bps,
                    structure_quality=float(policy.get("structure_quality", 50.0) or 50.0),
                    momentum_quality=float(policy.get("momentum_quality", 50.0) or 50.0),
                    volume_quality=float(policy.get("volume_quality", 50.0) or 50.0),
                    trade_quality=float(policy.get("trade_quality", 50.0) or 50.0),
                    market_support=float(policy.get("market_support", 50.0) or 50.0),
                    continuation_quality=float(policy.get("continuation_quality", 50.0) or 50.0),
                    risk_quality=float(policy.get("risk_quality", 50.0) or 50.0),
                    lane=lane,
                    candidate_score=float(policy.get("entry_score", 0.0) or 0.0),
                    candidate_recommendation=str(policy.get("entry_recommendation", "WATCH")),
                    candidate_risk=str(policy.get("reversal_risk", "MEDIUM")),
                    candidate_reasons=candidate_reasons,
                    atr_pct=atr_pct,
                    bullish_divergence=bool(policy.get("bullish_divergence", False)),
                    bearish_divergence=bool(policy.get("bearish_divergence", False)),
                    divergence_strength=float(policy.get("divergence_strength", 0.0) or 0.0),
                    divergence_age_bars=int(policy.get("divergence_age_bars", 99) or 99),
                )
            )
        except Exception:
            continue
    # Sub-hourly enrichment: fetch 5m OHLC for top 60 candidates to get m5m, m30m,
    # and volume-weighted 15m returns for true VWRS computation.
    # Limited to top 60 to avoid rate-limit blowup on 200+ coin pools.
    candidates.sort(key=lambda c: c.candidate_score, reverse=True)
    _vwrs_15m_map: dict[str, float] = {}   # pair → volume-weighted 15m raw return
    for c in candidates[:60]:
        try:
            meta = meta_map.get(to_kraken_symbol(c.pair), {})
            pair_key = str(meta.get("key") or c.pair)
            ohlc_5m = client.get_ohlc(pair_key, interval_min=5)
            result_5m = ohlc_5m.get("result", {})
            rows_5m = None
            for key, value in result_5m.items():
                if key == "last":
                    continue
                rows_5m = value
                break
            if rows_5m and len(rows_5m) >= 7:
                closes_5m = [float(r[4]) for r in rows_5m]
                c.m5m = _simple_momentum(closes_5m, 1)   # last 5-min bar change
                c.m30m = _simple_momentum(closes_5m, 6)  # 6 x 5min = 30min
                # Volume-weighted 15m return: 3 completed 5m bars
                # Kraken OHLC row: [time, open, high, low, close, vwap, volume, count]
                recent = rows_5m[-4:]   # 4 rows → 3 bar-to-bar intervals
                _vw_num, _vw_den = 0.0, 0.0
                for _i in range(1, len(recent)):
                    _prev_close = float(recent[_i - 1][4])
                    _curr_close = float(recent[_i][4])
                    _vol        = float(recent[_i][6])
                    if _prev_close > 0.0 and _vol > 0.0:
                        _vw_num += ((_curr_close / _prev_close) - 1.0) * _vol
                        _vw_den += _vol
                if _vw_den > 0.0:
                    _vwrs_15m_map[c.pair] = _vw_num / _vw_den
        except Exception:
            pass

    # Compute alignment and acceleration for all candidates
    for c in candidates:
        # Alignment: how many timeframes are pointing up (confirmation count, 0-7)
        c.momentum_alignment = sum([
            c.m5m > 0,
            c.m30m > 0,
            c.momentum_5 > 0,   # 5h
            c.m8h > 0,
            c.m24h > 0,
            c.m7d > 0,
            c.m30d > 0,
        ])
        # Acceleration: short-term outpacing long-term (breakout signal)
        # Positive = short > long = fresh breakout, Negative = fading
        c.momentum_acceleration = c.momentum_5 - c.m24h

    # VWRS: volume-weighted relative strength vs BTC on a 15m window.
    # For candidates in the top-60 enrichment set, uses 3 × 5m bars weighted by volume
    # — higher-volume bars count more, giving a truer picture of momentum quality.
    # Candidates outside the top 60 fall back to the 5h hourly differential.
    btc_candidate = next((c for c in candidates if c.pair in {"BTC/USD", "XBTUSD", "BTC/USDT"}), None)
    btc_mom5 = btc_candidate.momentum_5 if btc_candidate is not None else 0.0
    btc_vwrs_15m = (
        _vwrs_15m_map.get("BTC/USD")
        or _vwrs_15m_map.get("XBTUSD")
        or _vwrs_15m_map.get("BTC/USDT")
        or btc_mom5
    )
    for c in candidates:
        raw_15m = _vwrs_15m_map.get(c.pair)
        if raw_15m is not None:
            c.vwrs = raw_15m - btc_vwrs_15m    # volume-weighted 15m relative strength
        else:
            c.vwrs = c.momentum_5 - btc_mom5   # fallback: 5h hourly relative momentum

    feedback_map = _load_recent_execution_feedback([candidate.pair for candidate in candidates])
    for candidate in candidates:
        cost_snapshot = evaluate_trade_cost(
            {
                "lane": candidate.lane,
                "price": candidate.last,
                "atr": candidate.last * (candidate.atr_pct / 100.0),
                "spread_pct": candidate.spread_bps / 100.0,
                "structure_quality": candidate.structure_quality,
                "continuation_quality": candidate.continuation_quality,
                "momentum_quality": candidate.momentum_quality,
                "trade_quality": candidate.trade_quality,
                "entry_recommendation": candidate.candidate_recommendation,
                "promotion_tier": "promote" if candidate.candidate_recommendation in {"BUY", "STRONG_BUY"} else "skip",
            },
            "LONG",
        )
        realized = feedback_map.get(candidate.pair, {})
        candidate.expected_move_pct = float(cost_snapshot.expected_move_pct)
        candidate.total_cost_pct = float(cost_snapshot.total_cost_pct)
        candidate.required_edge_pct = float(cost_snapshot.required_edge_pct)
        candidate.net_edge_pct = float(cost_snapshot.net_edge_pct)
        candidate.tp_after_cost_valid = bool(
            cost_snapshot.expected_move_pct > cost_snapshot.total_cost_pct
            and cost_snapshot.expected_edge_pct > cost_snapshot.required_edge_pct
        )
        candidate.realized_cost_penalty_pct = float(realized.get("avg_total_cost_pct", 0.0) or 0.0)
        candidate.realized_slippage_pct = float(realized.get("avg_slippage_pct", 0.0) or 0.0)
        candidate.realized_follow_through_pct = float(realized.get("avg_follow_through_pct", 0.0) or 0.0)
        candidate.golden_profile_score = _golden_profile_score(candidate)
        candidate.golden_profile_tag = _golden_profile_tag(candidate)

    from core.memory.symbol_reliability import load_reliability_map, reliability_map_as_dict

    try:
        reliability_map = reliability_map_as_dict(load_reliability_map())
    except Exception:
        reliability_map = {}
    for candidate in candidates:
        try:
            _apply_candidate_final_score(candidate, reliability_map=reliability_map)
        except Exception:
            candidate.final_score = float(candidate.candidate_score)
            candidate.reliability_bonus = 0.0
            candidate.basket_fit_bonus = 0.0
            candidate.score_breakdown = None

    candidates = _annotate_leader_deltas(candidates)
    candidates.sort(key=lambda c: (c.rank_score, c.net_edge_pct, c.candidate_score, c.volume_usd), reverse=True)
    return candidates


async def build_scan_meta(candidates: list[Candidate], lane_meta: dict[str, list[str]]) -> dict:
    news = NewsSentimentFeed(fetch_interval_sec=0.0)
    dex = DexScreenerFeed()
    try:
        await news.maybe_update()
        news_context = news.snapshot.to_dict()
    except Exception:
        news_context = {}
    try:
        dex_context = await dex.fetch_market_summary([candidate.pair for candidate in candidates[:15]])
    except Exception:
        dex_context = {}
    # Load trade memory for synced packet context
    from core.policy.candidate_packet import build_candidate_packet
    from core.memory.trade_memory import TradeMemoryStore
    try:
        _memory_store = TradeMemoryStore()
        _behavior_score = _memory_store.build_behavior_score_block(lookback=50) or None
        _raw_lessons = _memory_store.build_lessons_block(max_lessons=6) or None
        # Re-label for Phi-3 context (build_lessons_block uses Nemo header)
        _learned_lessons = (
            _raw_lessons.replace("=== NEMO LEARNED LESSONS ===", "=== RECENT ACCOUNT LESSONS ===")
            if _raw_lessons else None
        )
    except Exception:
        _behavior_score = None
        _learned_lessons = None

    scan_candidates = []
    for item in candidates[:25]:
        candidate_packet = build_candidate_packet(
            features={
                **_candidate_final_score_features(item),
                "price": item.last,
                "entry_recommendation": str(item.candidate_recommendation),
                "reversal_risk": str(item.candidate_risk),
                "promotion_tier": "promote" if item.candidate_recommendation in {"BUY", "STRONG_BUY"} else "skip",
                "promotion_reason": (item.candidate_reasons or [""])[0] if item.candidate_reasons else "",
                "momentum_5": float(item.momentum_5),
                "momentum_14": float(item.momentum_14),
                "rotation_score": float(item.score),
                "volume_ratio": float(item.volume_ratio),
                "volume_surge": 0.0,
                "rsi": float(item.rsi),
                "price_zscore": float(item.price_zscore),
                "trend_1h": int(item.trend_1h),
                "regime_7d": "unknown",
                "macro_30d": "sideways",
                "trend_confirmed": bool(item.trend_1h > 0),
                "ranging_market": False,
                "short_tf_ready_5m": False,
                "short_tf_ready_15m": False,
                "ema9_above_ema20": False,
                "range_breakout_1h": False,
                "pullback_hold": False,
                "higher_low_count": 0,
                "overextended": False,
                "range_pos_1h": 0.0,
                "book_imbalance": 0.0,
                "book_wall_pressure": 0.0,
                "structure_quality": float(item.structure_quality),
                "continuation_quality": float(item.continuation_quality),
                "risk_quality": float(item.risk_quality),
                "trade_quality": float(item.trade_quality),
                "xgb_score": 0.0,
            },
            lesson_summary=_learned_lessons,
            behavior_score=_behavior_score,
        )
        scan_candidates.append(ScanCandidate(
            symbol=str(candidate_packet["symbol"]),
            lane=str(candidate_packet["lane"]),
            momentum_5=float(candidate_packet["momentum_5"]),
            momentum_14=float(candidate_packet["momentum_14"]),
            momentum_30=float(item.momentum_30),
            rotation_score=float(candidate_packet["rotation_score"]),
            rsi=float(candidate_packet["rsi"]),
            volume=float(item.volume_usd),
            trend_1h=int(candidate_packet["trend_1h"]),
            regime_7d=str(candidate_packet["regime_7d"]),
            macro_30d=str(candidate_packet["macro_30d"]),
            entry_score=float(candidate_packet["entry_score"]),
            entry_recommendation=str(candidate_packet["entry_recommendation"]),
            reversal_risk=str(candidate_packet["reversal_risk"]),
            expected_move_pct=float(item.expected_move_pct),
            total_cost_pct=float(item.total_cost_pct),
            required_edge_pct=float(item.required_edge_pct),
            net_edge_pct=float(candidate_packet["net_edge_pct"]),
            tp_after_cost_valid=bool(item.tp_after_cost_valid),
            golden_profile_score=float(item.golden_profile_score),
            golden_profile_tag=str(item.golden_profile_tag),
            final_score=float(candidate_packet["final_score"]),
            reliability_bonus=float(candidate_packet["reliability_bonus"]),
            basket_fit_bonus=float(candidate_packet["basket_fit_bonus"]),
            score_breakdown=dict(candidate_packet.get("score_breakdown") or {}),
            breakdown_notes=list(candidate_packet.get("breakdown_notes") or []),
            spread_penalty=float(candidate_packet.get("spread_penalty", 0.0) or 0.0),
            cost_penalty=float(candidate_packet.get("cost_penalty", 0.0) or 0.0),
            correlation_penalty=float(candidate_packet.get("correlation_penalty", 0.0) or 0.0),
            behavior_score=candidate_packet.get("behavior_score"),
            lesson_summary=list(candidate_packet.get("lesson_summary") or []),
            bullish_divergence=bool(candidate_packet["bullish_divergence"]),
            bearish_divergence=bool(candidate_packet["bearish_divergence"]),
            divergence_strength=float(candidate_packet["divergence_strength"]),
            divergence_age_bars=int(candidate_packet["divergence_age_bars"]),
        ))
    scan = phi3_scan_market(
        scan_candidates,
        news_context=news_context,
        market_context={"candidate_count": len(candidates), "lane_meta": lane_meta},
        dex_context=dex_context,
        behavior_score=_behavior_score,
        lesson_summary=(scan_candidates[0].lesson_summary if scan_candidates else None),
        learned_lessons=_learned_lessons,
    )
    lane_supervision = [
        item.to_dict()
        for item in phi3_supervise_lanes(
            scan_candidates,
            news_context=news_context,
            dex_context=dex_context,
        )
    ]
    hot_candidates = [
        {
            "symbol": candidate.pair,
            "lane": candidate.lane,
            "candidate_score": candidate.candidate_score,
            "rank_score": candidate.rank_score,
            "rank_delta": candidate.rank_delta,
            "lane_rank_delta": candidate.lane_rank_delta,
            "momentum_delta": candidate.momentum_delta,
            "volume_ratio_delta": candidate.volume_ratio_delta,
            "leader_urgency": candidate.leader_urgency,
            "leader_takeover": candidate.leader_takeover,
            "recommendation": candidate.candidate_recommendation,
            "risk": candidate.candidate_risk,
            "expected_move_pct": candidate.expected_move_pct,
            "total_cost_pct": candidate.total_cost_pct,
            "required_edge_pct": candidate.required_edge_pct,
            "net_edge_pct": candidate.net_edge_pct,
            "tp_after_cost_valid": candidate.tp_after_cost_valid,
            "golden_profile_score": candidate.golden_profile_score,
            "golden_profile_tag": candidate.golden_profile_tag,
            "reasons": candidate.candidate_reasons or [],
        }
        for candidate in candidates
        if (
            candidate.candidate_recommendation in {"BUY", "STRONG_BUY"}
            and _scan_quality_ok(candidate)
            and candidate.candidate_score >= 55.0
        )
    ][:25]
    avoid_candidates = [
        {
            "symbol": candidate.pair,
            "lane": candidate.lane,
            "candidate_score": candidate.candidate_score,
            "rank_score": candidate.rank_score,
            "rank_delta": candidate.rank_delta,
            "lane_rank_delta": candidate.lane_rank_delta,
            "momentum_delta": candidate.momentum_delta,
            "volume_ratio": candidate.volume_ratio,
            "volume_ratio_delta": candidate.volume_ratio_delta,
            "momentum_5": candidate.momentum_5,
            "leader_urgency": candidate.leader_urgency,
            "leader_takeover": candidate.leader_takeover,
            "recommendation": candidate.candidate_recommendation,
            "risk": candidate.candidate_risk,
            "expected_move_pct": candidate.expected_move_pct,
            "total_cost_pct": candidate.total_cost_pct,
            "required_edge_pct": candidate.required_edge_pct,
            "net_edge_pct": candidate.net_edge_pct,
            "tp_after_cost_valid": candidate.tp_after_cost_valid,
            "golden_profile_score": candidate.golden_profile_score,
            "golden_profile_tag": candidate.golden_profile_tag,
            "reasons": candidate.candidate_reasons or [],
        }
        for candidate in sorted(candidates, key=lambda item: item.candidate_score)
        if candidate.candidate_recommendation == "AVOID"
    ][:10]
    return {
        "phi3_scan": scan.to_dict(),
        "lane_supervision": lane_supervision,
        "news_context": news_context,
        "dex_context": dex_context,
        "lane_shortlists": {
            lane: [
                {
                    "symbol": candidate.pair,
                    "lane": candidate.lane,
                    "candidate_score": candidate.candidate_score,
                    "rank_score": candidate.rank_score,
                    "rank_delta": candidate.rank_delta,
                    "lane_rank_delta": candidate.lane_rank_delta,
                    "momentum_5": candidate.momentum_5,
                    "momentum_delta": candidate.momentum_delta,
                    "volume_ratio": candidate.volume_ratio,
                    "volume_ratio_delta": candidate.volume_ratio_delta,
                    "structure_quality": candidate.structure_quality,
                    "momentum_quality": candidate.momentum_quality,
                    "trade_quality": candidate.trade_quality,
                    "risk_quality": candidate.risk_quality,
                    "expected_move_pct": candidate.expected_move_pct,
                    "total_cost_pct": candidate.total_cost_pct,
                    "required_edge_pct": candidate.required_edge_pct,
                    "net_edge_pct": candidate.net_edge_pct,
                    "tp_after_cost_valid": candidate.tp_after_cost_valid,
                    "golden_profile_score": candidate.golden_profile_score,
                    "golden_profile_tag": candidate.golden_profile_tag,
                    "leader_urgency": candidate.leader_urgency,
                    "leader_takeover": candidate.leader_takeover,
                    "recommendation": candidate.candidate_recommendation,
                    "risk": candidate.candidate_risk,
                    "reasons": candidate.candidate_reasons or [],
                }
                for candidate in candidates
                if candidate.pair in lane_meta.get(lane, [])
            ]
            for lane in {"L1", "L2", "L3", "L4"}
        },
        "lane_meta": lane_meta,
        "top_ranked": [candidate.pair for candidate in candidates[:25]],
        "leaderboard": [
            {
                "symbol": candidate.pair,
                "lane": candidate.lane,
                "rank_score": candidate.rank_score,
                "candidate_score": candidate.candidate_score,
                "rank_delta": candidate.rank_delta,
                "lane_rank_delta": candidate.lane_rank_delta,
                "leader_urgency": candidate.leader_urgency,
                "leader_takeover": candidate.leader_takeover,
                "net_edge_pct": candidate.net_edge_pct,
                "total_cost_pct": candidate.total_cost_pct,
                "golden_profile_score": candidate.golden_profile_score,
                "golden_profile_tag": candidate.golden_profile_tag,
            }
            for candidate in candidates[:25]
        ],
        "top_scored": [
            {
                "symbol": candidate.pair,
                "lane": candidate.lane,
                "candidate_score": candidate.candidate_score,
                "rank_score": candidate.rank_score,
                "rank_delta": candidate.rank_delta,
                "lane_rank_delta": candidate.lane_rank_delta,
                "momentum_5": candidate.momentum_5,
                "momentum_delta": candidate.momentum_delta,
                "volume_ratio": candidate.volume_ratio,
                "volume_ratio_delta": candidate.volume_ratio_delta,
                "structure_quality": candidate.structure_quality,
                "momentum_quality": candidate.momentum_quality,
                "trade_quality": candidate.trade_quality,
                "risk_quality": candidate.risk_quality,
                "expected_move_pct": candidate.expected_move_pct,
                "total_cost_pct": candidate.total_cost_pct,
                "required_edge_pct": candidate.required_edge_pct,
                "net_edge_pct": candidate.net_edge_pct,
                "tp_after_cost_valid": candidate.tp_after_cost_valid,
                "realized_cost_penalty_pct": candidate.realized_cost_penalty_pct,
                "realized_slippage_pct": candidate.realized_slippage_pct,
                "realized_follow_through_pct": candidate.realized_follow_through_pct,
                "golden_profile_score": candidate.golden_profile_score,
                "golden_profile_tag": candidate.golden_profile_tag,
                "leader_urgency": candidate.leader_urgency,
                "leader_takeover": candidate.leader_takeover,
                "recommendation": candidate.candidate_recommendation,
                "risk": candidate.candidate_risk,
                "reasons": candidate.candidate_reasons or [],
            }
            for candidate in candidates[:25]
        ],
        "hot_candidates": hot_candidates,
        "avoid_candidates": avoid_candidates,
    }


def rebalance_universe(candidates: list[Candidate], policy: UniversePolicy) -> dict:
    current = load_universe()
    active = [normalize_pair(pair) for pair in current.get("active_pairs", [])]
    held_symbols = {normalize_pair(pair) for pair in load_synced_position_symbols()}

    if bool(get_runtime_setting("UNIVERSE_LOCKED")):
        locked_core = _core_active_universe()
        stabilized_active = [
            pair for pair in active if pair in held_symbols or pair in locked_core
        ]
        return save_universe(stabilized_active, "universe_locked: stabilization filter applied", meta=current.get("meta", {}))

    # Filter by liquidity and price
    prelim_filtered = [
        c
        for c in candidates
        if c.volume_usd >= policy.min_volume_usd
        and c.last >= policy.min_price
        and c.spread_bps <= policy.max_spread_bps
        and _universe_symbol_allowed(c.pair)
    ]
    core_candidates = [candidate for candidate in prelim_filtered if normalize_pair(candidate.pair) in _core_active_universe()]
    conditional_universe = _conditional_universe()
    filtered = [
        candidate
        for candidate in prelim_filtered
        if (
            normalize_pair(candidate.pair) in _core_active_universe()
            or (
                normalize_pair(candidate.pair) in conditional_universe
                and _conditional_candidate_allowed(candidate, core_candidates)
            )
            or (
                normalize_pair(candidate.pair) not in conditional_universe
                and not _strict_seed_only_enabled()
                and _broad_candidate_allowed(candidate)
            )
        )
    ]

    valid_filtered_pairs = {candidate.pair for candidate in filtered}
    active = [pair for pair in active if pair in valid_filtered_pairs or pair in held_symbols]

    if not filtered:
        return save_universe(active, "no candidates passed filters", meta={"top_ranked": []})

    shortlist_size = max(policy.active_min, min(policy.active_max, int(os.getenv("ROTATION_SHORTLIST_SIZE", "10"))))
    lane_shortlists, merged_shortlist, lane_meta = build_lane_shortlists(filtered, shortlist_size=shortlist_size)
    candidate_pool = filtered
    score_map = {c.pair: max(c.rank_score, c.candidate_score) for c in candidate_pool}
    base_segment_tags = {candidate.pair: _segment_for_symbol(candidate.pair, candidate.lane) for candidate in filtered}
    segment_tags = {candidate.pair: _layer_for_candidate(candidate) for candidate in filtered}

    target_total = min(policy.active_max, max(policy.active_min, shortlist_size))
    meme_candidates = [candidate for candidate in candidate_pool if segment_tags.get(candidate.pair) == "meme"]
    core_candidates = [candidate for candidate in candidate_pool if segment_tags.get(candidate.pair) == "core"]
    momentum_candidates = [candidate for candidate in candidate_pool if segment_tags.get(candidate.pair) == "momentum"]
    recovery_candidates = [candidate for candidate in candidate_pool if segment_tags.get(candidate.pair) == "recovery"]
    meme_cap = min(_meme_universe_cap(), target_total, len(meme_candidates)) if _meme_universe_enabled() else 0
    non_meme_target = max(0, target_total - meme_cap)
    core_target = min(len(core_candidates), max(1 if core_candidates else 0, int(round(non_meme_target * 0.45)))) if non_meme_target else 0
    momentum_target = min(len(momentum_candidates), max(1 if momentum_candidates else 0, int(round(non_meme_target * 0.35)))) if non_meme_target else 0
    recovery_target = min(len(recovery_candidates), max(1 if recovery_candidates else 0, non_meme_target - core_target - momentum_target)) if non_meme_target else 0

    assigned_non_meme = core_target + momentum_target + recovery_target
    if assigned_non_meme < non_meme_target:
        for layer_name, layer_candidates in (("momentum", momentum_candidates), ("recovery", recovery_candidates), ("core", core_candidates)):
            if assigned_non_meme >= non_meme_target:
                break
            if layer_name == "momentum" and momentum_target < len(layer_candidates):
                take = min(non_meme_target - assigned_non_meme, len(layer_candidates) - momentum_target)
                momentum_target += take
                assigned_non_meme += take
            elif layer_name == "recovery" and recovery_target < len(layer_candidates):
                take = min(non_meme_target - assigned_non_meme, len(layer_candidates) - recovery_target)
                recovery_target += take
                assigned_non_meme += take
            elif layer_name == "core" and core_target < len(layer_candidates):
                take = min(non_meme_target - assigned_non_meme, len(layer_candidates) - core_target)
                core_target += take
                assigned_non_meme += take

    core_current = [pair for pair in active if segment_tags.get(pair, "core") == "core"]
    momentum_current = [pair for pair in active if segment_tags.get(pair, "core") == "momentum"]
    recovery_current = [pair for pair in active if segment_tags.get(pair, "core") == "recovery"]
    meme_current = [pair for pair in active if segment_tags.get(pair, _segment_for_symbol(pair)) == "meme"]

    core_next, core_adds, core_removes = _select_segment_active(
        segment_candidates=core_candidates,
        current_active=core_current,
        held_symbols=held_symbols,
        target_size=core_target,
        policy=policy,
    )
    momentum_next, momentum_adds, momentum_removes = _select_segment_active(
        segment_candidates=momentum_candidates,
        current_active=momentum_current,
        held_symbols=held_symbols,
        target_size=momentum_target,
        policy=policy,
    )
    recovery_next, recovery_adds, recovery_removes = _select_segment_active(
        segment_candidates=recovery_candidates,
        current_active=recovery_current,
        held_symbols=held_symbols,
        target_size=recovery_target,
        policy=policy,
    )
    meme_next, meme_adds, meme_removes = _select_segment_active(
        segment_candidates=meme_candidates,
        current_active=meme_current,
        held_symbols=held_symbols,
        target_size=meme_cap,
        policy=policy,
    )

    next_active = core_next + momentum_next + recovery_next + meme_next
    if len(next_active) < target_total:
        for candidate in candidate_pool:
            if candidate.pair in next_active:
                continue
            if segment_tags.get(candidate.pair) == "meme" and len([pair for pair in next_active if segment_tags.get(pair, _segment_for_symbol(pair)) == "meme"]) >= meme_cap:
                continue
            next_active.append(candidate.pair)
            if len(next_active) >= target_total:
                break

    next_active = sorted(dict.fromkeys(next_active), key=lambda p: score_map.get(p, 0.0), reverse=True)
    next_active = clamp_active_size(next_active, policy.active_max)

    adds = core_adds + momentum_adds + recovery_adds + meme_adds
    removes = core_removes + momentum_removes + recovery_removes + meme_removes
    reason = f"adds={adds} removes={removes}"
    meta = asyncio.run(build_scan_meta(candidate_pool, lane_meta))
    meta["segment_tags"] = segment_tags
    meta["base_segment_tags"] = base_segment_tags
    meta["pool_segments"] = {
        "core": [candidate.pair for candidate in core_candidates],
        "momentum": [candidate.pair for candidate in momentum_candidates],
        "recovery": [candidate.pair for candidate in recovery_candidates],
        "meme": [candidate.pair for candidate in meme_candidates],
        "store_of_value": [candidate.pair for candidate in filtered if base_segment_tags.get(candidate.pair) == "store_of_value"],
        "major": [candidate.pair for candidate in filtered if base_segment_tags.get(candidate.pair) == "major"],
        "core_alt": [candidate.pair for candidate in filtered if base_segment_tags.get(candidate.pair) == "core_alt"],
    }
    meta["segment_counts"] = {
        "target_total": target_total,
        "core_target": core_target,
        "momentum_target": momentum_target,
        "recovery_target": recovery_target,
        "meme_cap": meme_cap,
        "core_active": len([pair for pair in next_active if segment_tags.get(pair, "core") == "core"]),
        "momentum_active": len([pair for pair in next_active if segment_tags.get(pair, "core") == "momentum"]),
        "recovery_active": len([pair for pair in next_active if segment_tags.get(pair, "core") == "recovery"]),
        "meme_active": len([pair for pair in next_active if segment_tags.get(pair, _segment_for_symbol(pair)) == "meme"]),
        "held_symbols": sorted(held_symbols),
    }
    return save_universe(next_active, reason, meta=meta)


def get_top_gainers(
    client: KrakenRestClient,
    valid_pool: list[str],
    meta_map: dict,
    top_n: int = 15,
) -> list[tuple[str, float, float, float]]:
    """Return (pair, pct_change_today, vol_usd_24h, last_price) for top N gainers in valid_pool."""
    min_vol = float(os.getenv("MIN_QUOTE_VOLUME_USD", "100000"))
    gainers: list[tuple[str, float, float, float]] = []
    for batch in chunked(valid_pool, 20):
        ticker_pairs = [
            str(meta_map[to_kraken_symbol(p)].get("key") or to_kraken_symbol(p))
            for p in batch
            if to_kraken_symbol(p) in meta_map
        ]
        if not ticker_pairs:
            continue
        try:
            tick = client.get_ticker(ticker_pairs).get("result", {})
        except Exception:
            continue
        for p in batch:
            meta = meta_map.get(to_kraken_symbol(p), {})
            ticker_key = (meta.get("altname") or meta.get("wsname") or meta.get("key") or p).upper()
            t = tick.get(ticker_key) or tick.get(p) or tick.get(meta.get("key"))
            if not t:
                continue
            try:
                last = float(t["c"][0])
                open_price = float(t["o"])
                vol_base = float(t["v"][1])
                vol_usd = vol_base * last
                if open_price <= 0 or last <= 0 or vol_usd < min_vol:
                    continue
                pct = (last - open_price) / open_price * 100.0
                gainers.append((p, pct, vol_usd, last))
            except Exception:
                continue
    gainers.sort(key=lambda x: x[1], reverse=True)
    return gainers[:top_n]


def _run_once(client: KrakenRestClient, policy: UniversePolicy) -> None:
    pool = load_pair_pool()
    if not pool:
        print("pair pool is empty")
        return
    valid_pool, invalid_pool, _meta = validate_and_filter_bench(client, pool)
    if invalid_pool:
        print(
            f"invalid pairs in pool (not on Kraken USD): {len(invalid_pool)} "
            f"{invalid_pool}"
        )
    if not valid_pool:
        print("no valid pairs after Kraken validation")
        return
    valid_pool = [pair for pair in valid_pool if _universe_symbol_allowed(pair)]
    if not valid_pool:
        print("no valid pairs after stabilization universe filter")
        return
    if os.getenv("WRITE_VALIDATED_BENCH", "false").lower() == "true":
        out_file = Path(os.getenv("VALIDATED_BENCH_FILE", "pair_pool_usd.valid.txt"))
        out_file.write_text("\n".join(valid_pool) + "\n")
        print(f"wrote validated bench: {out_file} ({len(valid_pool)} pairs)")
    candidates = rank_pairs(client, valid_pool, _meta)

    # Inject top gainers — catches breakout coins the momentum scan missed
    top_n_gainers = int(os.getenv("TOP_GAINERS_INJECT_COUNT", "10"))
    min_gain_pct = float(os.getenv("TOP_GAINERS_MIN_PCT", "5.0"))
    gainers = get_top_gainers(client, valid_pool, _meta, top_n=top_n_gainers)
    existing_pairs = {c.pair for c in candidates}
    injected: list[str] = []
    for pair, pct_change, vol_usd, last_price in gainers:
        if pct_change < min_gain_pct:
            continue
        boosted_score = min(pct_change * 0.5, 80.0)
        if pair in existing_pairs:
            for c in candidates:
                if c.pair == pair:
                    c.rank_score = max(c.rank_score, boosted_score)
                    break
        else:
            candidates.append(Candidate(
                pair=pair,
                score=boosted_score,
                volume_usd=vol_usd,
                last=last_price,
                rank_score=boosted_score,
                candidate_score=min(boosted_score, 48.0),
                candidate_recommendation="WATCH",
            ))
            injected.append(f"{pair}(+{pct_change:.1f}%)")
    if injected:
        print(f"[universe_manager] gainers injected: {injected}")

    result = rebalance_universe(candidates, policy)
    print(result)


def main() -> None:
    import time
    load_dotenv()
    client = KrakenRestClient()
    policy = get_policy()
    loop_mode = os.getenv("UNIVERSE_LOOP_MODE", "false").lower() == "true"
    refresh_sec = _refresh_interval_sec()

    _run_once(client, policy)

    if loop_mode:
        print(f"[universe_manager] Loop mode active — refreshing every {refresh_sec // 60}m")
        while True:
            time.sleep(refresh_sec)
            try:
                print(f"[universe_manager] Refreshing universe...")
                _run_once(client, policy)
            except Exception as exc:
                print(f"[universe_manager] Refresh error: {exc}")


if __name__ == "__main__":
    main()
