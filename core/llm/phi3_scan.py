from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from core.llm.client import phi3_advisory_chat
from core.llm.micro_prompts import INTERNAL_TO_PUBLIC_LANE, PUBLIC_TO_INTERNAL_LANE, phi3_supervise_lane
from core.llm.prompts import PHI3_SCAN_SYSTEM_PROMPT


@dataclass
class ScanCandidate:
    symbol: str
    lane: str
    momentum_5: float
    momentum_14: float
    momentum_30: float
    rotation_score: float
    rsi: float
    volume: float
    trend_1h: int
    regime_7d: str
    macro_30d: str
    entry_score: float = 0.0
    entry_recommendation: str = "WATCH"
    reversal_risk: str = "MEDIUM"
    expected_move_pct: float = 0.0
    total_cost_pct: float = 0.0
    required_edge_pct: float = 0.0
    net_edge_pct: float = 0.0
    tp_after_cost_valid: bool = False
    golden_profile_score: float = 0.0
    golden_profile_tag: str = "neutral"
    # Wave 3 — composite scoring (same data Nemo receives)
    final_score: float = 0.0
    reliability_bonus: float = 0.0
    basket_fit_bonus: float = 0.0
    score_breakdown: dict[str, Any] | None = None
    breakdown_notes: list[str] | None = None
    spread_penalty: float = 0.0
    cost_penalty: float = 0.0
    correlation_penalty: float = 0.0
    behavior_score: Any = None
    lesson_summary: list[str] | None = None
    bullish_divergence: bool = False
    bearish_divergence: bool = False
    divergence_strength: float = 0.0
    divergence_age_bars: int = 99

    @property
    def candidate_score(self) -> float:
        return self.entry_score

    @candidate_score.setter
    def candidate_score(self, value: float) -> None:
        self.entry_score = value

    @property
    def candidate_recommendation(self) -> str:
        return self.entry_recommendation

    @candidate_recommendation.setter
    def candidate_recommendation(self, value: str) -> None:
        self.entry_recommendation = value

    @property
    def candidate_risk(self) -> str:
        return self.reversal_risk

    @candidate_risk.setter
    def candidate_risk(self, value: str) -> None:
        self.reversal_risk = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "lane": self.lane,
            "momentum_5": self.momentum_5,
            "momentum_14": self.momentum_14,
            "momentum_30": self.momentum_30,
            "rotation_score": self.rotation_score,
            "rsi": self.rsi,
            "volume": self.volume,
            "trend_1h": self.trend_1h,
            "regime_7d": self.regime_7d,
            "macro_30d": self.macro_30d,
            "entry_score": self.entry_score,
            "entry_recommendation": self.entry_recommendation,
            "reversal_risk": self.reversal_risk,
            "expected_move_pct": self.expected_move_pct,
            "total_cost_pct": self.total_cost_pct,
            "required_edge_pct": self.required_edge_pct,
            "net_edge_pct": self.net_edge_pct,
            "tp_after_cost_valid": self.tp_after_cost_valid,
            "final_score": self.final_score,
            "reliability_bonus": self.reliability_bonus,
            "basket_fit_bonus": self.basket_fit_bonus,
            "score_breakdown": self.score_breakdown,
            "breakdown_notes": self.breakdown_notes,
            "spread_penalty": self.spread_penalty,
            "cost_penalty": self.cost_penalty,
            "correlation_penalty": self.correlation_penalty,
            "behavior_score": self.behavior_score,
            "lesson_summary": self.lesson_summary,
            "bullish_divergence": self.bullish_divergence,
            "bearish_divergence": self.bearish_divergence,
            "divergence_strength": self.divergence_strength,
            "divergence_age_bars": self.divergence_age_bars,
        }


@dataclass
class WatchlistItem:
    symbol: str
    tag: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketScanResult:
    watchlist: list[WatchlistItem]
    market_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "watchlist": [item.to_dict() for item in self.watchlist],
            "market_note": self.market_note,
        }


@dataclass
class LaneSupervisorResult:
    symbol: str
    lane_candidate: str
    lane_confidence: float
    lane_reason: str
    lane_conflict: bool
    narrative_tag: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ADJACENT_LANE_PAIRS = {
    ("L1", "L3"),
    ("L3", "L1"),
    ("L3", "L2"),
    ("L2", "L3"),
}


def _is_major_symbol(symbol: str) -> bool:
    return symbol.upper() in {
        "BTC/USD",
        "ETH/USD",
        "XRP/USD",
        "SOL/USD",
        "ADA/USD",
        "AVAX/USD",
        "LINK/USD",
        "DOGE/USD",
    }


def _is_strong_l4_candidate(candidate: ScanCandidate) -> bool:
    return (
        candidate.momentum_5 >= 0.015
        and candidate.rotation_score >= 0.65
        and candidate.rsi >= 68.0
        and candidate.volume > 0.0
    )


def _calibrated_lane_conflict(
    *,
    assigned_lane: str,
    candidate_lane: str,
    confidence: float,
    symbol: str,
) -> bool:
    if assigned_lane == candidate_lane:
        return False
    if confidence < 0.9:
        return False
    if (assigned_lane, candidate_lane) in _ADJACENT_LANE_PAIRS:
        return False
    if candidate_lane == "L4" and _is_major_symbol(symbol):
        return False
    return True


def _apply_lane_supervision_guardrails(candidate: ScanCandidate, lane_candidate: str, confidence: float, narrative: str) -> tuple[str, float, str]:
    # Keep majors from collapsing into L4 unless acceleration is genuinely strong.
    if lane_candidate == "L4" and candidate.lane != "L4":
        if _is_major_symbol(candidate.symbol) and not _is_strong_l4_candidate(candidate):
            return candidate.lane, min(confidence, 0.79), "trend_continuation" if candidate.trend_1h > 0 else "moderate_trend"
        if not _is_strong_l4_candidate(candidate):
            return candidate.lane, min(confidence, 0.84), narrative
    return lane_candidate, confidence, narrative


def _heuristic_scan(candidates: list[ScanCandidate]) -> MarketScanResult:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.final_score,
            item.net_edge_pct,
            item.momentum_14,
            item.volume,
        ),
        reverse=True,
    )
    watchlist: list[WatchlistItem] = []
    for candidate in ranked:
        recommendation = str(candidate.entry_recommendation or "WATCH").upper()
        risk = str(candidate.reversal_risk or "MEDIUM").upper()
        if recommendation == "AVOID" or risk == "HIGH":
            continue
        if not candidate.tp_after_cost_valid:
            continue
        if candidate.final_score < 55.0:
            continue
        if candidate.net_edge_pct < 0.35:
            continue
        if candidate.volume < 250_000.0:
            continue
        tag = "rotation_candidate"
        if candidate.lane == "L4" and candidate.momentum_5 > 0:
            tag = "meme_breakout"
        elif candidate.momentum_5 > candidate.momentum_14 > 0:
            tag = "early_acceleration"
        elif candidate.momentum_14 > 0 and candidate.trend_1h > 0:
            tag = "trend_continuation"
        confidence = min(
            max(
                (candidate.final_score / 100.0)
                + 0.05
                + min(candidate.net_edge_pct / 6.0, 0.25)
                + (0.08 if candidate.trend_1h > 0 and candidate.momentum_14 > 0 else 0.0),
                0.0,
            ),
            1.0,
        )
        watchlist.append(
            WatchlistItem(
                symbol=candidate.symbol,
                tag=tag,
                confidence=confidence,
                reason="heuristic_scan:final_score",
            )
        )
        if len(watchlist) >= 10:
            break
    return MarketScanResult(watchlist=watchlist, market_note="heuristic_scan")


def _normalize_lesson_summary(lesson_summary: list[str] | None = None, learned_lessons: str | None = None) -> list[str]:
    if lesson_summary:
        return [str(item).strip() for item in lesson_summary if str(item).strip()][:6]
    if learned_lessons:
        items: list[str] = []
        for line in str(learned_lessons).splitlines():
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


def phi3_scan_market(
    candidates: list[ScanCandidate],
    *,
    news_context: dict[str, Any] | None = None,
    market_context: dict[str, Any] | None = None,
    dex_context: dict[str, Any] | None = None,
    behavior_score: str | None = None,
    lesson_summary: list[str] | None = None,
    learned_lessons: str | None = None,
) -> MarketScanResult:
    normalized_lesson_summary = _normalize_lesson_summary(lesson_summary=lesson_summary, learned_lessons=learned_lessons)
    payload = {
        "candidates": [candidate.to_dict() for candidate in candidates[:25]],
        "news_context": news_context or {},
        "market_context": market_context or {},
        "dex_context": dex_context or {},
    }
    if behavior_score:
        payload["behavior_score"] = behavior_score
    if normalized_lesson_summary:
        payload["lesson_summary"] = normalized_lesson_summary
    try:
        raw = phi3_advisory_chat(payload, system=PHI3_SCAN_SYSTEM_PROMPT, max_tokens=400)
        parsed = json.loads(raw)
        items = []
        for item in parsed.get("watchlist", [])[:10]:
            if not isinstance(item, dict):
                continue
            items.append(
                WatchlistItem(
                    symbol=str(item.get("symbol", "")),
                    tag=str(item.get("tag", "scan_candidate")),
                    confidence=float(item.get("confidence", 0.0)),
                    reason=str(item.get("reason", "phi3_scan")),
                )
            )
        return MarketScanResult(
            watchlist=items,
            market_note=str(parsed.get("market_note", "")),
        )
    except Exception:
        return _heuristic_scan(candidates)


def _heuristic_lane_supervision(
    candidates: list[ScanCandidate],
) -> list[LaneSupervisorResult]:
    results: list[LaneSupervisorResult] = []
    for candidate in candidates[:15]:
        lane_candidate = candidate.lane
        narrative = "balanced_setup"
        conflict = False
        if candidate.lane == "L4" or _is_strong_l4_candidate(candidate):
            lane_candidate = "L4"
            narrative = "attention_breakout"
        elif candidate.trend_1h > 0 and candidate.momentum_14 > 0 and candidate.momentum_30 > 0:
            lane_candidate = "L1"
            narrative = "trend_continuation"
        elif candidate.lane == "L2":
            # Preserve L2 identity — rotation candidates should stay L2 unless clearly L1/L4
            lane_candidate = "L2"
            narrative = "reversion_candidate"
        elif candidate.regime_7d == "choppy" and candidate.rsi < 40:
            lane_candidate = "L2"
            narrative = "reversion_candidate"
        else:
            lane_candidate = "L3"
            narrative = "moderate_trend"

        if candidate.lane != lane_candidate:
            conflict = True

        confidence = 0.55
        if candidate.rotation_score > 0.15:
            confidence += 0.15
        if candidate.momentum_5 > 0 and candidate.momentum_14 > 0:
            confidence += 0.1
        confidence = min(max(confidence, 0.0), 1.0)
        lane_candidate, confidence, narrative = _apply_lane_supervision_guardrails(
            candidate,
            lane_candidate,
            confidence,
            narrative,
        )
        conflict = _calibrated_lane_conflict(
            assigned_lane=candidate.lane,
            candidate_lane=lane_candidate,
            confidence=confidence,
            symbol=candidate.symbol,
        )

        results.append(
            LaneSupervisorResult(
                symbol=candidate.symbol,
                lane_candidate=lane_candidate,
                lane_confidence=confidence,
                lane_reason="heuristic_lane_supervision",
                lane_conflict=conflict,
                narrative_tag=narrative,
            )
        )
    return results


def phi3_supervise_lanes(
    candidates: list[ScanCandidate],
    *,
    news_context: dict[str, Any] | None = None,
    dex_context: dict[str, Any] | None = None,
) -> list[LaneSupervisorResult]:
    results: list[LaneSupervisorResult] = []
    for candidate in candidates[:15]:
        advice = phi3_supervise_lane(
            {
                **candidate.to_dict(),
                "lane": INTERNAL_TO_PUBLIC_LANE.get(candidate.lane, "main"),
            },
            news_context=news_context,
            dex_context=dex_context,
        )
        lane_candidate = PUBLIC_TO_INTERNAL_LANE.get(advice.lane_candidate, "L3")
        confidence = advice.lane_confidence
        narrative = advice.narrative_tag
        lane_candidate, confidence, narrative = _apply_lane_supervision_guardrails(
            candidate,
            lane_candidate,
            confidence,
            narrative,
        )
        results.append(
            LaneSupervisorResult(
                symbol=candidate.symbol,
                lane_candidate=lane_candidate,
                lane_confidence=confidence,
                lane_reason=advice.reason,
                lane_conflict=_calibrated_lane_conflict(
                    assigned_lane=candidate.lane,
                    candidate_lane=lane_candidate,
                    confidence=confidence,
                    symbol=candidate.symbol,
                ),
                narrative_tag=narrative,
            )
        )
    return results or _heuristic_lane_supervision(candidates)
