from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from core.llm.client import phi3_chat
from core.llm.micro_prompts import INTERNAL_TO_PUBLIC_LANE, PUBLIC_TO_INTERNAL_LANE, phi3_supervise_lane
from core.llm.prompts import PHI3_LANE_SUPERVISOR_SYSTEM_PROMPT, PHI3_SCAN_SYSTEM_PROMPT


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _heuristic_scan(candidates: list[ScanCandidate]) -> MarketScanResult:
    ranked = sorted(candidates, key=lambda item: item.rotation_score, reverse=True)
    watchlist: list[WatchlistItem] = []
    for candidate in ranked[:10]:
        tag = "rotation_candidate"
        if candidate.lane == "L4" and candidate.momentum_5 > 0:
            tag = "meme_breakout"
        elif candidate.momentum_5 > candidate.momentum_14 > 0:
            tag = "early_acceleration"
        elif candidate.momentum_14 > 0 and candidate.trend_1h > 0:
            tag = "trend_continuation"
        confidence = min(max(candidate.rotation_score + 0.5, 0.0), 1.0)
        watchlist.append(
            WatchlistItem(
                symbol=candidate.symbol,
                tag=tag,
                confidence=confidence,
                reason="heuristic_scan",
            )
        )
    return MarketScanResult(watchlist=watchlist, market_note="heuristic_scan")


def phi3_scan_market(
    candidates: list[ScanCandidate],
    *,
    news_context: dict[str, Any] | None = None,
    market_context: dict[str, Any] | None = None,
    dex_context: dict[str, Any] | None = None,
) -> MarketScanResult:
    payload = {
        "candidates": [candidate.to_dict() for candidate in candidates[:25]],
        "news_context": news_context or {},
        "market_context": market_context or {},
        "dex_context": dex_context or {},
    }
    try:
        raw = phi3_chat(payload, system=PHI3_SCAN_SYSTEM_PROMPT, max_tokens=400)
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
        if candidate.lane == "L4" or (candidate.momentum_5 > 0.01 and candidate.volume > 0 and candidate.rsi >= 70):
            lane_candidate = "L4"
            narrative = "attention_breakout"
        elif candidate.trend_1h > 0 and candidate.momentum_14 > 0 and candidate.momentum_30 > 0:
            lane_candidate = "L1"
            narrative = "trend_continuation"
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
        results.append(
            LaneSupervisorResult(
                symbol=candidate.symbol,
                lane_candidate=PUBLIC_TO_INTERNAL_LANE.get(advice.lane_candidate, "L3"),
                lane_confidence=advice.lane_confidence,
                lane_reason=advice.reason,
                lane_conflict=advice.lane_conflict,
                narrative_tag=advice.narrative_tag,
            )
        )
    return results or _heuristic_lane_supervision(candidates)
