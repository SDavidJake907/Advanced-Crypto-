from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AggressionRecommendation:
    mode: str
    confidence: float
    score: float
    summary: str
    reasons: list[str]
    current_mode: str
    btc_dominance_level: str
    alt_market_posture: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _latest_market_trend(recent_decision_debug: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(recent_decision_debug):
        if not isinstance(item, dict):
            continue
        trend = str(item.get("market_trend", "") or "").strip().lower()
        if trend:
            return trend, _safe_int(item.get("market_trend_strength", 0), 0)
    return "unknown", 0


def _decision_activity(recent_decision_debug: list[dict[str, Any]]) -> tuple[float, float]:
    decision_rows = [
        item for item in recent_decision_debug
        if isinstance(item, dict)
        and str(item.get("symbol", "") or "").strip()
        and str(item.get("loop", "") or "").strip() == ""
    ]
    if not decision_rows:
        return 0.0, 0.0
    flat_count = sum(1 for item in decision_rows if str(item.get("signal", "") or "").upper() == "FLAT")
    live_open_count = sum(
        1
        for item in decision_rows
        if str(item.get("execution_status", "") or "").lower() in {"submitted", "filled"}
    )
    total = float(len(decision_rows))
    return flat_count / total, live_open_count / total


def _btc_dominance_level(btc_dom: float) -> str:
    if btc_dom <= 52.0 and btc_dom > 0.0:
        return "alt_tailwind"
    if btc_dom <= 55.0 and btc_dom > 0.0:
        return "neutral"
    if btc_dom <= 58.0:
        return "alt_caution"
    return "alt_high_caution"


def _alt_market_posture(*, btc_dom: float, fng: int, market_cap_change: float, market_trend: str) -> str:
    caution_score = 0
    if _btc_dominance_level(btc_dom) == "alt_caution":
        caution_score += 1
    elif _btc_dominance_level(btc_dom) == "alt_high_caution":
        caution_score += 2
    if fng <= 25:
        caution_score += 1
    elif fng >= 55:
        caution_score -= 1
    if market_cap_change <= -1.0:
        caution_score += 1
    elif market_cap_change >= 1.0:
        caution_score -= 1
    if market_trend == "bear":
        caution_score += 1
    elif market_trend == "bull":
        caution_score -= 1

    if caution_score >= 3:
        return "fragile_alts"
    if caution_score >= 1:
        return "cautious_alts"
    if caution_score <= -2:
        return "constructive_alts"
    return "mixed_alts"


def recommend_aggression_mode(
    *,
    runtime_values: dict[str, Any],
    universe_meta: dict[str, Any],
    recent_decision_debug: list[dict[str, Any]],
) -> AggressionRecommendation:
    current_mode = str(runtime_values.get("AGGRESSION_MODE", "NORMAL") or "NORMAL").upper()
    news_context = universe_meta.get("news_context", {}) if isinstance(universe_meta.get("news_context", {}), dict) else {}
    btc_dom = _safe_float(news_context.get("btc_dominance"), 0.0)
    fng = _safe_int(news_context.get("fng_value"), 50)
    market_cap_change = _safe_float(news_context.get("market_cap_change_24h"), 0.0)
    market_trend, market_strength = _latest_market_trend(recent_decision_debug)
    flat_rate, open_rate = _decision_activity(recent_decision_debug)
    btc_dom_level = _btc_dominance_level(btc_dom)
    alt_market_posture = _alt_market_posture(
        btc_dom=btc_dom,
        fng=fng,
        market_cap_change=market_cap_change,
        market_trend=market_trend,
    )

    score = 0.0
    reasons: list[str] = []

    if market_trend == "bear":
        score -= 2.0
        reasons.append(f"btc_market_bear({market_strength:+d})")
        if market_strength <= -3:
            score -= 1.0
            reasons.append("btc_market_bear_strong")
    elif market_trend == "bull":
        score += 2.0
        reasons.append(f"btc_market_bull({market_strength:+d})")
        if market_strength >= 3:
            score += 1.0
            reasons.append("btc_market_bull_strong")

    if btc_dom_level == "alt_high_caution":
        score += 1.5
        reasons.append(f"btc_dom_high_lead({btc_dom:.1f})")
    elif btc_dom_level == "alt_caution":
        score += 0.75
        reasons.append(f"btc_dom_lead({btc_dom:.1f})")
    elif btc_dom_level == "alt_tailwind":
        score += 0.5
        reasons.append(f"btc_dom_easing({btc_dom:.1f})")

    if fng <= 25:
        score -= 1.0
        reasons.append(f"fear_greed_fear({fng})")
    elif 45 <= fng <= 75:
        score += 0.5
        reasons.append(f"fear_greed_constructive({fng})")
    elif fng >= 85:
        score -= 0.5
        reasons.append(f"fear_greed_greedy({fng})")

    if market_cap_change <= -1.0:
        score -= 1.0
        reasons.append(f"market_cap_down({market_cap_change:+.2f})")
    elif market_cap_change >= 1.0:
        score += 1.0
        reasons.append(f"market_cap_up({market_cap_change:+.2f})")

    if flat_rate >= 0.85 and market_trend != "bull":
        score -= 0.5
        reasons.append(f"high_flat_rate({flat_rate:.0%})")
    if open_rate >= 0.20 and market_trend == "bull":
        score += 0.5
        reasons.append(f"healthy_open_rate({open_rate:.0%})")

    if alt_market_posture == "fragile_alts":
        score -= 1.0
        reasons.append("alt_market_fragile")
    elif alt_market_posture == "cautious_alts":
        score -= 0.5
        reasons.append("alt_market_cautious")
    elif alt_market_posture == "constructive_alts":
        score += 0.75
        reasons.append("alt_market_constructive")

    if score <= -2.5:
        mode = "DEFENSIVE"
    elif score >= 3.5:
        mode = "HIGH_OFFENSIVE"
    elif score >= 1.5:
        mode = "OFFENSIVE"
    else:
        mode = "NORMAL"

    confidence = min(0.95, max(0.35, 0.45 + min(abs(score), 4.0) * 0.1))
    summary = (
        f"deterministic aggression review recommends {mode} "
        f"(current={current_mode}, score={score:+.1f}, trend={market_trend}, "
        f"btc_dom={btc_dom:.1f}/{btc_dom_level}, alts={alt_market_posture}, "
        f"fng={fng}, mcap24h={market_cap_change:+.2f})"
    )
    return AggressionRecommendation(
        mode=mode,
        confidence=round(confidence, 2),
        score=round(score, 2),
        summary=summary,
        reasons=reasons,
        current_mode=current_mode,
        btc_dominance_level=btc_dom_level,
        alt_market_posture=alt_market_posture,
    )
