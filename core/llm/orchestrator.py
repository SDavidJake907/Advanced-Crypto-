from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any

from core.config.runtime import get_runtime_setting
from core.llm.client import advisory_provider_name
from core.llm.micro_prompts import deterministic_market_state_review
from core.llm.phi3_reflex import phi3_reflex

VISUAL_FEED_LOG = Path("logs/visual_phi3_feed.jsonl")
VISUAL_REVIEW_MAX_AGE_SEC = 120


@dataclass
class AdvisoryBundle:
    reflex: dict[str, Any]
    visual_review: dict[str, Any]
    market_state_review: dict[str, Any]
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_recent_visual_review(max_age_sec: int = VISUAL_REVIEW_MAX_AGE_SEC) -> dict[str, Any]:
    if not VISUAL_FEED_LOG.exists():
        return {}
    try:
        lines = VISUAL_FEED_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_sec)
    for line in reversed(lines[-25:]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "visual_review" or not payload.get("review_ok"):
            continue
        ts = payload.get("ts")
        try:
            parsed_ts = datetime.fromisoformat(str(ts))
        except Exception:
            parsed_ts = None
        if parsed_ts is not None and parsed_ts.tzinfo is None:
            parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
        if parsed_ts is not None and parsed_ts < cutoff:
            continue
        review = payload.get("review", {})
        if not isinstance(review, dict):
            review = {}
        return {
            "chart_state": review.get("chart_state"),
            "setup_quality": review.get("setup_quality"),
            "visual_risk": review.get("visual_risk"),
            "range_warning": review.get("range_warning"),
            "overextension_warning": review.get("overextension_warning"),
            "support_resistance_note": review.get("support_resistance_note"),
            "reason": review.get("reason"),
            "latency_ms": review.get("latency_ms"),
            "device": review.get("device"),
            "captured_at": ts,
        }
    return {}


def build_advisory_bundle(
    *,
    symbol: str,
    features: dict[str, Any],
    universe_context: dict[str, Any] | None = None,
) -> AdvisoryBundle:
    entry_recommendation = str(features.get("entry_recommendation", "")).upper()
    reversal_risk = str(features.get("reversal_risk", "")).upper()
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    trend_confirmed = bool(features.get("trend_confirmed"))
    top_candidate = bool((universe_context or {}).get("current_symbol_is_top_candidate"))
    advisory_min_entry_score = float(get_runtime_setting("ADVISORY_MIN_ENTRY_SCORE"))
    advisory_min_volume_ratio = float(get_runtime_setting("ADVISORY_MIN_VOLUME_RATIO"))
    enable_visual_context = bool(get_runtime_setting("ADVISORY_ENABLE_VISUAL_CONTEXT"))
    advisory_provider = advisory_provider_name()

    top_candidate_quality = (
        top_candidate
        and entry_recommendation != "AVOID"
        and reversal_risk != "HIGH"
        and (
            entry_score >= max(advisory_min_entry_score - 8.0, 48.0)
            or volume_ratio >= advisory_min_volume_ratio
        )
        and (rotation_score > 0.0 or momentum_5 > 0.0 or trend_confirmed)
    )

    t_phi3_start = time.perf_counter()
    reflex = phi3_reflex(features).to_dict()
    visual_review = _load_recent_visual_review() if (enable_visual_context and advisory_provider == "phi3" and top_candidate_quality) else {}
    market_state_review = deterministic_market_state_review(features).to_dict()
    phi3_ms = (time.perf_counter() - t_phi3_start) * 1000.0

    advisory_ms = 0.0

    return AdvisoryBundle(
        reflex=reflex,
        visual_review=visual_review,
        market_state_review=market_state_review,
        timings={
            "phi3_ms": round(phi3_ms, 2),
            "advisory_ms": round(advisory_ms, 2),
        },
    )
