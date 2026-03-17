from __future__ import annotations

from typing import Any

from core.policy.entry_verifier import compute_entry_verification
from core.policy.lane_classifier import classify_lane
from core.policy.lane_filters import apply_lane_filters
from core.policy.regime_state import update_regime_state
from core.policy.verdict import build_policy_verdict


def apply_policy_pipeline(symbol: str, features: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(features)
    enriched["lane"] = str(enriched.get("lane") or classify_lane(symbol, enriched)).upper()
    lane_filter = apply_lane_filters(enriched)
    regime_state = update_regime_state(symbol, enriched)
    enriched.update(regime_state)
    entry_verification = compute_entry_verification(
        {
            **enriched,
            "lane_filter_pass": lane_filter.passed,
            "lane_filter_reason": lane_filter.reason,
            "lane_filter_severity": lane_filter.severity,
        }
    )
    verdict = build_policy_verdict(
        lane_filter=lane_filter,
        regime_state=regime_state,
        entry_verification=entry_verification,
    )
    enriched.update(verdict.to_dict())
    enriched["policy_verdict"] = verdict.to_dict()
    return enriched
