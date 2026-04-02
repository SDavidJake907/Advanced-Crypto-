from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.policy.lane_filters import LaneFilterResult


@dataclass
class PolicyVerdict:
    lane: str
    lane_filter_pass: bool
    lane_filter_reason: str
    lane_filter_severity: str
    regime_raw: str
    regime_state: str
    regime_confirm_count: int
    regime_dwell: int
    entry_score: float
    entry_recommendation: str
    reversal_risk: str
    promotion_tier: str
    promotion_reason: str
    entry_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_policy_verdict(
    *,
    lane_filter: LaneFilterResult,
    regime_state: dict[str, Any],
    entry_verification: dict[str, Any],
) -> PolicyVerdict:
    entry_reasons = [str(reason) for reason in entry_verification.get("entry_reasons", [])]
    return PolicyVerdict(
        lane=lane_filter.lane,
        lane_filter_pass=lane_filter.passed,
        lane_filter_reason=lane_filter.reason,
        lane_filter_severity=lane_filter.severity,
        regime_raw=str(regime_state.get("regime_raw", "unknown") or "unknown"),
        regime_state=str(regime_state.get("regime_state", "unknown") or "unknown"),
        regime_confirm_count=int(regime_state.get("regime_confirm_count", 0) or 0),
        regime_dwell=int(regime_state.get("regime_dwell", 0) or 0),
        entry_score=float(entry_verification.get("entry_score", 0.0) or 0.0),
        entry_recommendation=str(entry_verification.get("entry_recommendation", "WATCH") or "WATCH"),
        reversal_risk=str(entry_verification.get("reversal_risk", "MEDIUM") or "MEDIUM"),
        promotion_tier=str(entry_verification.get("promotion_tier", "skip") or "skip"),
        promotion_reason=str(entry_verification.get("promotion_reason", "not_qualified") or "not_qualified"),
        entry_reasons=list(dict.fromkeys(entry_reasons)),
    )


def extract_policy_verdict(features: dict[str, Any]) -> PolicyVerdict:
    payload = features.get("policy_verdict", {})
    if isinstance(payload, PolicyVerdict):
        return payload
    if isinstance(payload, dict):
        source = payload
    else:
        source = features
    entry_reasons = [str(reason) for reason in source.get("entry_reasons", features.get("entry_reasons", []))]
    return PolicyVerdict(
        lane=str(source.get("lane", features.get("lane", "L3")) or "L3"),
        lane_filter_pass=bool(source.get("lane_filter_pass", features.get("lane_filter_pass", True))),
        lane_filter_reason=str(source.get("lane_filter_reason", features.get("lane_filter_reason", "")) or ""),
        lane_filter_severity=str(source.get("lane_filter_severity", features.get("lane_filter_severity", "ok")) or "ok"),
        regime_raw=str(source.get("regime_raw", features.get("regime_raw", "unknown")) or "unknown"),
        regime_state=str(source.get("regime_state", features.get("regime_state", "unknown")) or "unknown"),
        regime_confirm_count=int(source.get("regime_confirm_count", features.get("regime_confirm_count", 0)) or 0),
        regime_dwell=int(source.get("regime_dwell", features.get("regime_dwell", 0)) or 0),
        entry_score=float(source.get("entry_score", features.get("entry_score", 0.0)) or 0.0),
        entry_recommendation=str(
            source.get("entry_recommendation", features.get("entry_recommendation", "WATCH")) or "WATCH"
        ),
        reversal_risk=str(source.get("reversal_risk", features.get("reversal_risk", "MEDIUM")) or "MEDIUM"),
        promotion_tier=str(source.get("promotion_tier", features.get("promotion_tier", "skip")) or "skip"),
        promotion_reason=str(
            source.get("promotion_reason", features.get("promotion_reason", "not_qualified")) or "not_qualified"
        ),
        entry_reasons=list(dict.fromkeys(entry_reasons)),
    )
