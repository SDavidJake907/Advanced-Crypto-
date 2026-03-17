from __future__ import annotations

from dataclasses import asdict, dataclass

from core.policy.pipeline import apply_policy_pipeline
from core.policy.verdict import extract_policy_verdict


@dataclass
class CandidateScoreResult:
    candidate_score: float
    candidate_recommendation: str
    candidate_risk: str
    candidate_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def score_candidate(
    *,
    symbol: str = "UNKNOWN/USD",
    lane: str,
    momentum_5: float,
    momentum_14: float,
    momentum_30: float,
    trend_1h: int,
    volume_ratio: float,
    price_zscore: float,
    rsi: float,
    spread_bps: float,
    rotation_score: float,
) -> CandidateScoreResult:
    policy = apply_policy_pipeline(
        symbol,
        {
            "symbol": symbol,
            "lane": lane,
            "momentum": momentum_14,
            "momentum_5": momentum_5,
            "momentum_14": momentum_14,
            "momentum_30": momentum_30,
            "trend_1h": trend_1h,
            "volume_ratio": volume_ratio,
            "price_zscore": price_zscore,
            "rsi": rsi,
            "spread_pct": spread_bps / 100.0,
            "rotation_score": rotation_score,
            "regime_7d": "unknown",
            "macro_30d": "sideways",
            "price": 1.0,
            "atr": 0.0,
            "bb_bandwidth": 0.02,
            "market_regime": {"breadth": 0.5, "rv_mkt": 0.0},
            "correlation_row": [],
            "hurst": 0.5,
            "autocorr": 0.0,
            "entropy": 0.5,
        },
    )
    verdict = extract_policy_verdict(policy)
    reasons = list(verdict.entry_reasons)
    lane_filter_reason = verdict.lane_filter_reason
    if lane_filter_reason and lane_filter_reason not in reasons:
        reasons.append(lane_filter_reason)
    regime_state = verdict.regime_state
    if regime_state:
        reasons.append(regime_state)
    return CandidateScoreResult(
        candidate_score=verdict.entry_score,
        candidate_recommendation=verdict.entry_recommendation,
        candidate_risk=verdict.reversal_risk,
        candidate_reasons=reasons,
    )
