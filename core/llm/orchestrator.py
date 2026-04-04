from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

from core.llm.micro_prompts import deterministic_market_state_review
from core.llm.phi3_reflex import phi3_reflex

@dataclass
class AdvisoryBundle:
    reflex: dict[str, Any]
    visual_review: dict[str, Any]
    market_state_review: dict[str, Any]
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_advisory_bundle(
    *,
    symbol: str,
    features: dict[str, Any],
    universe_context: dict[str, Any] | None = None,
    market_state_review: dict[str, Any] | None = None,
) -> AdvisoryBundle:
    t_phi3_start = time.perf_counter()
    reflex = phi3_reflex(features).to_dict()
    visual_review: dict[str, Any] = {}
    resolved_market_state_review = (
        dict(market_state_review)
        if isinstance(market_state_review, dict)
        else deterministic_market_state_review(features).to_dict()
    )
    phi3_ms = (time.perf_counter() - t_phi3_start) * 1000.0

    advisory_ms = 0.0

    return AdvisoryBundle(
        reflex=reflex,
        visual_review=visual_review,
        market_state_review=resolved_market_state_review,
        timings={
            "phi3_ms": round(phi3_ms, 2),
            "advisory_ms": round(advisory_ms, 2),
        },
    )
