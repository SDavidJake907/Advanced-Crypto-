from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from core.llm.client import phi3_advisory_chat
from core.llm.prompts import PHI3_REFLEX_SYSTEM_PROMPT


@dataclass
class ReflexDecision:
    reflex: str
    micro_state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def phi3_reflex(features: dict[str, Any]) -> ReflexDecision:
    try:
        raw = phi3_advisory_chat(features, system=PHI3_REFLEX_SYSTEM_PROMPT)
        parsed = json.loads(raw)
        return ReflexDecision(
            reflex=str(parsed.get("reflex", "allow")),
            micro_state=str(parsed.get("micro_state", "stable")),
            reason=str(parsed.get("reason", "phi3_reflex")),
        )
    except Exception:
        pass

    price = float(features.get("price", 0.0))
    volume = float(features.get("volume", 0.0))
    bar_ts = features.get("bar_ts")

    if price <= 0.0:
        return ReflexDecision(
            reflex="block",
            micro_state="data_integrity_issue",
            reason="invalid_price",
        )
    if bar_ts in (None, ""):
        return ReflexDecision(
            reflex="block",
            micro_state="data_integrity_issue",
            reason="missing_bar_timestamp",
        )
    if volume < 0.0:
        return ReflexDecision(
            reflex="block",
            micro_state="data_integrity_issue",
            reason="invalid_negative_volume",
        )
    return ReflexDecision(
        reflex="allow",
        micro_state="stable",
        reason="data_ok",
    )
