from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from core.llm.client import phi3_chat
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
        raw = phi3_chat(features, system=PHI3_REFLEX_SYSTEM_PROMPT)
        parsed = json.loads(raw)
        return ReflexDecision(
            reflex=str(parsed.get("reflex", "block")),
            micro_state=str(parsed.get("micro_state", "model_unspecified")),
            reason=str(parsed.get("reason", "phi3_reflex")),
        )
    except Exception:
        pass

    rsi = float(features.get("rsi", 50.0))
    momentum = float(features.get("momentum", 0.0))
    price = float(features.get("price", 0.0))
    atr = float(features.get("atr", 0.0))
    volatility = float(features.get("volatility", features.get("volatility", 0.0)))
    volume = float(features.get("volume", 0.0))
    history_points = int(features.get("history_points", 0) or 0)
    indicators_ready = bool(features.get("indicators_ready", False))
    bb_upper = float(features.get("bb_upper", 0.0))
    bb_lower = float(features.get("bb_lower", 0.0))
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
    if volume <= 0.0:
        return ReflexDecision(
            reflex="block",
            micro_state="data_integrity_issue",
            reason="invalid_or_zero_volume",
        )
    if not indicators_ready:
        return ReflexDecision(
            reflex="allow",
            micro_state="feature_warmup",
            reason=f"insufficient_indicator_history:{history_points}",
        )
    if atr > 0.0 and volatility > 0.0 and atr / volatility > 2.5:
        return ReflexDecision(
            reflex="delay",
            micro_state="volatility_shock",
            reason="atr_volatility_spike",
        )

    if rsi >= 85.0 and price >= bb_upper > 0.0:
        return ReflexDecision(
            reflex="reduce_confidence",
            micro_state="upside_overextension",
            reason="near_term_overextension_up",
        )
    if rsi <= 15.0 and 0.0 < price <= bb_lower:
        return ReflexDecision(
            reflex="reduce_confidence",
            micro_state="downside_overextension",
            reason="near_term_overextension_down",
        )
    if abs(momentum) < 0.0005:
        return ReflexDecision(
            reflex="delay",
            micro_state="no_impulse",
            reason="no_short_term_impulse",
        )
    return ReflexDecision(
        reflex="allow",
        micro_state="stable",
        reason="no_micro_danger_detected",
    )
