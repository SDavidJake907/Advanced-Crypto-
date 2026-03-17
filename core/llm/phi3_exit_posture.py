from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from core.config.runtime import get_runtime_setting
from core.llm.client import phi3_chat
from core.llm.prompts import PHI3_EXIT_POSTURE_SYSTEM_PROMPT


@dataclass
class ExitPostureDecision:
    posture: str
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_confidence(value: Any, default: float = 0.55) -> float:
    try:
        conf = float(value)
    except Exception:
        return default
    return max(0.0, min(conf, 1.0))


def _heuristic_exit_posture(payload: dict[str, Any]) -> ExitPostureDecision:
    pnl_pct = float(payload.get("pnl_pct", 0.0))
    hold_minutes = float(payload.get("hold_minutes", 0.0))
    momentum = float(payload.get("momentum", payload.get("momentum_5", 0.0)) or 0.0)
    trend_1h = float(payload.get("trend_1h", 0.0) or 0.0)
    rsi = float(payload.get("rsi", 50.0) or 50.0)
    entry_price = float(payload.get("entry_price", 0.0) or 0.0)
    price = float(payload.get("price", 0.0) or 0.0)

    stale_min_hold = float(get_runtime_setting("EXIT_STALE_MIN_HOLD_MIN"))
    stale_abs_pnl = float(get_runtime_setting("EXIT_STALE_MAX_ABS_PNL_PCT"))
    tighten_min_pnl = float(get_runtime_setting("EXIT_TIGHTEN_MIN_PNL_PCT"))
    neg_pnl_exit = float(get_runtime_setting("EXIT_POSTURE_NEG_PNL_PCT"))

    if price <= 0.0 or entry_price <= 0.0:
        return ExitPostureDecision("RUN", "posture_data_incomplete", 0.40)

    if hold_minutes >= stale_min_hold and abs(pnl_pct) <= stale_abs_pnl and abs(momentum) < 0.0015:
        return ExitPostureDecision("STALE", "time_stop_no_progress", 0.82)

    if pnl_pct <= neg_pnl_exit and momentum < 0.0 and trend_1h <= 0.0:
        return ExitPostureDecision("EXIT", "loser_with_trend_decay", 0.84)

    if pnl_pct >= tighten_min_pnl and (momentum < 0.0 or rsi >= 72.0):
        return ExitPostureDecision("TIGHTEN", "protect_open_profit", 0.74)

    if pnl_pct > 0.0 and trend_1h > 0.0 and momentum >= 0.0:
        return ExitPostureDecision("RUN", "trend_intact_let_run", 0.72)

    return ExitPostureDecision("RUN", "default_hold_state", 0.58)


def phi3_exit_posture(payload: dict[str, Any]) -> ExitPostureDecision:
    if bool(get_runtime_setting("EXIT_POSTURE_USE_PHI3")):
        try:
            raw = phi3_chat(payload, system=PHI3_EXIT_POSTURE_SYSTEM_PROMPT, max_tokens=180)
            parsed = json.loads(raw)
            posture = str(parsed.get("posture", "RUN")).strip().upper()
            if posture not in {"RUN", "TIGHTEN", "EXIT", "STALE"}:
                posture = "RUN"
            reason = str(parsed.get("reason", "phi3_exit_posture")).strip() or "phi3_exit_posture"
            confidence = _clamp_confidence(parsed.get("confidence"), 0.60)
            return ExitPostureDecision(posture=posture, reason=reason, confidence=confidence)
        except Exception:
            pass
    return _heuristic_exit_posture(payload)
