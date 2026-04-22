from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass
class RegimeState:
    state: str = "unknown"
    candidate: str = "unknown"
    candidate_count: int = 0
    dwell_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MACHINES: dict[str, RegimeState] = {}
_MACHINES_LOCK = Lock()
_CONFIRM_REQUIRED = 2
_DWELL_MIN = {"unknown": 0, "bullish": 2, "sideways": 2, "bearish": 2, "volatile": 0, "blowoff": 1}


def _raw_regime(features: dict[str, Any]) -> str:
    trend_1h = int(features.get("trend_1h", 0) or 0)
    regime_7d = str(features.get("regime_7d", "unknown") or "unknown").lower()
    macro_30d = str(features.get("macro_30d", "sideways") or "sideways").lower()
    market_regime = features.get("market_regime", {}) or {}
    breadth = float(market_regime.get("breadth", 0.0) or 0.0)
    rv_mkt = float(market_regime.get("rv_mkt", 0.0) or 0.0)
    atr = float(features.get("atr", 0.0) or 0.0)
    price = float(features.get("price", 0.0) or 0.0)
    atr_pct = (atr / price) if price > 0.0 else 0.0

    # Blow-off Detection (Macro Dynamic Mandate)
    # 1h RSI > 85, Dist from EMA26 > 15%, Vertical structure
    rsi_1h = float(features.get("rsi_1h", 0.0) or 0.0)
    ema26_dist = float(features.get("dist_ema26_pct", 0.0) or 0.0)
    blowoff_tag = bool(features.get("blowoff_structure", False))
    
    if rsi_1h >= 85.0 or ema26_dist >= 15.0 or blowoff_tag:
        return "blowoff"

    if atr_pct >= 0.03 or rv_mkt >= 0.02:
        return "volatile"
    if regime_7d == "trending" and macro_30d == "bull" and trend_1h > 0 and breadth >= 0.45:
        return "bullish"
    if macro_30d == "bear" and trend_1h < 0:
        return "bearish"
    if regime_7d == "choppy" or breadth <= 0.40:
        return "sideways"
    return "bullish" if trend_1h > 0 else "sideways"


def get_blowoff_adjustments() -> dict[str, Any]:
    """Return specific settings to apply automatically during/after a blow-off."""
    return {
        "AGGRESSION_MODE": "DEFENSIVE",
        "NEMOTRON_GATE_MIN_VOLUME_RATIO": 2.5,
        "NEMOTRON_GATE_MIN_NET_EDGE_PCT": 0.5,
        "EXIT_ATR_STOP_MULT": 3.0, # Wider initial stop for volatility
        "EXIT_TRAIL_ATR_MULT": 0.5, # Very tight trail to lock in the blow-off top
        "L1_EXIT_TIGHTEN_MIN_PNL_PCT": 2.0,
    }


def update_regime_state(symbol: str, features: dict[str, Any]) -> dict[str, Any]:
    key = str(symbol).strip().upper()
    raw = _raw_regime(features)
    with _MACHINES_LOCK:
        machine = _MACHINES.get(key) or RegimeState()

        if raw == "volatile":
            machine.state = "volatile"
            machine.candidate = "volatile"
            machine.candidate_count = 0
            machine.dwell_count = 0
            _MACHINES[key] = machine
            return {"regime_raw": raw, "regime_state": machine.state, "regime_confirm_count": 0, "regime_dwell": 0}

        if raw == machine.state:
            machine.candidate = raw
            machine.candidate_count = 0
            machine.dwell_count += 1
        else:
            if raw != machine.candidate:
                machine.candidate = raw
                machine.candidate_count = 1
            else:
                machine.candidate_count += 1

            dwell_min = _DWELL_MIN.get(machine.state, 2)
            if machine.candidate_count >= _CONFIRM_REQUIRED and machine.dwell_count >= dwell_min:
                machine.state = machine.candidate
                machine.candidate_count = 0
                machine.dwell_count = 1
            else:
                machine.dwell_count += 1

        _MACHINES[key] = machine
        return {
            "regime_raw": raw,
            "regime_state": machine.state,
            "regime_confirm_count": machine.candidate_count,
            "regime_dwell": machine.dwell_count,
        }
