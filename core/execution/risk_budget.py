from __future__ import annotations

from core.config.runtime import get_runtime_setting, is_meme_lane


def estimate_entry_risk_fraction(*, price: float, atr: float, lane: str | None) -> float:
    resolved_lane = str(lane or "L3").upper()
    if resolved_lane == "L1":
        stop_mult = float(get_runtime_setting("L1_EXIT_ATR_STOP_MULT"))
        min_stop_pct = float(get_runtime_setting("L1_EXIT_MIN_STOP_PCT")) / 100.0
    elif resolved_lane == "L2":
        stop_mult = float(get_runtime_setting("L2_EXIT_ATR_STOP_MULT"))
        min_stop_pct = float(get_runtime_setting("L2_EXIT_MIN_STOP_PCT")) / 100.0
    elif resolved_lane == "L4" or is_meme_lane(resolved_lane):
        stop_mult = float(get_runtime_setting("MEME_EXIT_ATR_STOP_MULT"))
        min_stop_pct = float(get_runtime_setting("MEME_EXIT_MIN_STOP_PCT")) / 100.0
    else:
        stop_mult = float(get_runtime_setting("EXIT_ATR_STOP_MULT"))
        min_stop_pct = float(get_runtime_setting("EXIT_MIN_STOP_PCT")) / 100.0

    atr_risk_fraction = 0.0
    if price > 0.0 and atr > 0.0:
        atr_risk_fraction = (atr * stop_mult) / price
    return max(1e-6, min(1.0, max(min_stop_pct, atr_risk_fraction)))
