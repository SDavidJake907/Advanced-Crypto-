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


def calculate_dynamic_risk_pct(
    base_risk_pct: float,
    cash: float,
    atr_pct: float,
    trade_quality_scale: float = 1.0,
) -> float:
    # Phase 4 dynamic sizing: compute risk sizing based on max capital loss and absolute ATR stop distances.
    # The 'base_risk_pct' is intentionally ignored in favor of structural volatility equations.
    max_capital_loss_pct = 2.0 
    
    # Estimate typical stop distance in percentage using 1.8 ATR global standard
    stop_distance_pct = atr_pct * 1.8 * 100.0
    
    if stop_distance_pct > 0.0:
        position_size_pct = max_capital_loss_pct / stop_distance_pct * 100.0
    else:
        position_size_pct = 10.0
        
    # Scale down maximum position exposure as account equity grows
    if cash >= 2000.0:
        position_size_pct = min(position_size_pct, 5.0)
    elif cash >= 500.0:
        position_size_pct = min(position_size_pct, 10.0)
    elif cash >= 200.0:
        position_size_pct = min(position_size_pct, 15.0)
    else:
        position_size_pct = min(position_size_pct, 25.0)
        
    # Structural volatility penalty: extreme relative swings reduce maximum exposure
    if atr_pct > 0.05:
        position_size_pct *= 0.50
    elif atr_pct > 0.03:
        position_size_pct *= 0.75
        
    # Trade quality: thin/poor execution orderbooks downscale execution sizes
    position_size_pct *= trade_quality_scale
    
    return max(1.0, min(50.0, position_size_pct))
