from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
import zoneinfo
from typing import Any

_LOGGER = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    phase: str
    mode: str
    vol_gate: float | str
    atr_armor: float
    logic: str
    sprint_week: int
    sprint_phase: str
    kill_switch: bool = False
    tp_bypass: bool = False
    pause_trader: bool = False

# AKDT is UTC-8
AKDT = zoneinfo.ZoneInfo("America/Anchorage")

def get_current_session() -> SessionConfig:
    """Determine the current trading phase based on the Elite Master Blueprint (AKDT)."""
    now_dt = datetime.now(AKDT)
    now = now_dt.time()
    day_of_month = now_dt.day
    
    # 21-Day Sprint Logic (Fractal Monthly Cycle)
    sprint_day = ((day_of_month - 1) % 21) + 1
    sprint_week = ((sprint_day - 1) // 7) + 1
    sprint_phases = {1: "Accumulation", 2: "Ignition", 3: "Distribution"}
    sprint_phase = sprint_phases.get(sprint_week, "Neutral")

    # Default: Handoff / Neutralize
    phase = "Handoff"
    mode = "NEUTRALIZE"
    vol_gate = 1.0
    atr_armor = 1.8
    logic = "London Kill Switch"
    kill_switch = True
    tp_bypass = False
    pause_trader = False

    # ASIA OPEN: 16:00 - 00:00 (Defense / Accumulation)
    if now >= time(16, 0) or now < time(0, 0):
        phase = "Asia Open"
        mode = "DEFENSIVE"
        vol_gate = 1.62
        atr_armor = 1.42
        logic = "Floor Scan / Accumulation"
        kill_switch = False

    # LONDON OPEN: 00:00 - 04:00 (Neutral / Ignition)
    elif now >= time(0, 0) and now < time(4, 0):
        phase = "London Open"
        mode = "NEUTRAL"
        vol_gate = 2.24
        atr_armor = 1.42
        logic = "Trend Discovery / Ignition"
        kill_switch = False

    # LONDON EXIT / AUDIT: 04:00 - 04:30 (Calibration)
    elif now >= time(4, 0) and now < time(4, 30):
        phase = "London Exit"
        mode = "CALIBRATION"
        vol_gate = "Dynamic"
        atr_armor = 1.65
        logic = "PAUSE TRADER / Analyze London End"
        kill_switch = False
        pause_trader = True

    # PRE-NYC CALIBRATION: 04:30 - 05:30 (Calibration)
    elif now >= time(4, 30) and now < time(5, 30):
        phase = "Pre-NYC"
        mode = "CALIBRATION"
        vol_gate = "Dynamic"
        atr_armor = 1.65
        logic = "TOOL CALL / Update Overrides"
        kill_switch = False
        pause_trader = True

    # NYC FLUSH: 05:30 - 08:30 (Aggressive / Strike)
    elif now >= time(5, 30) and now < time(8, 30):
        phase = "NYC Flush"
        mode = "AGGRESSIVE"
        vol_gate = 3.00
        atr_armor = 1.42
        logic = "Momentum Strike / Scraper Ignition"
        kill_switch = False
        
        # NYC Power Close Check (08:00 - 08:30)
        if now >= time(8, 0):
            tp_bypass = True 
            logic = "NYC Power Close (TP Bypass)"

    # KILL SWITCH: 08:30 (Hard Reset)
    elif now >= time(8, 30) and now < time(9, 0):
        phase = "Kill Switch"
        mode = "NEUTRALIZE"
        vol_gate = 1.0
        atr_armor = 1.8
        logic = "LIQUIDATE L4 (Scrapers) / Lock Paycheck"
        kill_switch = True # This flag is used by the orchestrator to liquidate L4

    # NYC CLOSE / SETTLEMENT: 11:30 - 13:00 (Defense)
    elif now >= time(11, 30) and now < time(13, 0):
        phase = "NYC Close"
        mode = "DEFENSIVE"
        vol_gate = 1.62
        atr_armor = 1.65
        logic = "Audit L1-L3 / Fractal Decay"
        kill_switch = False

    return SessionConfig(
        phase=phase, 
        mode=mode, 
        vol_gate=vol_gate, 
        atr_armor=atr_armor, 
        logic=logic, 
        sprint_week=sprint_week, 
        sprint_phase=sprint_phase, 
        kill_switch=kill_switch, 
        tp_bypass=tp_bypass,
        pause_trader=pause_trader
    )

def apply_session_to_features(features: dict[str, Any], session: SessionConfig) -> dict[str, Any]:
    features["session_phase"] = session.phase
    features["session_mode"] = session.mode
    features["session_vol_gate"] = session.vol_gate
    features["session_atr_armor"] = session.atr_armor
    features["session_kill_switch"] = session.kill_switch
    features["session_tp_bypass"] = session.tp_bypass
    features["session_sprint_week"] = session.sprint_week
    features["session_sprint_phase"] = session.sprint_phase
    features["session_pause_trader"] = session.pause_trader
    return features
