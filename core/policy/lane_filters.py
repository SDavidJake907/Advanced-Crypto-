from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.config.runtime import get_runtime_setting


@dataclass
class LaneFilterResult:
    lane: str
    passed: bool
    reason: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_lane_filters(features: dict[str, Any]) -> LaneFilterResult:
    lane = str(features.get("lane", "L3") or "L3").upper()
    volume_ratio = float(features.get("volume_ratio", 1.0) or 1.0)
    rsi = float(features.get("rsi", 50.0) or 50.0)
    trend_1h = int(features.get("trend_1h", 0) or 0)
    bb_bandwidth = float(features.get("bb_bandwidth", 0.0) or 0.0)
    atr = float(features.get("atr", 0.0) or 0.0)
    price = float(features.get("price", 0.0) or 0.0)
    atr_pct = (atr / price) if price > 0.0 else 0.0
    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    spread_pct = float(features.get("spread_pct", 0.0) or 0.0) / 100.0
    price_zscore = float(features.get("price_zscore", 0.0) or 0.0)
    volume_surge = float(features.get("volume_surge", 0.0) or 0.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    leader_urgency = float(features.get("leader_urgency", 0.0) or 0.0)
    leader_takeover = bool(features.get("leader_takeover", False))
    symbol_trending = bool(features.get("sentiment_symbol_trending", False))

    if spread_pct > (0.05 if lane == "L4" else 0.015):
        return LaneFilterResult(lane=lane, passed=False, reason="spread_hard", severity="hard")

    if lane == "L1":
        if volume_ratio < 0.20:
            # vol_low: only hard-veto if structure is also weak
            real_participation = rotation_score >= 0.05 or momentum_5 >= 0.002 or volume_surge >= 0.10
            if not real_participation:
                return LaneFilterResult(lane=lane, passed=False, reason="lane1_vol_low", severity="soft")
            return LaneFilterResult(lane=lane, passed=True, reason="lane1_vol_low_overridden", severity="soft")
        # RSI and Trend removed for L1 leaders - let Nemo/Phi decide
        return LaneFilterResult(lane=lane, passed=True, reason="lane1_ok", severity="ok")

    if lane == "L2":
        strong_mover_signal = (
            rotation_score >= 0.08
            or momentum_5 >= 0.005
            or volume_surge >= 0.15
            or leader_takeover
            or leader_urgency >= 5.0
        )

        if strong_mover_signal:
            return LaneFilterResult(lane=lane, passed=True, reason="lane2_leader_signal", severity="ok")

        if trend_1h < -1 and momentum_14 <= -0.005:
            return LaneFilterResult(lane=lane, passed=False, reason="lane2_trend_negative", severity="soft")
        
        if volume_ratio < 0.10 and volume_surge < 0.05:
            real_participation = rotation_score >= 0.08 or momentum_5 >= 0.005 or volume_surge >= 0.08
            if not real_participation:
                return LaneFilterResult(lane=lane, passed=False, reason="lane2_vol_low", severity="soft")
            return LaneFilterResult(lane=lane, passed=True, reason="lane2_vol_low_overridden", severity="soft")
        
        if rsi > 85.0:
            return LaneFilterResult(lane=lane, passed=False, reason="lane2_rsi_hot", severity="soft")
        return LaneFilterResult(lane=lane, passed=True, reason="lane2_ok", severity="ok")

    if lane == "L3":
        if rsi < 38.0:
            return LaneFilterResult(lane=lane, passed=False, reason="lane3_rsi_low", severity="soft")
        if rsi > 72.0:
            return LaneFilterResult(lane=lane, passed=False, reason="lane3_rsi_hot", severity="soft")
        if trend_1h < 0:
            return LaneFilterResult(lane=lane, passed=False, reason="lane3_trend_negative", severity="soft")
        if momentum_14 < -0.005:
            return LaneFilterResult(lane=lane, passed=False, reason="lane3_momo_low", severity="soft")
        if volume_ratio < 0.25:
            real_participation = rotation_score >= 0.06 or momentum_5 >= 0.003 or volume_surge >= 0.08
            if not real_participation:
                return LaneFilterResult(lane=lane, passed=False, reason="lane3_vol_low", severity="soft")
            return LaneFilterResult(lane=lane, passed=True, reason="lane3_vol_low_overridden", severity="soft")
        if volume_ratio < 0.5:
            return LaneFilterResult(lane=lane, passed=True, reason="lane3_vol_low_warning", severity="soft")
        if momentum_5 > 0.012 or volume_surge >= 0.6:
            return LaneFilterResult(lane=lane, passed=False, reason="lane3_heat_high", severity="soft")
        if atr_pct > 0.06:
            return LaneFilterResult(lane=lane, passed=False, reason="lane3_atr_high", severity="soft")
        return LaneFilterResult(lane=lane, passed=True, reason="lane3_ok", severity="ok")

    if lane == "L4":
        meme_min_volume_ratio = float(get_runtime_setting("MEME_LANE_MIN_VOLUME_RATIO"))
        meme_min_volume_surge = float(get_runtime_setting("MEME_LANE_MIN_VOLUME_SURGE"))
        meme_soft_max_spread = float(get_runtime_setting("MEME_LANE_SOFT_MAX_SPREAD_PCT")) / 100.0
        if (
            volume_ratio < meme_min_volume_ratio
            and volume_surge < meme_min_volume_surge
            and momentum_5 <= -0.002
            and not symbol_trending
        ):
            return LaneFilterResult(lane=lane, passed=False, reason="lane4_heat_low", severity="soft")
        if spread_pct > meme_soft_max_spread:
            return LaneFilterResult(lane=lane, passed=False, reason="lane4_spread_wide", severity="soft")
        return LaneFilterResult(lane=lane, passed=True, reason="lane4_ok", severity="ok")

    return LaneFilterResult(lane=lane, passed=True, reason="lane_unknown_passthrough", severity="ok")
