from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.config.runtime import get_runtime_setting, is_meme_lane


@dataclass
class TradeCostAssessment:
    actionable: bool
    expected_edge_pct: float
    spread_pct: float
    fee_round_trip_pct: float
    slippage_pct: float
    total_cost_pct: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_trade_cost(features: dict[str, Any], signal: str) -> TradeCostAssessment:
    lane = str(features.get("lane", "L3"))
    spread_pct = max(float(features.get("spread_pct", 0.0) or 0.0), 0.0)
    maker_fee_pct = float(get_runtime_setting("EXEC_MAKER_FEE_PCT")) / 100.0
    slippage_key = "MEME_EXEC_SLIPPAGE_ATR_MULT" if is_meme_lane(lane) else "EXEC_SLIPPAGE_ATR_MULT"
    max_spread_key = "MEME_EXEC_MAX_SPREAD_PCT" if is_meme_lane(lane) else "EXEC_MAX_SPREAD_PCT"
    slippage_atr_mult = float(get_runtime_setting(slippage_key))
    max_spread_pct = float(get_runtime_setting(max_spread_key))

    price = float(features.get("price", 0.0) or 0.0)
    atr = float(features.get("atr", 0.0) or 0.0)
    atr_norm = (atr / price) if price > 0.0 else 0.0
    slippage_pct = max(atr_norm * slippage_atr_mult, 0.0002)
    fee_round_trip_pct = maker_fee_pct * 2.0
    total_cost_pct = spread_pct + fee_round_trip_pct + slippage_pct

    momentum_5 = float(features.get("momentum_5", 0.0) or 0.0)
    momentum_14 = float(features.get("momentum_14", 0.0) or 0.0)
    rotation_score = float(features.get("rotation_score", 0.0) or 0.0)
    entry_score = float(features.get("entry_score", 0.0) or 0.0)

    if signal == "LONG":
        directional_momentum = max(momentum_5, 0.0) * 0.6 + max(momentum_14, 0.0) * 0.4
    elif signal == "SHORT":
        directional_momentum = max(-momentum_5, 0.0) * 0.6 + max(-momentum_14, 0.0) * 0.4
    else:
        directional_momentum = 0.0

    expected_edge_pct = directional_momentum + max(rotation_score, 0.0) * 0.10 + max(entry_score, 0.0) * 0.15

    reasons: list[str] = []
    actionable = True
    if spread_pct > max_spread_pct:
        actionable = False
        reasons.append(f"spread_high({spread_pct:.2f}>{max_spread_pct:.2f})")
    if expected_edge_pct <= total_cost_pct:
        actionable = False
        reasons.append("net_edge_below_cost")

    return TradeCostAssessment(
        actionable=actionable,
        expected_edge_pct=expected_edge_pct,
        spread_pct=spread_pct,
        fee_round_trip_pct=fee_round_trip_pct,
        slippage_pct=slippage_pct,
        total_cost_pct=total_cost_pct,
        reasons=reasons,
    )
