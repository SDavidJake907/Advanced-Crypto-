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
    expected_move_pct: float = 0.0
    required_edge_pct: float = 0.0
    net_edge_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expects_aggressive_entry(features: dict[str, Any], signal: str, lane: str, spread_pct: float) -> bool:
    if signal != "LONG" or not bool(get_runtime_setting("TRADE_COST_ASSUME_AGGRESSIVE_ENTRY_TAKER")):
        return False
    if not bool(get_runtime_setting("ENTRY_AGGRESSIVE_TAKER_ENABLED")):
        return False
    max_spread_pct = float(get_runtime_setting("ENTRY_AGGRESSIVE_TAKER_MAX_SPREAD_PCT"))
    if spread_pct <= 0.0 or spread_pct > max_spread_pct:
        return False
    promotion_tier = str(features.get("promotion_tier", "skip") or "skip").lower()
    entry_recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
    return (
        lane == "L4"
        or (lane == "L1" and promotion_tier == "promote")
        or (promotion_tier == "promote" and entry_recommendation in {"BUY", "STRONG_BUY"})
    )


def evaluate_trade_cost(features: dict[str, Any], signal: str) -> TradeCostAssessment:
    lane = str(features.get("lane", "L3"))
    spread_pct = max(float(features.get("spread_pct", 0.0) or 0.0), 0.0)
    maker_fee_pct = float(get_runtime_setting("EXEC_MAKER_FEE_PCT"))
    taker_fee_pct = float(get_runtime_setting("EXEC_TAKER_FEE_PCT"))
    slippage_key = "MEME_EXEC_SLIPPAGE_ATR_MULT" if is_meme_lane(lane) else "EXEC_SLIPPAGE_ATR_MULT"
    max_spread_key = "MEME_EXEC_MAX_SPREAD_PCT" if is_meme_lane(lane) else "EXEC_MAX_SPREAD_PCT"
    slippage_atr_mult = float(get_runtime_setting(slippage_key))
    max_spread_pct = float(get_runtime_setting(max_spread_key))
    safety_buffer_pct = float(get_runtime_setting("TRADE_COST_SAFETY_BUFFER_PCT"))

    price = float(features.get("price", 0.0) or 0.0)
    atr = float(features.get("atr", 0.0) or 0.0)
    atr_norm = (atr / price) if price > 0.0 else 0.0
    slippage_pct = max(atr_norm * slippage_atr_mult * 100.0, 0.02)
    aggressive_entry = _expects_aggressive_entry(features, signal, lane, spread_pct)
    entry_fee_pct = taker_fee_pct if aggressive_entry else maker_fee_pct
    # Use a conservative blended exit assumption because many real exits are not patient maker fills.
    exit_fee_pct = max(maker_fee_pct, taker_fee_pct * 0.75)
    fee_round_trip_pct = entry_fee_pct + exit_fee_pct
    total_cost_pct = spread_pct + fee_round_trip_pct + slippage_pct + safety_buffer_pct
    structure_quality = float(features.get("structure_quality", 50.0) or 50.0)
    continuation_quality = float(features.get("continuation_quality", 50.0) or 50.0)
    momentum_quality = float(features.get("momentum_quality", 50.0) or 50.0)
    trade_quality = float(features.get("trade_quality", 50.0) or 50.0)

    if lane == "L1":
        tp_mult = float(get_runtime_setting("L1_EXIT_ATR_TAKE_PROFIT_MULT"))
    elif lane == "L2":
        tp_mult = float(get_runtime_setting("L2_EXIT_ATR_TAKE_PROFIT_MULT"))
    elif is_meme_lane(lane):
        tp_mult = float(get_runtime_setting("MEME_EXIT_ATR_TAKE_PROFIT_MULT"))
    else:
        tp_mult = float(get_runtime_setting("EXIT_ATR_TAKE_PROFIT_MULT"))

    expected_move_pct = atr_norm * tp_mult * 100.0
    quality_scale = max(min((0.35 * structure_quality + 0.35 * continuation_quality + 0.15 * momentum_quality + 0.15 * trade_quality) / 100.0, 1.2), 0.35)
    expected_edge_pct = expected_move_pct * quality_scale
    min_edge_mult = float(get_runtime_setting("TRADE_COST_MIN_EDGE_MULT"))
    min_expected_edge_pct = float(get_runtime_setting("TRADE_COST_MIN_EXPECTED_EDGE_PCT"))
    required_edge_pct = max(total_cost_pct * min_edge_mult, min_expected_edge_pct)
    net_edge_pct = expected_edge_pct - total_cost_pct

    reasons: list[str] = []
    actionable = True
    if spread_pct > max_spread_pct:
        actionable = False
        reasons.append(f"spread_high({spread_pct:.2f}>{max_spread_pct:.2f})")
    if signal in {"LONG", "SHORT"} and expected_edge_pct <= required_edge_pct:
        actionable = False
        reasons.append(f"edge_below_cost_floor({expected_edge_pct:.2f}<={required_edge_pct:.2f})")

    return TradeCostAssessment(
        actionable=actionable,
        expected_edge_pct=expected_edge_pct,
        spread_pct=spread_pct,
        fee_round_trip_pct=fee_round_trip_pct,
        slippage_pct=slippage_pct,
        total_cost_pct=total_cost_pct,
        reasons=reasons,
        expected_move_pct=expected_move_pct,
        required_edge_pct=required_edge_pct,
        net_edge_pct=net_edge_pct,
    )
