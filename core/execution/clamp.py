from __future__ import annotations

import math
from dataclasses import dataclass

from core.execution.kraken_rules import KrakenPairRule


@dataclass
class ClampResult:
    ok: bool
    reason: str
    price: float
    size_base: float
    notional_usd: float


def clamp_order_size(
    *,
    pair: str,
    price: float,
    desired_usd: float,
    rule: KrakenPairRule,
    equity_usd: float,
    max_position_usd: float,
    risk_per_trade_pct: float,
    min_notional_usd: float,
    max_min_trade_risk_mult: float = 999.0,
    clamp_up: bool = True,
    kelly_fraction: float = 1.0,
    risk_fraction_of_notional: float = 1.0,
) -> ClampResult:
    # Apply Kelly criterion sizing before other limits
    kelly_fraction = max(0.1, min(1.0, float(kelly_fraction)))
    desired_usd *= kelly_fraction
    # Hard caps
    risk_fraction_of_notional = max(1e-6, min(1.0, float(risk_fraction_of_notional)))
    max_risk_usd = equity_usd * (risk_per_trade_pct / 100.0)
    max_size = min(max_position_usd, equity_usd)

    if desired_usd <= 0 or price <= 0:
        return ClampResult(False, "invalid_size_or_price", 0.0, 0.0, 0.0)

    # round price
    price_rounded = round(price, rule.pair_decimals)
    if price_rounded <= 0:
        return ClampResult(False, "price_rounding_invalid", 0.0, 0.0, 0.0)

    # desired base size
    size_base = desired_usd / price_rounded
    ordermin_notional = rule.ordermin * price_rounded
    min_required = max(min_notional_usd, ordermin_notional, rule.costmin or 0.0)

    clamped_to_min_trade = False
    if desired_usd < min_required:
        if not clamp_up:
            return ClampResult(False, "too_small", price_rounded, 0.0, 0.0)
        min_trade_risk_usd = min_required * risk_fraction_of_notional
        if max_risk_usd > 0.0 and min_trade_risk_usd > (max_risk_usd * max(float(max_min_trade_risk_mult), 0.0)):
            return ClampResult(False, "min_trade_exceeds_risk_budget", price_rounded, 0.0, 0.0)
        desired_usd = min_required
        size_base = desired_usd / price_rounded
        clamped_to_min_trade = True

    # Enforce caps
    if desired_usd > max_size:
        return ClampResult(False, "exceeds_max_position", price_rounded, 0.0, 0.0)
    if desired_usd > equity_usd:
        return ClampResult(False, "exceeds_equity", price_rounded, 0.0, 0.0)
    if max_risk_usd <= 0.0:
        return ClampResult(False, "risk_budget_zero", price_rounded, 0.0, 0.0)
    estimated_trade_risk_usd = desired_usd * risk_fraction_of_notional
    if estimated_trade_risk_usd > max_risk_usd and not clamped_to_min_trade:
        return ClampResult(False, "exceeds_risk_guard", price_rounded, 0.0, 0.0)

    # floor base size to lot_decimals — always round down so we never order more than we hold
    factor = 10 ** rule.lot_decimals
    size_base = math.floor(size_base * factor) / factor
    notional = size_base * price_rounded
    if size_base <= 0.0:
        return ClampResult(False, "rounded_qty_zero", price_rounded, 0.0, 0.0)
    if size_base < rule.ordermin:
        return ClampResult(False, "ordermin_not_met", price_rounded, 0.0, 0.0)

    return ClampResult(True, "ok", price_rounded, size_base, notional)
