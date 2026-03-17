from __future__ import annotations

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
    clamp_up: bool = True,
    kelly_fraction: float = 1.0,
) -> ClampResult:
    # Apply Kelly criterion sizing before other limits
    kelly_fraction = max(0.1, min(1.0, float(kelly_fraction)))
    desired_usd *= kelly_fraction
    # Hard caps
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

    if desired_usd < min_required:
        if not clamp_up:
            return ClampResult(False, "too_small", price_rounded, 0.0, 0.0)
        desired_usd = min_required
        size_base = desired_usd / price_rounded

    # Enforce caps
    if desired_usd > max_size:
        return ClampResult(False, "exceeds_max_position", price_rounded, 0.0, 0.0)
    if desired_usd > equity_usd:
        return ClampResult(False, "exceeds_equity", price_rounded, 0.0, 0.0)
    if max_risk_usd <= 0.0:
        return ClampResult(False, "risk_budget_zero", price_rounded, 0.0, 0.0)
    if desired_usd > max_risk_usd:
        return ClampResult(False, "exceeds_risk_guard", price_rounded, 0.0, 0.0)

    # round base size to lot_decimals
    size_base = round(size_base, rule.lot_decimals)
    notional = size_base * price_rounded
    if size_base <= 0.0:
        return ClampResult(False, "rounded_qty_zero", price_rounded, 0.0, 0.0)
    if size_base < rule.ordermin:
        return ClampResult(False, "ordermin_not_met", price_rounded, 0.0, 0.0)

    return ClampResult(True, "ok", price_rounded, size_base, notional)
