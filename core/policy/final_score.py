"""Unified final trade score — pre-computed before any AI model sees a candidate.

The goal is simple: give Nemo one number it can trust, with a full breakdown,
so it ranks candidates instead of recomputing arithmetic.

Formula
-------
final_score = entry_score
            + reflex_bonus        (+5 allow / -5 block / 0 neutral)
            + reliability_bonus   (symbol win-rate, -10 to +10)
            + basket_fit_bonus    (low correlation to current holds, 0 to +8)
            - spread_penalty      (spread_pct * 12, capped at 15)
            - cost_penalty        (from point_breakdown, already computed)
            - correlation_penalty (avg portfolio correlation, 0 to -8)

Output is clamped [0, 100].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FinalTradeScore:
    symbol: str
    final_score: float          # 0–100 composite
    entry_score: float          # raw setup score from entry_verifier
    reflex_bonus: float         # Phi-3 reflex signal adjustment
    divergence_bonus: float     # RSI divergence adjustment
    fear_greed_bonus: float     # macro sentiment context adjustment
    btc_dominance_bonus: float  # BTC dominance context adjustment
    reliability_bonus: float    # per-symbol win-rate adjustment
    basket_fit_bonus: float     # low correlation to held positions
    spread_penalty: float       # spread cost penalty
    cost_penalty: float         # from point_breakdown.cost_penalty_pts
    correlation_penalty: float  # avg portfolio correlation penalty
    net_edge_pct: float         # expected edge after all costs
    score_breakdown: dict[str, Any]
    breakdown_notes: list[str]  # human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Contribution:
    value: float
    note: str = ""


@dataclass(frozen=True)
class _CostPenaltyContribution:
    spread_penalty: float
    cost_penalty: float
    net_edge_pct: float
    spread_note: str = ""
    cost_note: str = ""


@dataclass(frozen=True)
class _SetupContribution:
    entry_score: float
    reflex_bonus: float
    divergence_bonus: float
    fear_greed_bonus: float
    btc_dominance_bonus: float
    notes: tuple[str, ...]

    @property
    def total(self) -> float:
        return self.entry_score + self.reflex_bonus + self.divergence_bonus + self.fear_greed_bonus + self.btc_dominance_bonus


@dataclass(frozen=True)
class _BasketContribution:
    basket_fit_bonus: float
    correlation_penalty: float
    notes: tuple[str, ...]

    @property
    def total(self) -> float:
        return self.basket_fit_bonus - self.correlation_penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean_abs_correlation(correlation_row: Any) -> float:
    if correlation_row is None:
        return 0.0
    row = np.asarray(correlation_row, dtype=np.float64).ravel()
    if row.size == 0:
        return 0.0
    filtered = [abs(float(v)) for v in row if np.isfinite(v) and abs(float(v)) < 0.999]
    return float(np.mean(filtered)) if filtered else 0.0


def _reflex_bonus(features: dict[str, Any]) -> tuple[float, str]:
    """
    Read the Phi-3 reflex result already stored in features (set by orchestrator).
    Returns (bonus, note).
    """
    reflex = features.get("reflex") or {}
    if isinstance(reflex, dict):
        signal = str(reflex.get("reflex", "") or "").lower()
    else:
        signal = str(reflex).lower()

    if signal == "allow":
        return 5.0, "reflex:allow(+5)"
    if signal == "block":
        return -5.0, "reflex:block(-5)"
    return 0.0, ""


def _reliability_bonus(symbol: str, reliability_map: dict[str, Any]) -> tuple[float, str]:
    """
    Look up per-symbol win-rate bonus from a pre-loaded reliability map.
    Map format: { "ETH/USD": {"win_rate": 0.62, "trade_count": 18}, ... }
    Returns (bonus, note).
    """
    if not reliability_map:
        return 0.0, ""
    rec = reliability_map.get(symbol) or reliability_map.get(symbol.upper())
    if not rec:
        return 0.0, ""
    win_rate = float(rec.get("win_rate", 0.0) or 0.0)
    count = int(rec.get("trade_count", 0) or 0)
    if count < 5:
        return 0.0, ""           # not enough history to have an opinion

    if win_rate >= 0.65:
        return 8.0, f"reliable(wr={win_rate:.0%},+8)"
    if win_rate >= 0.55:
        return 4.0, f"reliable(wr={win_rate:.0%},+4)"
    if win_rate >= 0.45:
        return 0.0, ""
    if win_rate >= 0.35:
        return -6.0, f"unreliable(wr={win_rate:.0%},-6)"
    return -10.0, f"poor_history(wr={win_rate:.0%},-10)"


def _divergence_bonus(features: dict[str, Any]) -> tuple[float, str]:
    lane = str(features.get("lane", "L3") or "L3").upper()
    bullish = bool(features.get("bullish_divergence", False))
    bearish = bool(features.get("bearish_divergence", False))
    strength = float(features.get("divergence_strength", features.get("rsi_divergence_strength", 0.0)) or 0.0)
    age = int(features.get("divergence_age_bars", features.get("rsi_divergence_age", 99)) or 99)
    if not bullish and not bearish:
        legacy = int(features.get("rsi_divergence", 0) or 0)
        bullish = legacy == 1
        bearish = legacy == -1
    if bullish == bearish:
        return 0.0, ""

    lane_mult = {"L1": 0.85, "L2": 1.0, "L3": 1.0, "L4": 0.65}.get(lane, 1.0)
    if age <= 2:
        freshness = 1.0
    elif age <= 6:
        freshness = 0.75
    elif age <= 10:
        freshness = 0.45
    elif age <= 15:
        freshness = 0.2
    else:
        freshness = 0.0

    adjustment = _clamp(_clamp(strength, 0.0, 1.0) * freshness * lane_mult * 6.0, 0.0, 6.0)
    if adjustment <= 0.0:
        return 0.0, ""
    if bullish:
        return adjustment, f"bull_div(age={age},+{adjustment:.1f})"
    return -adjustment, f"bear_div(age={age},{-adjustment:.1f})"


def _fear_greed_bonus(features: dict[str, Any]) -> tuple[float, str]:
    raw = features.get("sentiment_fng_value")
    try:
        fng = float(raw)
    except (TypeError, ValueError):
        return 0.0, ""

    fng = _clamp(fng, 0.0, 100.0)
    trend_confirmed = bool(features.get("trend_confirmed", False))
    ranging_market = bool(features.get("ranging_market", False))
    overextended = bool(features.get("overextended", False))

    if overextended and fng >= 70.0:
        penalty = _clamp(((fng - 70.0) / 30.0) * 1.5, 0.0, 1.5)
        return -penalty, f"fng_greed_extended({int(round(fng))},-{penalty:.1f})" if penalty >= 0.5 else ""

    if not trend_confirmed and fng <= 25.0:
        penalty = _clamp(((25.0 - fng) / 25.0) * 2.0, 0.0, 2.0)
        return -penalty, f"fng_extreme_fear({int(round(fng))},-{penalty:.1f})" if penalty >= 0.5 else ""

    if trend_confirmed and not ranging_market:
        bonus = _clamp((fng - 50.0) / 25.0, -1.0, 1.0) * 1.5
        if abs(bonus) < 0.25:
            return 0.0, ""
        sign = "+" if bonus > 0 else ""
        return bonus, f"fng_trend_context({int(round(fng))},{sign}{bonus:.1f})"

    return 0.0, ""


def _btc_dominance_bonus(features: dict[str, Any]) -> tuple[float, str]:
    raw = features.get("sentiment_btc_dominance")
    try:
        btc_dominance = float(raw)
    except (TypeError, ValueError):
        return 0.0, ""

    btc_dominance = _clamp(btc_dominance, 0.0, 100.0)
    symbol = str(features.get("symbol", "") or "").upper()
    trend_confirmed = bool(features.get("trend_confirmed", False))
    ranging_market = bool(features.get("ranging_market", False))
    lane = str(features.get("lane", "L3") or "L3").upper()
    is_btc = symbol == "BTC/USD"
    is_eth = symbol == "ETH/USD"
    is_major = is_btc or is_eth

    if btc_dominance < 55.0:
        return 0.0, ""

    pressure = _clamp((btc_dominance - 55.0) / 10.0, 0.0, 1.0)
    if is_btc:
        bonus = round(pressure * 1.2, 2)
        return bonus, f"btc_dom_btc_tailwind({btc_dominance:.1f},+{bonus:.1f})" if bonus >= 0.5 else ""
    if is_eth:
        bonus = round(pressure * 0.5, 2)
        return bonus, f"btc_dom_eth_support({btc_dominance:.1f},+{bonus:.1f})" if bonus >= 0.5 else ""

    # Alts are more fragile when BTC dominance is elevated, especially in range/mixed tape.
    penalty_mult = 1.0
    if ranging_market:
        penalty_mult += 0.25
    if not trend_confirmed:
        penalty_mult += 0.35
    if lane == "L4":
        penalty_mult += 0.25
    penalty = round(_clamp(pressure * penalty_mult * 1.2, 0.0, 2.0), 2)
    return -penalty, f"btc_dom_alt_caution({btc_dominance:.1f},-{penalty:.1f})" if penalty >= 0.5 else ""


def _basket_fit_bonus(features: dict[str, Any], held_correlation_map: dict[str, float]) -> tuple[float, str]:
    """
    Bonus for symbols that are uncorrelated with what's currently held.
    held_correlation_map: { "ETH/USD": 0.82, "BTC/USD": 0.45 } — correlation of this
    candidate with each currently held position.
    Returns (bonus, note).
    """
    if not held_correlation_map:
        return 6.0, "basket:uncorrelated(+6)"   # nothing held = free slot bonus

    max_corr = max(abs(v) for v in held_correlation_map.values())
    avg_corr = float(np.mean([abs(v) for v in held_correlation_map.values()]))

    if max_corr >= 0.85:
        return 0.0, f"basket:high_corr(max={max_corr:.2f},+0)"
    if max_corr >= 0.70:
        return 2.0, f"basket:moderate_corr(max={max_corr:.2f},+2)"
    if avg_corr < 0.40:
        return 8.0, f"basket:diversifying(avg={avg_corr:.2f},+8)"
    return 4.0, f"basket:low_corr(avg={avg_corr:.2f},+4)"


def _spread_penalty(features: dict[str, Any]) -> tuple[float, str]:
    spread_pct = float(features.get("spread_pct", 0.0) or 0.0)
    if spread_pct <= 0.0:
        return 0.0, ""
    penalty = _clamp(spread_pct * 12.0, 0.0, 15.0)
    return penalty, f"spread({spread_pct:.2f}%,-{penalty:.1f})" if penalty > 0.5 else ""


def _cost_penalty_from_breakdown(features: dict[str, Any]) -> tuple[float, str]:
    pb = features.get("point_breakdown") or {}
    if not isinstance(pb, dict):
        return 0.0, ""
    penalty = float(pb.get("cost_penalty_pts", 0.0) or 0.0)
    net_edge = float(pb.get("net_edge_pct", 0.0) or 0.0)
    note = f"cost_pen(-{penalty:.1f})" if penalty > 0.5 else ""
    return penalty, note, net_edge


def _correlation_penalty(features: dict[str, Any]) -> tuple[float, str]:
    """Penalise high average correlation with the broader portfolio."""
    avg_corr = _mean_abs_correlation(features.get("correlation_row"))
    if avg_corr <= 0.60:
        return 0.0, ""
    if avg_corr >= 0.90:
        penalty = 8.0
    elif avg_corr >= 0.80:
        penalty = 5.0
    else:
        penalty = 2.0
    return penalty, f"portfolio_corr({avg_corr:.2f},-{penalty:.0f})"


def _setup_contribution(features: dict[str, Any]) -> _SetupContribution:
    entry_score = float(features.get("entry_score", 0.0) or 0.0)
    reflex_bonus, reflex_note = _reflex_bonus(features)
    divergence_bonus, divergence_note = _divergence_bonus(features)
    fear_greed_bonus, fear_greed_note = _fear_greed_bonus(features)
    btc_dominance_bonus, btc_dominance_note = _btc_dominance_bonus(features)
    notes = tuple(note for note in (reflex_note, divergence_note, fear_greed_note, btc_dominance_note) if note)
    return _SetupContribution(
        entry_score=entry_score,
        reflex_bonus=reflex_bonus,
        divergence_bonus=divergence_bonus,
        fear_greed_bonus=fear_greed_bonus,
        btc_dominance_bonus=btc_dominance_bonus,
        notes=notes,
    )


def _reliability_contribution(symbol: str, reliability_map: dict[str, Any]) -> _Contribution:
    value, note = _reliability_bonus(symbol, reliability_map)
    return _Contribution(value=value, note=note)


def _cost_penalty_contribution(features: dict[str, Any]) -> _CostPenaltyContribution:
    spread_penalty, spread_note = _spread_penalty(features)
    cost_penalty, cost_note, net_edge_pct = _cost_penalty_from_breakdown(features)
    return _CostPenaltyContribution(
        spread_penalty=spread_penalty,
        cost_penalty=cost_penalty,
        net_edge_pct=net_edge_pct,
        spread_note=spread_note,
        cost_note=cost_note,
    )


def _basket_contribution(features: dict[str, Any], held_correlation_map: dict[str, float]) -> _BasketContribution:
    basket_fit_bonus, basket_note = _basket_fit_bonus(features, held_correlation_map)
    correlation_penalty, correlation_note = _correlation_penalty(features)
    notes = tuple(note for note in (basket_note, correlation_note) if note)
    return _BasketContribution(
        basket_fit_bonus=basket_fit_bonus,
        correlation_penalty=correlation_penalty,
        notes=notes,
    )


def _build_score_breakdown(
    *,
    setup: _SetupContribution,
    reliability: _Contribution,
    cost: _CostPenaltyContribution,
    basket: _BasketContribution,
) -> dict[str, Any]:
    return {
        "setup_contribution": round(setup.total, 3),
        "reliability_contribution": round(reliability.value, 3),
        "cost_penalty_contribution": round(-(cost.spread_penalty + cost.cost_penalty), 3),
        "basket_contribution": round(basket.total, 3),
        "setup_pts": round(setup.entry_score, 1),
        "reflex_bonus": round(setup.reflex_bonus, 1),
        "divergence_bonus": round(setup.divergence_bonus, 1),
        "fear_greed_bonus": round(setup.fear_greed_bonus, 1),
        "btc_dominance_bonus": round(setup.btc_dominance_bonus, 1),
        "reliability_bonus": round(reliability.value, 1),
        "basket_fit": round(basket.basket_fit_bonus, 1),
        "spread_penalty": -round(cost.spread_penalty, 1),
        "cost_penalty": -round(cost.cost_penalty, 1),
        "correlation_penalty": -round(basket.correlation_penalty, 1),
        "net_edge_pct": round(cost.net_edge_pct, 3),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_final_score(
    features: dict[str, Any],
    *,
    reliability_map: dict[str, Any] | None = None,
    held_correlation_map: dict[str, float] | None = None,
) -> FinalTradeScore:
    """Compute the unified final trade score for one candidate.

    Args:
        features:              Full feature dict for the symbol (post-entry_verifier).
        reliability_map:       Optional per-symbol win-rate records.
                               { "ETH/USD": {"win_rate": 0.62, "trade_count": 18} }
        held_correlation_map:  Optional correlations of this candidate with each
                               currently held symbol.
                               { "BTC/USD": 0.78, "SOL/USD": 0.32 }

    Returns:
        FinalTradeScore dataclass with final_score and full breakdown.
    """
    symbol = str(features.get("symbol", "UNKNOWN/USD"))
    setup = _setup_contribution(features)
    reliability = _reliability_contribution(symbol, reliability_map or {})
    cost = _cost_penalty_contribution(features)
    basket = _basket_contribution(features, held_correlation_map or {})

    final = (
        setup.total
        + reliability.value
        + basket.basket_fit_bonus
        - cost.spread_penalty
        - cost.cost_penalty
        - basket.correlation_penalty
    )
    final = _clamp(round(final, 2), 0.0, 100.0)

    notes = [n for n in [*setup.notes, reliability.note, *basket.notes, cost.spread_note, cost.cost_note] if n]
    score_breakdown = _build_score_breakdown(
        setup=setup,
        reliability=reliability,
        cost=cost,
        basket=basket,
    )
    score_breakdown["notes"] = notes

    return FinalTradeScore(
        symbol=symbol,
        final_score=final,
        entry_score=setup.entry_score,
        reflex_bonus=setup.reflex_bonus,
        divergence_bonus=setup.divergence_bonus,
        fear_greed_bonus=setup.fear_greed_bonus,
        btc_dominance_bonus=setup.btc_dominance_bonus,
        reliability_bonus=reliability.value,
        basket_fit_bonus=basket.basket_fit_bonus,
        spread_penalty=cost.spread_penalty,
        cost_penalty=cost.cost_penalty,
        correlation_penalty=basket.correlation_penalty,
        net_edge_pct=cost.net_edge_pct,
        score_breakdown=score_breakdown,
        breakdown_notes=notes,
    )
