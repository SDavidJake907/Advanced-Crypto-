"""Deterministic market trend state using BTC/USD as the market-wide barometer.

Replaces the Phi-3 market state review. Produces a human-readable context string
for Nemo and a numeric bias that adjusts per-symbol entry scores.

No hard blocks — trend state is advisory only. Nemo still decides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MarketTrendState:
    trend: str          # "bull" | "neutral" | "bear"
    strength: int       # -4 to +4 (sum of signals)
    score_bias: float   # points added/subtracted from entry scores
    summary: str        # short string passed to Nemo as context

    def to_nemo_context(self) -> str:
        return self.summary


def _safe_batch_value(features_batch: dict[str, Any], key: str, idx: int, default: Any) -> Any:
    values = features_batch.get(key, default)
    if isinstance(values, (str, bytes)):
        return values
    if hasattr(values, "__getitem__"):
        try:
            return values[idx]
        except (IndexError, KeyError, TypeError):
            return default
    return values if values is not None else default


def compute_market_trend(
    features_batch: dict[str, Any],
    symbols: list[str],
    *,
    bear_score_penalty: float = 8.0,
    bull_score_boost: float = 4.0,
) -> MarketTrendState:
    """Compute market-wide trend state from BTC/USD features.

    Signals scored (each -1 or +1):
      1. trend_1h direction (BTC up/down over last 8 1h bars)
      2. ema9 above ema20 (short-term EMA alignment)
      3. trend_confirmed (structural trend confirmed)
      4. macro_30d direction (monthly context: bull/bear/sideways)

    strength >=  3 → bull  (score +bull_score_boost)
    strength <= -2 → bear  (score -bear_score_penalty)
    otherwise      → neutral
    """
    btc_sym = next(
        (s for s in symbols if "BTC" in s.upper()),
        None,
    )
    if btc_sym is None:
        return MarketTrendState(
            trend="unknown",
            strength=0,
            score_bias=0.0,
            summary="market:unknown bias:neutral",
        )

    sym_list = list(symbols)
    try:
        idx = sym_list.index(btc_sym)
    except ValueError:
        return MarketTrendState(
            trend="unknown",
            strength=0,
            score_bias=0.0,
            summary="market:unknown bias:neutral",
        )

    # --- Extract BTC signals safely ---
    trend_1h = int(_safe_batch_value(features_batch, "trend_1h", idx, 0) or 0)
    trend_conf = bool(_safe_batch_value(features_batch, "trend_confirmed", idx, False))
    macro = str(_safe_batch_value(features_batch, "macro_30d", idx, "sideways") or "sideways").lower()

    ema9_ok = bool(_safe_batch_value(features_batch, "ema9_above_ema20", idx, False))
    if not ema9_ok:
        struct_l23 = features_batch.get("struct_l23", {}) if isinstance(features_batch.get("struct_l23", {}), dict) else {}
        ema9_ok = bool(_safe_batch_value(struct_l23, "ema9_above_ema20", idx, False))

    # --- Score signals ---
    s1 = 1 if trend_1h > 0 else (-1 if trend_1h < 0 else 0)   # 1h direction
    s2 = 1 if ema9_ok else -1                                   # EMA alignment
    s3 = 1 if trend_conf else -1                                # structural trend
    s4 = 1 if macro == "bull" else (-1 if macro == "bear" else 0)  # monthly context

    strength = s1 + s2 + s3 + s4  # range: -4 to +4

    if strength >= 3:
        trend = "bull"
        score_bias = bull_score_boost
        lane_bias = "long-friendly"
    elif strength <= -2:
        trend = "bear"
        score_bias = -bear_score_penalty
        lane_bias = "caution"
    else:
        trend = "neutral"
        score_bias = 0.0
        lane_bias = "neutral"

    summary = (
        f"market:{trend}({strength:+d}) "
        f"btc_1h:{'up' if trend_1h > 0 else 'down' if trend_1h < 0 else 'flat'} "
        f"ema:{'ok' if ema9_ok else 'cross'} "
        f"30d:{macro} "
        f"bias:{lane_bias}"
    )

    return MarketTrendState(
        trend=trend,
        strength=strength,
        score_bias=score_bias,
        summary=summary,
    )
