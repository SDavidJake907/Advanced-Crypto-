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

    # --- Extract BTC signals ---
    trend_1h = int(features_batch.get("trend_1h", [0])[idx] if hasattr(features_batch.get("trend_1h", []), "__getitem__") else 0)
    ema9_ok = bool(features_batch.get("ema9_above_ema20", [False])[idx] if hasattr(features_batch.get("ema9_above_ema20", []), "__getitem__") else False)
    trend_conf = bool(features_batch.get("trend_confirmed", [False])[idx] if hasattr(features_batch.get("trend_confirmed", []), "__getitem__") else False)
    macro = str(features_batch.get("macro_30d", ["sideways"])[idx] if hasattr(features_batch.get("macro_30d", []), "__getitem__") else "sideways").lower()

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
