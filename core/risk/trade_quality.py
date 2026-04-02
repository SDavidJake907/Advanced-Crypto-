from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.config.runtime import get_runtime_setting, is_meme_lane


@dataclass
class TradeQuality:
    score: float                # 0.0 – 1.0
    band: str                   # "excellent" | "good" | "thin" | "poor"
    size_scale: float           # multiply into risk_per_trade_pct / notional
    limit_offset_scale: float   # multiply into limit_offset_bps
    slippage_risk: str          # "low" | "medium" | "high"
    thin_market_flag: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_trade_quality(features: dict[str, Any], lane: str) -> TradeQuality:
    """Evaluate execution quality for a given setup.

    Answers: "assuming the setup is valid, how expensive / messy is it to get
    in right now?"

    Does NOT decide whether the setup is valid — that is entry_score's job.
    These two must remain separate.

    Inputs (all already present in the features dict):
        spread_pct   — bid/ask spread as % of mid price
        atr          — ATR(14) in price units
        price        — current mid price
        volume_ratio — current volume / 20-bar average volume

    Outputs:
        score              0.0–1.0
        band               excellent / good / thin / poor
        size_scale         1.0 = full size, 0.75 = thin, 0.50 = poor
        limit_offset_scale 1.0 = normal, 1.5 = thin, 2.0 = poor
        slippage_risk      low / medium / high
        thin_market_flag   True when band is thin or poor
    """
    spread_pct   = max(float(features.get("spread_pct",   0.0) or 0.0), 0.0)
    price        = max(float(features.get("price",        1.0) or 1.0), 1e-10)
    atr          = max(float(features.get("atr",          0.0) or 0.0), 0.0)
    atr_pct      = atr / price
    volume_ratio = max(float(features.get("volume_ratio", 1.0) or 1.0), 0.0)

    max_spread_key = "MEME_EXEC_MAX_SPREAD_PCT" if is_meme_lane(lane) else "EXEC_MAX_SPREAD_PCT"
    max_spread_pct = float(get_runtime_setting(max_spread_key))

    # ── Sub-score 1: Spread quality (50% weight) ─────────────────────────────
    # 1.0 = zero spread, 0.0 = at the hard-block ceiling
    spread_score = max(0.0, 1.0 - spread_pct / max(max_spread_pct, 0.01))

    # ── Sub-score 2: ATR vs spread (30% weight) ──────────────────────────────
    # How large is the expected move relative to execution friction?
    # Ratio >= 5  → score 1.0 (ATR is 5× spread, plenty of room)
    # Ratio == 1  → score 0.0 (move ≈ friction, not worth the cost)
    spread_frac     = max(spread_pct / 100.0, 1e-6)
    atr_spread_ratio = atr_pct / spread_frac
    atr_score = min(1.0, max(0.0, (atr_spread_ratio - 1.0) / 4.0))

    # ── Sub-score 3: Depth / liquidity proxy (20% weight) ────────────────────
    # volume_ratio: 2.0+ = excellent participation, 0.5 = thin
    depth_score = min(1.0, max(0.0, (volume_ratio - 0.5) / 1.5))

    score = round(spread_score * 0.50 + atr_score * 0.30 + depth_score * 0.20, 3)

    reasons: list[str] = []

    if score >= 0.75:
        band               = "excellent"
        size_scale         = 1.00
        limit_offset_scale = 1.0
        slippage_risk      = "low"
        thin_market_flag   = False
    elif score >= 0.50:
        band               = "good"
        size_scale         = 1.00
        limit_offset_scale = 1.0
        slippage_risk      = "low"
        thin_market_flag   = False
    elif score >= 0.25:
        band               = "thin"
        size_scale         = 0.75
        limit_offset_scale = 1.5
        slippage_risk      = "medium"
        thin_market_flag   = True
        reasons.append("thin_execution")
    else:
        band               = "poor"
        size_scale         = 0.50
        limit_offset_scale = 2.0
        slippage_risk      = "high"
        thin_market_flag   = True
        reasons.append("poor_execution")

    if spread_score < 0.40:
        reasons.append(f"spread_wide({spread_pct:.3f}%)")
    if depth_score < 0.30:
        reasons.append(f"depth_thin(vol_ratio={volume_ratio:.2f})")
    if atr_score < 0.20 and spread_pct > 0.10:
        reasons.append("atr_tight_vs_spread")

    return TradeQuality(
        score=score,
        band=band,
        size_scale=size_scale,
        limit_offset_scale=limit_offset_scale,
        slippage_risk=slippage_risk,
        thin_market_flag=thin_market_flag,
        reasons=reasons,
    )
