from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class Features:
    trend: Literal["up", "down", "flat"]
    trend_strength: float
    volatility: Literal["low", "normal", "high"]
    atr_pct: float
    breakout: bool
    dist_to_breakout_pct: float
    volume_confirm: bool
    stretch: Literal["normal", "stretched"]
    z20: float


def ema(values: list[float], period: int) -> float:
    if len(values) < period:
        return float("nan")
    k = 2 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return float("nan")
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return float("nan")
    return sum(trs[-period:]) / period


def zscore(values: list[float], period: int) -> float:
    if len(values) < period:
        return float("nan")
    window = values[-period:]
    mean = sum(window) / period
    var = sum((v - mean) ** 2 for v in window) / period
    std = math.sqrt(max(var, 1e-12))
    return (values[-1] - mean) / std


def compute_features(
    *,
    closes_15m: list[float],
    highs_15m: list[float],
    lows_15m: list[float],
    volumes_15m: list[float],
    closes_1h: list[float],
) -> Features | None:
    if len(closes_15m) < 60 or len(closes_1h) < 60:
        return None

    ema21_1h = ema(closes_1h, 21)
    ema55_1h = ema(closes_1h, 55)
    trend = "flat"
    if ema21_1h > ema55_1h:
        trend = "up"
    elif ema21_1h < ema55_1h:
        trend = "down"

    trend_strength = (ema21_1h - ema55_1h) / closes_1h[-1]

    atr14 = atr(highs_15m, lows_15m, closes_15m, 14)
    atr_pct = atr14 / closes_15m[-1] if atr14 == atr14 else float("nan")

    # Volatility regime based on ATR%
    atr_window = []
    for i in range(14, len(closes_15m)):
        sub_atr = atr(highs_15m[: i + 1], lows_15m[: i + 1], closes_15m[: i + 1], 14)
        if sub_atr == sub_atr:
            atr_window.append(sub_atr / closes_15m[i])
    atr_mean = sum(atr_window[-30:]) / max(len(atr_window[-30:]), 1)
    vol = "normal"
    if atr_pct > atr_mean * 1.5:
        vol = "high"
    elif atr_pct < atr_mean * 0.7:
        vol = "low"

    # Compare the latest close against the prior 20 completed candles.
    donchian_high = max(highs_15m[-21:-1])
    donchian_low = min(lows_15m[-21:-1])
    breakout = closes_15m[-1] > donchian_high
    dist_to_breakout_pct = (
        abs(closes_15m[-1] - donchian_high) / closes_15m[-1] * 100.0
    )

    # Volume confirmation
    vol_avg = sum(volumes_15m[-20:]) / 20
    vol_ratio = volumes_15m[-1] / vol_avg if vol_avg > 0 else 0.0
    volume_confirm = vol_ratio >= 1.2

    # Stretch filter (z-score)
    z20 = zscore(closes_15m, 20)
    stretch = "stretched" if abs(z20) >= 2.5 else "normal"

    return Features(
        trend=trend,
        trend_strength=trend_strength,
        volatility=vol,
        atr_pct=atr_pct,
        breakout=breakout,
        dist_to_breakout_pct=dist_to_breakout_pct,
        volume_confirm=volume_confirm,
        stretch=stretch,
        z20=z20,
    )
