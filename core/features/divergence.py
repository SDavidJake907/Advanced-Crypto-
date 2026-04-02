from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DivergenceSignal:
    bullish_divergence: bool
    bearish_divergence: bool
    divergence_strength: float
    divergence_age_bars: int

    @property
    def rsi_divergence(self) -> int:
        if self.bullish_divergence:
            return 1
        if self.bearish_divergence:
            return -1
        return 0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rsi_series(closes: np.ndarray | list[float], period: int = 14) -> np.ndarray:
    """Wilder RSI, same length as closes, seeded with neutral values."""
    arr = np.asarray(closes, dtype=np.float64).ravel()
    n = arr.size
    result = np.full(n, 50.0, dtype=np.float64)
    if n < period + 2:
        return result

    deltas = np.diff(arr)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for idx in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + float(gains[idx])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[idx])) / period
        if avg_loss < 1e-10:
            result[idx + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[idx + 1] = 100.0 - (100.0 / (1.0 + rs))
    return result


def _swing_points(values: np.ndarray, lookback: int) -> tuple[list[int], list[int]]:
    lows: list[int] = []
    highs: list[int] = []
    if values.size < (lookback * 2 + 1):
        return lows, highs
    for idx in range(lookback, values.size - lookback):
        window = values[idx - lookback : idx + lookback + 1]
        if values[idx] == np.min(window):
            lows.append(idx)
        if values[idx] == np.max(window):
            highs.append(idx)
    return lows, highs


def detect_rsi_divergence(
    closes: np.ndarray | list[float],
    *,
    rsi_values: np.ndarray | list[float] | None = None,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> DivergenceSignal:
    prices = np.asarray(closes, dtype=np.float64).ravel()
    if prices.size == 0:
        return DivergenceSignal(False, False, 0.0, 99)

    if prices.size > scan_bars:
        prices = prices[-scan_bars:]

    rsi = (
        rsi_series(prices, period=rsi_period)
        if rsi_values is None
        else np.asarray(rsi_values, dtype=np.float64).ravel()[-prices.size :]
    )
    if rsi.size != prices.size:
        rsi = rsi_series(prices, period=rsi_period)

    min_bars = rsi_period + swing_lookback * 2 + 2
    if prices.size < min_bars:
        return DivergenceSignal(False, False, 0.0, 99)

    swing_lows, swing_highs = _swing_points(prices, swing_lookback)
    signal = DivergenceSignal(False, False, 0.0, 99)

    if len(swing_lows) >= 2:
        i1, i2 = swing_lows[-2], swing_lows[-1]
        p1, p2 = float(prices[i1]), float(prices[i2])
        r1, r2 = float(rsi[i1]), float(rsi[i2])
        if p2 < p1 * 0.999 and r2 > r1 + 1.0:
            price_drop = (p1 - p2) / max(abs(p1), 1e-9)
            rsi_lift = (r2 - r1) / 15.0
            strength = _clamp(price_drop * 8.0 + rsi_lift, 0.0, 1.0)
            age = min(99, prices.size - 1 - i2)
            signal = DivergenceSignal(True, False, strength, age)

    if len(swing_highs) >= 2:
        i1, i2 = swing_highs[-2], swing_highs[-1]
        p1, p2 = float(prices[i1]), float(prices[i2])
        r1, r2 = float(rsi[i1]), float(rsi[i2])
        if p2 > p1 * 1.001 and r2 < r1 - 1.0:
            price_rise = (p2 - p1) / max(abs(p1), 1e-9)
            rsi_fade = (r1 - r2) / 15.0
            strength = _clamp(price_rise * 8.0 + rsi_fade, 0.0, 1.0)
            age = min(99, prices.size - 1 - i2)
            bearish = DivergenceSignal(False, True, strength, age)
            if (not signal.bullish_divergence and not signal.bearish_divergence) or bearish.divergence_age_bars <= signal.divergence_age_bars:
                signal = bearish

    return signal


def bullish_divergence(
    closes: np.ndarray | list[float],
    *,
    rsi_values: np.ndarray | list[float] | None = None,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> bool:
    return detect_rsi_divergence(
        closes,
        rsi_values=rsi_values,
        rsi_period=rsi_period,
        swing_lookback=swing_lookback,
        scan_bars=scan_bars,
    ).bullish_divergence


def bearish_divergence(
    closes: np.ndarray | list[float],
    *,
    rsi_values: np.ndarray | list[float] | None = None,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> bool:
    return detect_rsi_divergence(
        closes,
        rsi_values=rsi_values,
        rsi_period=rsi_period,
        swing_lookback=swing_lookback,
        scan_bars=scan_bars,
    ).bearish_divergence


def divergence_strength(
    closes: np.ndarray | list[float],
    *,
    rsi_values: np.ndarray | list[float] | None = None,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> float:
    return detect_rsi_divergence(
        closes,
        rsi_values=rsi_values,
        rsi_period=rsi_period,
        swing_lookback=swing_lookback,
        scan_bars=scan_bars,
    ).divergence_strength


def divergence_age_bars(
    closes: np.ndarray | list[float],
    *,
    rsi_values: np.ndarray | list[float] | None = None,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> int:
    return detect_rsi_divergence(
        closes,
        rsi_values=rsi_values,
        rsi_period=rsi_period,
        swing_lookback=swing_lookback,
        scan_bars=scan_bars,
    ).divergence_age_bars


def compute_rsi_divergence_batch(
    frames: list[pd.DataFrame],
    *,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bullish: list[bool] = []
    bearish: list[bool] = []
    strengths: list[float] = []
    ages: list[int] = []
    signals: list[int] = []

    for frame in frames:
        if frame.empty or "close" not in frame.columns:
            signal = DivergenceSignal(False, False, 0.0, 99)
        else:
            signal = detect_rsi_divergence(
                frame["close"].to_numpy(dtype=np.float64),
                rsi_period=rsi_period,
                swing_lookback=swing_lookback,
                scan_bars=scan_bars,
            )
        bullish.append(signal.bullish_divergence)
        bearish.append(signal.bearish_divergence)
        strengths.append(signal.divergence_strength)
        ages.append(signal.divergence_age_bars)
        signals.append(signal.rsi_divergence)

    return (
        np.asarray(bullish, dtype=bool),
        np.asarray(bearish, dtype=bool),
        np.asarray(strengths, dtype=np.float64),
        np.asarray(ages, dtype=np.int8),
        np.asarray(signals, dtype=np.int8),
    )
