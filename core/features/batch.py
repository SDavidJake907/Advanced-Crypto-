from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd

import cuda_features
from core.features.cuda_atr import compute_atr_gpu
from core.features.cuda_bollinger import compute_bollinger_gpu
from core.features.cuda_correlation import compute_correlation_gpu
from core.features.cuda_rsi import compute_rsi_gpu
from core.features.divergence import compute_rsi_divergence_batch
from core.features.multi_timeframe.trend_1h import compute_trend_1h_batch
from core.features.multi_timeframe.trend_7d import compute_regime_7d_batch
from core.features.multi_timeframe.trend_30d import compute_macro_30d_batch
from core.features.trend_state import compute_trend_state
from core.policy.pipeline import apply_policy_pipeline

try:
    from core.features.cuda_northstar import (
        compute_northstar_batch_features_gpu,
        compute_northstar_fingerprint_gpu,
    )
except Exception:  # pragma: no cover - build/runtime fallback
    def compute_northstar_batch_features_gpu(prices: np.ndarray) -> dict[str, np.ndarray]:
        n_assets = int(np.asarray(prices).shape[0]) if np.asarray(prices).ndim == 2 else 0
        return {
            "hurst": np.full(n_assets, 0.5, dtype=np.float64),
            "entropy": np.full(n_assets, 0.5, dtype=np.float64),
            "autocorr": np.zeros(n_assets, dtype=np.float64),
        }

    def compute_northstar_fingerprint_gpu(
        prices: np.ndarray,
        *,
        btc_idx: int,
        eth_idx: int,
    ) -> dict[str, Any]:
        return {
            "metrics": np.zeros(8, dtype=np.float64),
            "r_mkt": 0.0,
            "r_btc": 0.0,
            "r_eth": 0.0,
            "breadth": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "rv_mkt": 0.0,
            "corr_avg": 0.0,
        }


_ALIGN_MIN_BARS = 10  # Low threshold so symbols warm up fast; momentum_30 will be 0 for symbols with < 31 bars (acceptable)


def _align_frames(ohlc_by_symbol: dict[str, pd.DataFrame]) -> tuple[list[str], list[pd.DataFrame]]:
    # Filter out cold symbols — even one symbol with 1 bar would otherwise truncate
    # the entire batch to 1 bar, making all features return defaults for everyone.
    qualified = {s: f for s, f in ohlc_by_symbol.items() if len(f) >= _ALIGN_MIN_BARS}
    if not qualified:
        qualified = ohlc_by_symbol  # graceful fallback: include all if none qualify

    symbols = list(qualified.keys())
    if not symbols:
        return [], []

    min_len = min(len(frame) for frame in qualified.values())
    if min_len == 0:
        return symbols, [frame.iloc[0:0].copy() for frame in qualified.values()]

    frames = [qualified[symbol].tail(min_len).reset_index(drop=True) for symbol in symbols]
    return symbols, frames


def _prepare_frames(ohlc_by_symbol: dict[str, pd.DataFrame]) -> tuple[list[str], list[pd.DataFrame], np.ndarray]:
    symbols = list(ohlc_by_symbol.keys())
    if not symbols:
        return [], [], np.array([], dtype=np.int64)

    frames = [ohlc_by_symbol[symbol].reset_index(drop=True) for symbol in symbols]
    history_points = np.asarray([len(frame) for frame in frames], dtype=np.int64)
    return symbols, frames, history_points


def _pad_series(values: np.ndarray, target_len: int, *, pad_value: float) -> np.ndarray:
    if values.size >= target_len:
        return values[-target_len:]
    pad_width = target_len - values.size
    return np.pad(values, (pad_width, 0), mode="constant", constant_values=pad_value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_score_0_100(value: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return _clamp01((float(value) - low) / (high - low)) * 100.0


def _inverted_bounded_score_0_100(value: float, low: float, high: float) -> float:
    return 100.0 - _bounded_score_0_100(value, low, high)


def _percentile_score_0_100(values: np.ndarray, value: float, *, invert: bool = False) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 50.0
    pct = float(np.mean(arr <= float(value)))
    score = pct * 100.0
    return 100.0 - score if invert else score


def _mean_abs_correlation(correlation_row: Any) -> float:
    row = np.asarray(correlation_row, dtype=np.float64).ravel()
    if row.size == 0:
        return 0.0
    filtered = [abs(float(v)) for v in row if np.isfinite(v) and abs(float(v)) < 0.999]
    if not filtered:
        return 0.0
    return float(np.mean(filtered))


def _cross_section_percentile_scores(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, 50.0, dtype=np.float64)
    scores = np.asarray([_percentile_score_0_100(finite, value, invert=invert) for value in arr], dtype=np.float64)
    return scores


def _volatility_state_from_metrics(
    *,
    atr_pct: float,
    volatility_percentile: float,
    compression_score: float,
    expansion_score: float,
    atr_expanding: bool,
) -> str:
    if atr_pct >= 0.05 or volatility_percentile >= 92.0:
        return "overheated"
    if atr_expanding and expansion_score >= 62.0:
        return "expanding"
    if compression_score >= 68.0:
        return "compressed"
    return "normal"


def _build_padded_matrix(
    frames: list[pd.DataFrame],
    column: str,
    target_len: int,
    *,
    default_value: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for frame in frames:
        if frame.empty or column not in frame.columns:
            rows.append(np.full(target_len, default_value, dtype=np.float64))
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        pad_value = float(values[0]) if values.size else default_value
        if column == "volume":
            pad_value = 0.0
        rows.append(_pad_series(values, target_len, pad_value=pad_value))
    return np.vstack(rows) if rows else np.zeros((0, target_len), dtype=np.float64)


def _compute_momentum(prices: np.ndarray, lookback: int) -> np.ndarray:
    n_assets, n_points = prices.shape
    if n_points < lookback + 1:
        return np.zeros(n_assets, dtype=np.float64)
    latest = prices[:, -1]
    previous = prices[:, -(lookback + 1)]
    valid = previous > 0.0
    momentum = np.zeros(n_assets, dtype=np.float64)
    momentum[valid] = (latest[valid] / previous[valid]) - 1.0
    return momentum


def _compute_rotation_score(
    *,
    momentum_1: np.ndarray,
    momentum_5: np.ndarray,
    momentum_14: np.ndarray,
    momentum_30: np.ndarray,
    volume_ratio: np.ndarray,
    rsi: np.ndarray,
    trend_1h: np.ndarray,
    regime_7d: list[str],
    correlation: np.ndarray,
) -> np.ndarray:
    n_assets = momentum_5.shape[0]
    if n_assets == 0:
        return np.array([], dtype=np.float64)

    trend_bonus = np.where(trend_1h > 0, 0.15, np.where(trend_1h < 0, -0.15, 0.0))
    regime_bonus = np.asarray([0.1 if regime == "trending" else 0.0 for regime in regime_7d], dtype=np.float64)
    overextension_penalty = np.clip((rsi - 70.0) / 30.0, 0.0, 1.0) * 0.2

    if correlation.size:
        crowding_penalty = np.clip((np.mean(np.abs(correlation), axis=1) - 0.5) / 0.5, 0.0, 1.0) * 0.15
    else:
        crowding_penalty = np.zeros(n_assets, dtype=np.float64)

    # Volume confirmation: log-scaled bonus/penalty for volume vs baseline
    volume_bonus = 0.1 * np.log(np.maximum(volume_ratio, 0.01))

    return (
        (0.6 * momentum_5)
        + (0.2 * momentum_1)
        + (0.2 * momentum_14)
        + trend_bonus
        + regime_bonus
        + volume_bonus
        - overextension_penalty
        - crowding_penalty
    )


def _compute_volume_ratio(frames: list[pd.DataFrame], lookback: int = 20) -> np.ndarray:
    ratios: list[float] = []
    for frame in frames:
        if "volume" not in frame.columns or frame.empty:
            ratios.append(1.0)
            continue
        series = frame["volume"].astype(float)
        window = series.tail(lookback + 1)
        baseline = float(window.iloc[:-1].mean()) if len(window) > 1 else float(window.mean())
        current = float(series.iloc[-1])
        if baseline <= 0.0:
            ratios.append(1.0)
        else:
            ratios.append(current / baseline)
    return np.asarray(ratios, dtype=np.float64)


def _compute_volume_surge(frames: list[pd.DataFrame], lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    surge_scores: list[float] = []
    surge_flags: list[bool] = []
    for frame in frames:
        if "volume" not in frame.columns or frame.empty:
            surge_scores.append(0.0)
            surge_flags.append(False)
            continue
        series = frame["volume"].astype(float)
        window = series.tail(lookback + 1)
        baseline = float(window.iloc[:-1].mean()) if len(window) > 1 else float(window.mean())
        current = float(series.iloc[-1])
        if baseline <= 0.0:
            surge_scores.append(0.0)
            surge_flags.append(False)
            continue
        surge = max((current / baseline) - 1.0, 0.0)
        surge_scores.append(surge)
        surge_flags.append(surge >= 0.5)
    return np.asarray(surge_scores, dtype=np.float64), np.asarray(surge_flags, dtype=bool)


def _ema_series(values: np.ndarray, span: int) -> np.ndarray:
    """Simple EMA for a 1-D array."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _compute_vwio(frames: list[pd.DataFrame], ema_span: int = 10) -> np.ndarray:
    """Volume Weighted Imbalance Oscillator per asset.

    VWIO = EMA(buy_vol - sell_vol, 10) / EMA(total_vol, 10)

    where buy_vol  = volume when close > open  (up-close bar)
          sell_vol = volume when close < open  (down-close bar)

    Output is bounded roughly in [-1, +1]:
      > 0  → buying pressure dominant
      < 0  → selling pressure dominant
      ~0   → balanced / indeterminate
    """
    results: list[float] = []
    for frame in frames:
        if (
            frame.empty
            or "close" not in frame.columns
            or "open" not in frame.columns
            or "volume" not in frame.columns
            or len(frame) < ema_span + 1
        ):
            results.append(0.0)
            continue

        closes = frame["close"].to_numpy(dtype=np.float64)
        opens = frame["open"].to_numpy(dtype=np.float64)
        volumes = frame["volume"].to_numpy(dtype=np.float64)

        buy_vol = np.where(closes > opens, volumes, 0.0)
        sell_vol = np.where(closes < opens, volumes, 0.0)
        imbalance = buy_vol - sell_vol

        ema_imb = _ema_series(imbalance, ema_span)
        ema_vol = _ema_series(volumes, ema_span)

        last_vol = float(ema_vol[-1])
        if last_vol < 1e-10:
            results.append(0.0)
        else:
            results.append(float(np.clip(ema_imb[-1] / last_vol, -1.0, 1.0)))

    return np.asarray(results, dtype=np.float64)


def _compute_obv_divergence(frames: list[pd.DataFrame], lookback: int = 5) -> np.ndarray:
    """Compute OBV divergence per asset.

    Returns int array per asset:
      +1 = bullish divergence  (price falling, OBV rising  — smart money accumulating)
      -1 = bearish divergence  (price rising,  OBV falling — distribution, reversal risk)
       0 = no divergence
    """
    results: list[int] = []
    for frame in frames:
        if frame.empty or "close" not in frame.columns or "volume" not in frame.columns or len(frame) < lookback + 2:
            results.append(0)
            continue
        closes = frame["close"].to_numpy(dtype=np.float64)
        volumes = frame["volume"].to_numpy(dtype=np.float64)

        # Build OBV series
        obv = np.zeros(len(closes), dtype=np.float64)
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]

        # Compare trend over last `lookback` bars
        price_now, price_then = float(closes[-1]), float(closes[-(lookback + 1)])
        obv_now, obv_then = float(obv[-1]), float(obv[-(lookback + 1)])

        price_up = price_now > price_then * 1.002   # 0.2% threshold to avoid noise
        price_down = price_now < price_then * 0.998
        obv_up = obv_now > obv_then
        obv_down = obv_now < obv_then

        if price_up and obv_down:
            results.append(-1)   # bearish divergence
        elif price_down and obv_up:
            results.append(1)    # bullish divergence
        else:
            results.append(0)
    return np.asarray(results, dtype=np.int8)


def _rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's smoothed RSI — returns array same length as closes (first period bars = 50.0)."""
    n = len(closes)
    result = np.full(n, 50.0, dtype=np.float64)
    if n < period + 2:
        return result
    deltas = np.diff(closes.astype(np.float64))
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
        if avg_loss < 1e-10:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return result


def _compute_rsi_divergence(
    frames: list[pd.DataFrame],
    *,
    rsi_period: int = 14,
    swing_lookback: int = 5,
    scan_bars: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect RSI divergence per asset.

    Returns three arrays (one value per asset):
        signal:   int8   — +1 bullish, -1 bearish, 0 none
        strength: float64 — 0.0–1.0, magnitude of RSI vs price mismatch
        age:      int8   — bars since the most-recent divergence swing (capped at 99)

    Bullish:  price lower low  + RSI higher low  → momentum improving into dip
    Bearish:  price higher high + RSI lower high  → momentum fading into extension
    """
    signals: list[int] = []
    strengths: list[float] = []
    ages: list[int] = []

    min_bars = rsi_period + swing_lookback * 2 + 2
    for frame in frames:
        if frame.empty or "close" not in frame.columns or len(frame) < min_bars:
            signals.append(0)
            strengths.append(0.0)
            ages.append(99)
            continue

        closes = frame["close"].to_numpy(dtype=np.float64)
        closes = closes[-scan_bars:] if len(closes) > scan_bars else closes
        rsi = _rsi_series(closes, period=rsi_period)
        n = len(closes)

        swing_lows: list[int] = []
        swing_highs: list[int] = []
        for i in range(swing_lookback, n - swing_lookback):
            window = closes[i - swing_lookback : i + swing_lookback + 1]
            if closes[i] == np.min(window):
                swing_lows.append(i)
            if closes[i] == np.max(window):
                swing_highs.append(i)

        signal = 0
        strength = 0.0
        age = 99

        # Bullish: price lower low + RSI higher low
        if len(swing_lows) >= 2:
            i1, i2 = swing_lows[-2], swing_lows[-1]
            p1, p2 = closes[i1], closes[i2]
            r1, r2 = rsi[i1], rsi[i2]
            if p2 < p1 * 0.999 and r2 > r1 + 1.0:
                signal = 1
                strength = min(1.0, (r2 - r1) / 15.0)
                age = min(99, n - 1 - i2)

        # Bearish: price higher high + RSI lower high (overrides bullish if more recent)
        if len(swing_highs) >= 2:
            i1, i2 = swing_highs[-2], swing_highs[-1]
            p1, p2 = closes[i1], closes[i2]
            r1, r2 = rsi[i1], rsi[i2]
            if p2 > p1 * 1.001 and r2 < r1 - 1.0:
                bear_age = min(99, n - 1 - i2)
                if signal == 0 or bear_age <= age:
                    signal = -1
                    strength = min(1.0, (r1 - r2) / 15.0)
                    age = bear_age

        signals.append(signal)
        strengths.append(strength)
        ages.append(age)

    return (
        np.asarray(signals, dtype=np.int8),
        np.asarray(strengths, dtype=np.float64),
        np.asarray(ages, dtype=np.int8),
    )


def _empty_struct(n: int, price_ref: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ema_9": price_ref.copy(),
        "ema_20": price_ref.copy(),
        "ema_26": price_ref.copy(),
        "ema9_above_ema20": np.zeros(n, dtype=bool),
        "ema9_above_ema26": np.zeros(n, dtype=bool),
        "price_above_ema20": np.zeros(n, dtype=bool),
        "ema_slope_9": np.zeros(n, dtype=np.float64),
        "ema_cross_distance_pct": np.zeros(n, dtype=np.float64),
        "range_pos_1h": np.full(n, 0.5, dtype=np.float64),
        "range_pos_4h": np.full(n, 0.5, dtype=np.float64),
        "range_breakout_1h": np.zeros(n, dtype=bool),
        "higher_low_count": np.zeros(n, dtype=np.int8),
        "pivot_break": np.zeros(n, dtype=bool),
        "pullback_hold": np.zeros(n, dtype=bool),
    }


def _compute_structure_features(
    symbols: list[str],
    primary_by_symbol: dict[str, pd.DataFrame] | None,
    range_by_symbol: dict[str, pd.DataFrame] | None,
    *,
    range_window: int = 24,
    range_4h_window: int = 16,
    breakout_lookback: int = 20,
) -> dict[str, np.ndarray]:
    """Channel continuation / breakout structure features.

    primary_by_symbol  – bars used for EMA9/EMA20, higher lows, pivot, pullback
    range_by_symbol    – bars used for range position and breakout detection
    range_window       – how many range bars define the "full range" (range_pos_1h)
    range_4h_window    – how many range bars define the "zone" (range_pos_4h)
    breakout_lookback  – how many prior range bars define the breakout ceiling

    Lane mapping (caller decides):
      L4  → primary=5m,  range=15m, windows=(12, 4,  12)  fast meme
      L2/3→ primary=15m, range=1h,  windows=(24, 16, 20)  swing/rotation
      L1  → primary=1h,  range=1h,  windows=(72, 24, 40)  blue-chip continuation
    """
    n = len(symbols)
    ema_9_arr = np.zeros(n, dtype=np.float64)
    ema_20_arr = np.zeros(n, dtype=np.float64)
    ema_26_arr = np.zeros(n, dtype=np.float64)
    ema9_above_ema20 = np.zeros(n, dtype=bool)
    ema9_above_ema26 = np.zeros(n, dtype=bool)
    price_above_ema20 = np.zeros(n, dtype=bool)
    ema_slope_9 = np.zeros(n, dtype=np.float64)
    ema_cross_distance_pct = np.zeros(n, dtype=np.float64)
    range_pos_1h = np.full(n, 0.5, dtype=np.float64)
    range_pos_4h = np.full(n, 0.5, dtype=np.float64)
    range_breakout_1h = np.zeros(n, dtype=bool)
    higher_low_count = np.zeros(n, dtype=np.int8)
    pivot_break = np.zeros(n, dtype=bool)
    pullback_hold = np.zeros(n, dtype=bool)

    for i, symbol in enumerate(symbols):
        frame = (primary_by_symbol or {}).get(symbol)
        if frame is None:
            frame = pd.DataFrame()
        if not frame.empty and "close" in frame.columns and len(frame) >= 21:
            closes = frame["close"].to_numpy(dtype=np.float64)
            ema9 = _ema_series(closes, 9)
            ema20 = _ema_series(closes, 20)
            ema26 = _ema_series(closes, 26)
            ema_9_arr[i] = ema9[-1]
            ema_20_arr[i] = ema20[-1]
            ema_26_arr[i] = ema26[-1]
            ema9_above_ema20[i] = bool(ema9[-1] > ema20[-1])
            ema9_above_ema26[i] = bool(ema9[-1] > ema26[-1])
            price_above_ema20[i] = bool(closes[-1] > ema20[-1])
            if len(ema9) >= 4 and ema9[-4] > 1e-10:
                ema_slope_9[i] = float((ema9[-1] - ema9[-4]) / ema9[-4])
            if ema26[-1] > 1e-10:
                ema_cross_distance_pct[i] = float((ema9[-1] - ema26[-1]) / ema26[-1])

            # Higher low count: bars in last 10 where low[i] > low[i-1]
            if "low" in frame.columns and len(frame) >= 11:
                lows = frame["low"].to_numpy(dtype=np.float64)[-11:]
                higher_low_count[i] = int(np.sum(lows[1:] > lows[:-1]))

            # Pivot break: close above max high of bars [-20:-2] (excludes last 2 noisy bars)
            if "high" in frame.columns and len(frame) >= 7:
                highs = frame["high"].to_numpy(dtype=np.float64)
                pivot_window = highs[max(-20, -len(highs)):-2]
                if pivot_window.size > 0:
                    pivot_break[i] = bool(closes[-1] > float(pivot_window.max()))

            # Pullback hold: price touched EMA9 (within 0.5%) in bars [-4:-1] then recovered
            if len(closes) >= 6:
                for j in range(-4, -1):
                    e9 = float(ema9[j])
                    if e9 > 1e-10 and abs(closes[j] - e9) / e9 < 0.005:
                        pullback_hold[i] = bool(closes[-1] > closes[-2])
                        break

        # Range features from range_by_symbol (timeframe depends on lane)
        frame_r = (range_by_symbol or {}).get(symbol)
        if frame_r is not None and not frame_r.empty and len(frame_r) >= 5:
            price_now = float(frame_r["close"].iloc[-1]) if "close" in frame_r.columns else 0.0
            has_hl = "high" in frame_r.columns and "low" in frame_r.columns

            def _range_pos(bars: pd.DataFrame, window: int) -> float:
                tail = bars.tail(window)
                h = float(tail["high"].max()) if has_hl else float(tail["close"].max())
                lo = float(tail["low"].min()) if has_hl else float(tail["close"].min())
                if h <= lo:
                    return 0.5
                return float(np.clip((price_now - lo) / (h - lo), 0.0, 1.0))

            range_pos_1h[i] = _range_pos(frame_r, range_window)
            range_pos_4h[i] = _range_pos(frame_r, range_4h_window)

            # Breakout: price above highest high of prior N bars (not counting last bar)
            if len(frame_r) >= 5:
                lookback = min(breakout_lookback, len(frame_r) - 1)
                prior = frame_r.iloc[-(lookback + 1):-1]
                prior_peak = float(prior["high"].max()) if has_hl else float(prior["close"].max())
                range_breakout_1h[i] = bool(price_now > prior_peak)

    return {
        "ema_9": ema_9_arr,
        "ema_20": ema_20_arr,
        "ema_26": ema_26_arr,
        "ema9_above_ema20": ema9_above_ema20,
        "ema9_above_ema26": ema9_above_ema26,
        "price_above_ema20": price_above_ema20,
        "ema_slope_9": ema_slope_9,
        "ema_cross_distance_pct": ema_cross_distance_pct,
        "range_pos_1h": range_pos_1h,
        "range_pos_4h": range_pos_4h,
        "range_breakout_1h": range_breakout_1h,
        "higher_low_count": higher_low_count,
        "pivot_break": pivot_break,
        "pullback_hold": pullback_hold,
    }


def _compute_price_zscore(
    prices: np.ndarray,
    *,
    middle: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
) -> np.ndarray:
    denom = (upper - lower) / 4.0
    zscore = np.zeros(prices.shape[0], dtype=np.float64)
    valid = denom > 1e-12
    zscore[valid] = (prices[:, -1][valid] - middle[valid]) / denom[valid]
    return zscore


def _compute_short_tf_features(
    ohlc_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    lookback: int = 5,
    min_bars: int = 6,
) -> tuple[list[float], list[int], list[bool]]:
    """Compute momentum and trend for a short timeframe per-symbol. Returns (momentum, trend, ready) lists."""
    momentums: list[float] = []
    trends: list[int] = []
    readiness: list[bool] = []
    for symbol in symbols:
        frame = ohlc_by_symbol.get(symbol)
        required = max(lookback + 1, min_bars)
        if frame is None or frame.empty or len(frame) < required:
            momentums.append(0.0)
            trends.append(0)
            readiness.append(False)
            continue
        closes = frame["close"].to_numpy(dtype=np.float64)
        prev = closes[-(lookback + 1)]
        latest = closes[-1]
        mom = (latest / prev - 1.0) if prev > 0.0 else 0.0
        trend = 1 if latest > closes[-lookback] else (-1 if latest < closes[-lookback] else 0)
        momentums.append(float(mom))
        trends.append(int(trend))
    return momentums, trends, readiness


def _compute_vwrs(
    frames: list[pd.DataFrame],
    btc_frame: pd.DataFrame | None,
    lookback: int = 20,
) -> np.ndarray:
    """Compute Volume-Weighted Relative Strength vs BTC over a window."""
    results: list[float] = []
    btc_rets: np.ndarray | None = None
    
    if btc_frame is not None and not btc_frame.empty and len(btc_frame) > lookback:
        btc_closes = btc_frame["close"].to_numpy(dtype=np.float64)
        btc_window = btc_closes[-(lookback + 1):]
        btc_rets = np.zeros(lookback, dtype=np.float64)
        for i in range(1, len(btc_window)):
            prev = btc_window[i - 1]
            btc_rets[i - 1] = (btc_window[i] - prev) / prev if prev > 0 else 0.0

    for frame in frames:
        if frame.empty or "close" not in frame.columns or "volume" not in frame.columns or len(frame) <= lookback:
            results.append(0.0)
            continue
            
        closes = frame["close"].to_numpy(dtype=np.float64)[-(lookback + 1):]
        volumes = frame["volume"].to_numpy(dtype=np.float64)[-(lookback):]
        
        rets = np.zeros(lookback, dtype=np.float64)
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            rets[i - 1] = (closes[i] - prev) / prev if prev > 0 else 0.0

        if btc_rets is not None and len(btc_rets) == len(rets):
            rel_rets = rets - btc_rets
        else:
            rel_rets = rets
            
        vol_sum = np.sum(volumes)
        if vol_sum <= 0:
            results.append(0.0)
        else:
            weighted_ret = np.sum(rel_rets * volumes) / vol_sum
            results.append(float(weighted_ret))
            
    return np.asarray(results, dtype=np.float64)


def _build_coin_profile(
    features_batch: dict[str, Any],
    asset_idx: int,
    *,
    lane: str,
) -> dict[str, float]:
    lane_upper = str(lane or "L3").upper()
    if lane_upper == "L1":
        struct = features_batch["struct_l1"]
    elif lane_upper == "L4":
        struct = features_batch["struct_l4"]
    else:
        struct = features_batch["struct_l23"]

    momentum_5 = float(features_batch["momentum_5"][asset_idx])
    momentum_14 = float(features_batch["momentum_14"][asset_idx])
    momentum_30 = float(features_batch["momentum_30"][asset_idx])
    volume_ratio = float(features_batch["volume_ratio"][asset_idx])
    volume_surge = float(features_batch["volume_surge"][asset_idx])
    vwio = float(features_batch["vwio"][asset_idx])
    spread_pct = float(features_batch.get("spread_pct", np.zeros(len(features_batch["symbols"]), dtype=np.float64))[asset_idx] if "spread_pct" in features_batch else 0.0)
    atr = float(features_batch["atr"][asset_idx])
    price = float(features_batch["price"][asset_idx])
    book_valid = bool(features_batch.get("book_valid", np.ones(len(features_batch["symbols"]), dtype=bool))[asset_idx] if "book_valid" in features_batch else True)
    regime_7d = str(features_batch["regime_7d"][asset_idx]).lower()
    macro_30d = str(features_batch["macro_30d"][asset_idx]).lower()
    market_regime = features_batch.get("market_regime", {})
    breadth = float((market_regime or {}).get("breadth", 0.0) or 0.0)
    market_cap_change = float((market_regime or {}).get("median", 0.0) or 0.0)
    trend_confirmed = bool(features_batch["trend_confirmed"][asset_idx])
    hurst = float(features_batch["hurst"][asset_idx])
    rsi = float(features_batch["rsi"][asset_idx])
    zscore = abs(float(features_batch["price_zscore"][asset_idx]))
    corr_row = np.asarray(features_batch["correlation"][asset_idx], dtype=np.float64)
    crowding = _mean_abs_correlation(corr_row)
    obv_divergence = int(features_batch["obv_divergence"][asset_idx])

    ema_stack_score = (
        1.0
        if bool(struct["ema9_above_ema20"][asset_idx]) and bool(struct["price_above_ema20"][asset_idx])
        else 0.5 if (bool(struct["ema9_above_ema20"][asset_idx]) or bool(struct["price_above_ema20"][asset_idx])) else 0.0
    )
    breakout_score = 1.0 if bool(struct["range_breakout_1h"][asset_idx]) else 0.0
    pivot_score = 1.0 if bool(struct["pivot_break"][asset_idx]) else 0.0
    pullback_score = 1.0 if bool(struct["pullback_hold"][asset_idx]) else 0.0
    higher_low_score = _bounded_score_0_100(float(struct["higher_low_count"][asset_idx]), 0.0, 6.0) / 100.0
    range_pos_1h = float(struct["range_pos_1h"][asset_idx])
    range_pos_4h = float(struct["range_pos_4h"][asset_idx])
    range_position_score = 1.0 - min(abs(range_pos_1h - 0.72) / 0.72, 1.0) * 0.65
    if range_pos_4h >= 0.92:
        range_position_score *= 0.6
    structure_quality = (
        ema_stack_score * 25.0
        + breakout_score * 20.0
        + pivot_score * 15.0
        + pullback_score * 20.0
        + higher_low_score * 10.0
        + _clamp01(range_position_score) * 10.0
    )

    momentum_quality = (
        _percentile_score_0_100(features_batch["momentum_5"], momentum_5) * 0.40
        + _percentile_score_0_100(features_batch["momentum_14"], momentum_14) * 0.35
        + _percentile_score_0_100(features_batch["momentum_30"], momentum_30) * 0.15
        + _bounded_score_0_100(float(features_batch["trend_1h"][asset_idx]), -1.0, 1.0) * 0.10
    )

    volume_quality = (
        _percentile_score_0_100(features_batch["volume_ratio"], volume_ratio) * 0.45
        + _percentile_score_0_100(features_batch["volume_surge"], volume_surge) * 0.35
        + _bounded_score_0_100(vwio, -1.0, 1.0) * 0.20
    )

    atr_pct = (atr / price) if price > 0.0 else 0.0
    spread_score = _inverted_bounded_score_0_100(spread_pct, 0.0, 1.5 if lane_upper != "L4" else 2.5)
    slippage_score = _inverted_bounded_score_0_100(atr_pct, 0.0, 0.08 if lane_upper != "L4" else 0.12)
    atr_efficiency = _inverted_bounded_score_0_100(atr_pct, 0.0, 0.10 if lane_upper != "L4" else 0.15)
    book_quality_score = 100.0 if book_valid else 0.0
    trade_quality = (
        spread_score * 0.40
        + slippage_score * 0.30
        + atr_efficiency * 0.20
        + book_quality_score * 0.10
    )

    breadth_score = _bounded_score_0_100(breadth, -1.0, 1.0)
    regime_score = 75.0 if regime_7d == "trending" else (30.0 if regime_7d == "choppy" else 50.0)
    macro_score = 80.0 if macro_30d == "bull" else (25.0 if macro_30d == "bear" else 50.0)
    market_cap_score = _bounded_score_0_100(market_cap_change, -0.05, 0.05)
    market_support = (
        breadth_score * 0.30
        + regime_score * 0.25
        + macro_score * 0.25
        + market_cap_score * 0.20
    )

    persistence_score = _bounded_score_0_100(hurst, 0.40, 0.70)
    trend_confirmed_score = 100.0 if trend_confirmed else 35.0
    ema_slope_score = _bounded_score_0_100(float(struct["ema_slope_9"][asset_idx]), -0.01, 0.02)
    not_overextended_score = (
        _inverted_bounded_score_0_100(zscore, 0.0, 2.4 if lane_upper != "L4" else 3.1) * 0.55
        + _inverted_bounded_score_0_100(max(rsi - (85.0 if lane_upper == "L4" else 75.0), 0.0), 0.0, 15.0) * 0.45
    )
    continuation_quality = (
        persistence_score * 0.35
        + _bounded_score_0_100(float(struct["higher_low_count"][asset_idx]), 0.0, 6.0) * 0.20
        + trend_confirmed_score * 0.20
        + ema_slope_score * 0.10
        + not_overextended_score * 0.15
    )

    exhaustion_score = (
        _inverted_bounded_score_0_100(max(rsi - (85.0 if lane_upper == "L4" else 75.0), 0.0), 0.0, 15.0) * 0.60
        + _inverted_bounded_score_0_100(max(30.0 - rsi, 0.0), 0.0, 20.0) * 0.40
    )
    zscore_score = _inverted_bounded_score_0_100(zscore, 0.0, 2.4 if lane_upper != "L4" else 3.1)
    crowding_score = _inverted_bounded_score_0_100(crowding, 0.3, 0.95)
    obv_score = 100.0 if obv_divergence == 1 else (20.0 if obv_divergence == -1 else 60.0)
    risk_quality = (
        exhaustion_score * 0.35
        + zscore_score * 0.25
        + crowding_score * 0.20
        + obv_score * 0.20
    )

    return {
        "structure_quality": round(max(0.0, min(100.0, structure_quality)), 2),
        "momentum_quality": round(max(0.0, min(100.0, momentum_quality)), 2),
        "volume_quality": round(max(0.0, min(100.0, volume_quality)), 2),
        "trade_quality": round(max(0.0, min(100.0, trade_quality)), 2),
        "market_support": round(max(0.0, min(100.0, market_support)), 2),
        "continuation_quality": round(max(0.0, min(100.0, continuation_quality)), 2),
        "risk_quality": round(max(0.0, min(100.0, risk_quality)), 2),
    }


def compute_features_batch(
    ohlc_by_symbol: dict[str, pd.DataFrame],
    *,
    ohlc_5m_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_15m_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_1h_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_7d_by_symbol: dict[str, pd.DataFrame] | None = None,
    ohlc_30d_by_symbol: dict[str, pd.DataFrame] | None = None,
    lookback_mom: int = 14,
    lookback_vol: int = 20,
    lookback_rsi: int = 14,
    lookback_atr: int = 14,
    lookback_bollinger: int = 20,
    bollinger_num_std: float = 2.0,
    finbert_scores: dict[str, float] | None = None,
    xgb_model: Any = None,
) -> dict[str, Any]:
    symbols, frames, history_points = _prepare_frames(ohlc_by_symbol)
    n_assets = len(symbols)
    if n_assets == 0:
        return {
            "symbols": [],
            "momentum": np.array([], dtype=np.float64),
            "momentum_5": np.array([], dtype=np.float64),
            "momentum_14": np.array([], dtype=np.float64),
            "momentum_30": np.array([], dtype=np.float64),
            "rotation_score": np.array([], dtype=np.float64),
            "volatility": np.array([], dtype=np.float64),
            "volume": np.array([], dtype=np.float64),
            "volume_ratio": np.array([], dtype=np.float64),
            "volume_surge": np.array([], dtype=np.float64),
            "volume_surge_flag": np.array([], dtype=bool),
            "price_zscore": np.array([], dtype=np.float64),
            "history_points": np.array([], dtype=np.int64),
            "indicators_ready": np.array([], dtype=bool),
            "feature_status": [],
            "feature_failure_reason": [],
            "rsi": np.array([], dtype=np.float64),
            "atr": np.array([], dtype=np.float64),
            "hurst": np.array([], dtype=np.float64),
            "entropy": np.array([], dtype=np.float64),
            "autocorr": np.array([], dtype=np.float64),
            "bb_middle": np.array([], dtype=np.float64),
            "bb_upper": np.array([], dtype=np.float64),
            "bb_lower": np.array([], dtype=np.float64),
            "bb_bandwidth": np.array([], dtype=np.float64),
            "price": np.array([], dtype=np.float64),
            "bar_ts": [],
            "bar_idx": [],
            "bar_interval_seconds": [],
            "correlation": np.zeros((0, 0), dtype=np.float64),
            "market_fingerprint": np.zeros(8, dtype=np.float64),
            "market_regime": {
                "r_mkt": 0.0,
                "r_btc": 0.0,
                "r_eth": 0.0,
                "breadth": 0.0,
                "median": 0.0,
                "iqr": 0.0,
                "rv_mkt": 0.0,
                "corr_avg": 0.0,
            },
            "trend_1h": np.array([], dtype=np.int64),
            "regime_7d": [],
            "macro_30d": [],
            "ma_7": np.array([], dtype=np.float64),
            "ma_26": np.array([], dtype=np.float64),
            "macd": np.array([], dtype=np.float64),
            "macd_signal": np.array([], dtype=np.float64),
            "macd_hist": np.array([], dtype=np.float64),
            "adx": np.array([], dtype=np.float64),
            "obv_divergence": np.array([], dtype=np.int8),
            "bullish_divergence": np.array([], dtype=bool),
            "bearish_divergence": np.array([], dtype=bool),
            "divergence_strength": np.array([], dtype=np.float64),
            "divergence_age_bars": np.array([], dtype=np.int8),
            "rsi_divergence": np.array([], dtype=np.int8),
            "rsi_divergence_strength": np.array([], dtype=np.float64),
            "rsi_divergence_age": np.array([], dtype=np.int8),
            "vwio": np.array([], dtype=np.float64),
            "trend_confirmed": np.array([], dtype=bool),
            "ranging_market": np.array([], dtype=bool),
            "vwrs": np.array([], dtype=np.float64),
            "momentum_5m": np.array([], dtype=np.float64),
            "trend_5m": np.array([], dtype=np.int64),
            "short_tf_ready_5m": np.array([], dtype=bool),
            "momentum_15m": np.array([], dtype=np.float64),
            "trend_15m": np.array([], dtype=np.int64),
            "short_tf_ready_15m": np.array([], dtype=bool),
            "finbert_score": np.array([], dtype=np.float64),
            "xgb_score": np.array([], dtype=np.float64),
            "struct_l1": {k: np.array([], dtype=np.float64) for k in ("ema_9","ema_20","ema_26","ema_slope_9","ema_cross_distance_pct","range_pos_1h","range_pos_4h")},
            "struct_l23": {k: np.array([], dtype=np.float64) for k in ("ema_9","ema_20","ema_26","ema_slope_9","ema_cross_distance_pct","range_pos_1h","range_pos_4h")},
            "struct_l4": {k: np.array([], dtype=np.float64) for k in ("ema_9","ema_20","ema_26","ema_slope_9","ema_cross_distance_pct","range_pos_1h","range_pos_4h")},
        }

    n_points = int(history_points.max()) if history_points.size else 0
    prices = _build_padded_matrix(frames, "close", n_points, default_value=0.0)
    highs = _build_padded_matrix(frames, "high", n_points, default_value=0.0)
    lows = _build_padded_matrix(frames, "low", n_points, default_value=0.0)
    price_latest = np.asarray(
        [float(frame["close"].iloc[-1]) if len(frame) and "close" in frame.columns else 0.0 for frame in frames],
        dtype=np.float64,
    )
    volumes = np.asarray(
        [float(frame["volume"].iloc[-1]) if len(frame) and "volume" in frame.columns else 0.0 for frame in frames],
        dtype=np.float64,
    )

    bar_ts: list[str | None] = []
    bar_idx: list[int | None] = []
    bar_interval_seconds: list[int | None] = []
    for frame in frames:
        frame_points = len(frame)
        if frame_points == 0 or "timestamp" not in frame.columns:
            bar_ts.append(None)
            bar_idx.append(None)
            bar_interval_seconds.append(None)
            continue
        last_ts = pd.Timestamp(frame["timestamp"].iloc[-1])
        bar_ts.append(last_ts.isoformat())
        if frame_points > 1:
            prev_ts = pd.Timestamp(frame["timestamp"].iloc[-2])
            interval = int((last_ts - prev_ts).total_seconds())
            bar_interval_seconds.append(interval)
            bar_idx.append(int(last_ts.timestamp() // interval) if interval > 0 else None)
        else:
            bar_interval_seconds.append(None)
            bar_idx.append(None)

    required_points = max(
        lookback_mom + 1,
        lookback_vol + 1,
        lookback_rsi + 1,
        lookback_atr + 1,
        lookback_bollinger,
    )
    indicators_ready = history_points >= required_points
    feature_status = np.where(indicators_ready, "ready", "warmup")
    feature_failure_reason = [
        "" if ready else f"insufficient_history:{int(points)}/{required_points}"
        for ready, points in zip(indicators_ready.tolist(), history_points.tolist(), strict=False)
    ]

    if n_points == 0:
        zeros = np.zeros(n_assets, dtype=np.float64)
        return {
            "symbols": symbols,
            "momentum": zeros.copy(),
            "momentum_1": zeros.copy(),
            "momentum_5": zeros.copy(),
            "momentum_14": zeros.copy(),
            "momentum_30": zeros.copy(),
            "rotation_score": zeros.copy(),
            "atr_expanding": np.zeros(n_assets, dtype=bool),
            "volatility": zeros.copy(),
            "volume": volumes.copy() if n_assets else zeros.copy(),
            "volume_ratio": np.ones(n_assets, dtype=np.float64),
            "volume_surge": zeros.copy(),
            "volume_surge_flag": np.zeros(n_assets, dtype=bool),
            "price_zscore": zeros.copy(),
            "history_points": history_points,
            "indicators_ready": np.zeros(n_assets, dtype=bool),
            "feature_status": ["missing"] * n_assets,
            "feature_failure_reason": ["missing_ohlc"] * n_assets,
            "rsi": zeros.copy(),
            "atr": zeros.copy(),
            "hurst": np.full(n_assets, 0.5, dtype=np.float64),
            "entropy": np.full(n_assets, 0.5, dtype=np.float64),
            "autocorr": zeros.copy(),
            "bb_middle": zeros.copy(),
            "bb_upper": zeros.copy(),
            "bb_lower": zeros.copy(),
            "bb_bandwidth": zeros.copy(),
            "price": price_latest,
            "bar_ts": bar_ts,
            "bar_idx": bar_idx,
            "bar_interval_seconds": bar_interval_seconds,
            "correlation": np.eye(n_assets, dtype=np.float64),
            "market_fingerprint": np.zeros(8, dtype=np.float64),
            "market_regime": {
                "r_mkt": 0.0,
                "r_btc": 0.0,
                "r_eth": 0.0,
                "breadth": 0.0,
                "median": 0.0,
                "iqr": 0.0,
                "rv_mkt": 0.0,
                "corr_avg": 0.0,
            },
            "trend_1h": np.zeros(n_assets, dtype=np.int64),
            "regime_7d": ["unknown"] * n_assets,
            "macro_30d": ["sideways"] * n_assets,
            "ma_7": price_latest.copy(),
            "ma_26": price_latest.copy(),
            "macd": zeros.copy(),
            "macd_signal": zeros.copy(),
            "macd_hist": zeros.copy(),
            "adx": zeros.copy(),
            "obv_divergence": np.zeros(n_assets, dtype=np.int8),
            "bullish_divergence": np.zeros(n_assets, dtype=bool),
            "bearish_divergence": np.zeros(n_assets, dtype=bool),
            "divergence_strength": np.zeros(n_assets, dtype=np.float64),
            "divergence_age_bars": np.full(n_assets, 99, dtype=np.int8),
            "rsi_divergence": np.zeros(n_assets, dtype=np.int8),
            "rsi_divergence_strength": np.zeros(n_assets, dtype=np.float64),
            "rsi_divergence_age": np.full(n_assets, 99, dtype=np.int8),
            "vwio": np.zeros(n_assets, dtype=np.float64),
            "trend_confirmed": np.zeros(n_assets, dtype=bool),
            "ranging_market": np.zeros(n_assets, dtype=bool),
            "vwrs": np.zeros(n_assets, dtype=np.float64),
            "momentum_5m": zeros.copy(),
            "trend_5m": np.zeros(n_assets, dtype=np.int64),
            "short_tf_ready_5m": np.zeros(n_assets, dtype=bool),
            "momentum_15m": zeros.copy(),
            "trend_15m": np.zeros(n_assets, dtype=np.int64),
            "short_tf_ready_15m": np.zeros(n_assets, dtype=bool),
            "finbert_score": zeros.copy(),
            "xgb_score": np.full(n_assets, 50.0, dtype=np.float64),
            "struct_l1": _empty_struct(n_assets, price_latest),
            "struct_l23": _empty_struct(n_assets, price_latest),
            "struct_l4": _empty_struct(n_assets, price_latest),
        }

    cfg = cuda_features.FeatureConfig()
    cfg.lookback_mom = lookback_mom
    cfg.lookback_vol = lookback_vol
    out = cuda_features.compute_features_gpu(prices.flatten().tolist(), n_assets, n_points, cfg)
    rsi = compute_rsi_gpu(prices, lookback=lookback_rsi)
    atr = compute_atr_gpu(highs, lows, prices, lookback=lookback_atr)
    northstar = compute_northstar_batch_features_gpu(prices)
    bollinger = compute_bollinger_gpu(prices, lookback=lookback_bollinger, num_std=bollinger_num_std)
    correlation = np.eye(n_assets, dtype=np.float64)
    ready_indices = np.flatnonzero(indicators_ready)
    if ready_indices.size >= 2:
        ready_correlation = compute_correlation_gpu(prices[ready_indices])
        correlation[np.ix_(ready_indices, ready_indices)] = np.asarray(ready_correlation, dtype=np.float64)
    volume_ratio = _compute_volume_ratio(frames, lookback=lookback_vol)
    volume_surge, volume_surge_flag = _compute_volume_surge(frames, lookback=lookback_vol)
    obv_divergence = _compute_obv_divergence(frames)
    bull_div, bear_div, div_strength, div_age, rsi_div_signal = compute_rsi_divergence_batch(frames)
    rsi_div_strength = div_strength
    rsi_div_age = div_age
    vwio = _compute_vwio(frames)
    price_zscore = _compute_price_zscore(
        prices,
        middle=np.asarray(bollinger["middle"], dtype=np.float64),
        upper=np.asarray(bollinger["upper"], dtype=np.float64),
        lower=np.asarray(bollinger["lower"], dtype=np.float64),
    )
    ready_symbols = [symbols[idx] for idx in ready_indices]
    btc_idx = ready_symbols.index("BTC/USD") if "BTC/USD" in ready_symbols else 0
    eth_idx = ready_symbols.index("ETH/USD") if "ETH/USD" in ready_symbols else (1 if ready_indices.size > 1 else 0)
    market_fingerprint = (
        compute_northstar_fingerprint_gpu(prices[ready_indices], btc_idx=btc_idx, eth_idx=eth_idx)
        if ready_indices.size
        else {
            "metrics": np.zeros(8, dtype=np.float64),
            "r_mkt": 0.0,
            "r_btc": 0.0,
            "r_eth": 0.0,
            "breadth": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "rv_mkt": 0.0,
            "corr_avg": 0.0,
        }
    )
    def _ordered_mapping(mapping: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
        if not mapping:
            return {}
        return {symbol: mapping[symbol] for symbol in symbols if symbol in mapping}

    trend_1h = compute_trend_1h_batch(_ordered_mapping(ohlc_1h_by_symbol) or ohlc_by_symbol)["trend_1h"]
    regime_7d = compute_regime_7d_batch(_ordered_mapping(ohlc_7d_by_symbol) or ohlc_by_symbol)["regime_7d"]
    macro_30d = compute_macro_30d_batch(_ordered_mapping(ohlc_30d_by_symbol) or ohlc_by_symbol)["macro_30d"]
    trend_state = compute_trend_state(prices, np.asarray(bollinger["bandwidth"], dtype=np.float64), highs=highs, lows=lows)
    momentum_1 = _compute_momentum(prices, 1)
    momentum_5 = _compute_momentum(prices, 5)
    momentum_14 = _compute_momentum(prices, 14)
    momentum_30 = _compute_momentum(prices, 30)
    rotation_score = _compute_rotation_score(
        momentum_1=momentum_1,
        momentum_5=momentum_5,
        momentum_14=momentum_14,
        momentum_30=momentum_30,
        volume_ratio=np.asarray(volume_ratio, dtype=np.float64),
        rsi=np.asarray(rsi, dtype=np.float64),
        trend_1h=np.asarray(trend_1h, dtype=np.int64),
        regime_7d=regime_7d,
        correlation=np.asarray(correlation, dtype=np.float64),
    )
    # VoV: is volatility expanding? Compare normalized std of last 5 bars vs prior 5 bars
    if n_points >= 12:
        recent_vol = np.std(prices[:, -5:], axis=1) / (np.mean(np.abs(prices[:, -5:]), axis=1) + 1e-10)
        prior_vol = np.std(prices[:, -10:-5], axis=1) / (np.mean(np.abs(prices[:, -10:-5]), axis=1) + 1e-10)
        atr_expanding = recent_vol > prior_vol * 1.05
        vol_expansion_ratio = recent_vol / (prior_vol + 1e-10)
    else:
        atr_expanding = np.zeros(n_assets, dtype=bool)
        vol_expansion_ratio = np.ones(n_assets, dtype=np.float64)

    if np.any(~indicators_ready):
        not_ready = ~indicators_ready
        out_momentum = np.asarray(out.momentum, dtype=np.float64)
        out_volatility = np.asarray(out.volatility, dtype=np.float64)
        rsi = np.asarray(rsi, dtype=np.float64)
        atr = np.asarray(atr, dtype=np.float64)
        hurst = np.asarray(northstar["hurst"], dtype=np.float64)
        entropy = np.asarray(northstar["entropy"], dtype=np.float64)
        autocorr = np.asarray(northstar["autocorr"], dtype=np.float64)
        bb_middle = np.asarray(bollinger["middle"], dtype=np.float64)
        bb_upper = np.asarray(bollinger["upper"], dtype=np.float64)
        bb_lower = np.asarray(bollinger["lower"], dtype=np.float64)
        bb_bandwidth = np.asarray(bollinger["bandwidth"], dtype=np.float64)
        ma_7 = np.asarray(trend_state["ma_7"], dtype=np.float64)
        ma_26 = np.asarray(trend_state["ma_26"], dtype=np.float64)
        macd = np.asarray(trend_state["macd"], dtype=np.float64)
        macd_signal = np.asarray(trend_state["macd_signal"], dtype=np.float64)
        macd_hist = np.asarray(trend_state["macd_hist"], dtype=np.float64)
        adx = np.asarray(trend_state["adx"], dtype=np.float64)
        obv_div = np.asarray(obv_divergence, dtype=np.int8)
        bull_div = np.asarray(bull_div, dtype=bool)
        bear_div = np.asarray(bear_div, dtype=bool)
        div_strength = np.asarray(div_strength, dtype=np.float64)
        div_age = np.asarray(div_age, dtype=np.int8)
        rsi_div_sig = np.asarray(rsi_div_signal, dtype=np.int8)
        rsi_div_str = np.asarray(rsi_div_strength, dtype=np.float64)
        rsi_div_ag = np.asarray(rsi_div_age, dtype=np.int8)
        vwio_arr = np.asarray(vwio, dtype=np.float64)
        trend_confirmed = np.asarray(trend_state["trend_confirmed"], dtype=bool)
        ranging_market = np.asarray(trend_state["ranging_market"], dtype=bool)

        for arr in (
            out_momentum,
            momentum_1,
            momentum_5,
            momentum_14,
            momentum_30,
            rotation_score,
            out_volatility,
            volume_surge,
            price_zscore,
            rsi,
            atr,
            autocorr,
            bb_bandwidth,
            macd,
            macd_signal,
            macd_hist,
            adx,
            vwio_arr,
            rsi_div_str,
        ):
            arr[not_ready] = 0.0
        for arr in (hurst, entropy):
            arr[not_ready] = 0.5
        for arr in (bb_middle, bb_upper, bb_lower, ma_7, ma_26):
            arr[not_ready] = price_latest[not_ready]
        atr_expanding[not_ready] = False
        obv_div[not_ready] = 0
        bull_div[not_ready] = False
        bear_div[not_ready] = False
        div_strength[not_ready] = 0.0
        div_age[not_ready] = 99
        rsi_div_sig[not_ready] = 0
        rsi_div_ag[not_ready] = 99
        trend_confirmed[not_ready] = False
        ranging_market[not_ready] = False
    else:
        out_momentum = np.asarray(out.momentum, dtype=np.float64)
        out_volatility = np.asarray(out.volatility, dtype=np.float64)
        rsi = np.asarray(rsi, dtype=np.float64)
        atr = np.asarray(atr, dtype=np.float64)
        hurst = np.asarray(northstar["hurst"], dtype=np.float64)
        entropy = np.asarray(northstar["entropy"], dtype=np.float64)
        autocorr = np.asarray(northstar["autocorr"], dtype=np.float64)
        bb_middle = np.asarray(bollinger["middle"], dtype=np.float64)
        bb_upper = np.asarray(bollinger["upper"], dtype=np.float64)
        bb_lower = np.asarray(bollinger["lower"], dtype=np.float64)
        bb_bandwidth = np.asarray(bollinger["bandwidth"], dtype=np.float64)
        ma_7 = np.asarray(trend_state["ma_7"], dtype=np.float64)
        ma_26 = np.asarray(trend_state["ma_26"], dtype=np.float64)
        macd = np.asarray(trend_state["macd"], dtype=np.float64)
        macd_signal = np.asarray(trend_state["macd_signal"], dtype=np.float64)
        macd_hist = np.asarray(trend_state["macd_hist"], dtype=np.float64)
        adx = np.asarray(trend_state["adx"], dtype=np.float64)
        obv_div = np.asarray(obv_divergence, dtype=np.int8)
        bull_div = np.asarray(bull_div, dtype=bool)
        bear_div = np.asarray(bear_div, dtype=bool)
        div_strength = np.asarray(div_strength, dtype=np.float64)
        div_age = np.asarray(div_age, dtype=np.int8)
        rsi_div_sig = np.asarray(rsi_div_signal, dtype=np.int8)
        rsi_div_str = np.asarray(rsi_div_strength, dtype=np.float64)
        rsi_div_ag = np.asarray(rsi_div_age, dtype=np.int8)
        vwio_arr = np.asarray(vwio, dtype=np.float64)
        trend_confirmed = np.asarray(trend_state["trend_confirmed"], dtype=bool)
        ranging_market = np.asarray(trend_state["ranging_market"], dtype=bool)

    atr_pct = np.where(price_latest > 0.0, atr / np.maximum(price_latest, 1e-10), 0.0)
    volatility_percentile = _cross_section_percentile_scores(out_volatility)
    bandwidth_percentile = _cross_section_percentile_scores(bb_bandwidth)
    compression_score = np.clip(
        (0.55 * _cross_section_percentile_scores(out_volatility, invert=True))
        + (0.45 * _cross_section_percentile_scores(bb_bandwidth, invert=True)),
        0.0,
        100.0,
    )
    expansion_score = np.clip(
        (0.45 * volatility_percentile)
        + (0.35 * bandwidth_percentile)
        + (0.20 * np.clip((vol_expansion_ratio - 1.0) * 100.0, 0.0, 100.0)),
        0.0,
        100.0,
    )
    volatility_state = [
        _volatility_state_from_metrics(
            atr_pct=float(atr_pct[i]),
            volatility_percentile=float(volatility_percentile[i]),
            compression_score=float(compression_score[i]),
            expansion_score=float(expansion_score[i]),
            atr_expanding=bool(atr_expanding[i]),
        )
        for i in range(n_assets)
    ]

    # Structure features per lane timeframe:
    #   L4  → 5m  primary, 15m range (fast meme:  3h full range, 1h zone)
    #   L2/3→ 15m primary, 1h  range (swing:     24h full range, 16h zone)
    #   L1  → 1h  primary, 1h  range (blue-chip:  3d full range, 24h zone)
    struct_l4 = _compute_structure_features(
        symbols, ohlc_5m_by_symbol, ohlc_15m_by_symbol,
        range_window=12, range_4h_window=4, breakout_lookback=12,
    )
    struct_l23 = _compute_structure_features(
        symbols, ohlc_15m_by_symbol, ohlc_1h_by_symbol,
        range_window=24, range_4h_window=16, breakout_lookback=20,
    )
    struct_l1 = _compute_structure_features(
        symbols, ohlc_1h_by_symbol, ohlc_1h_by_symbol,
        range_window=72, range_4h_window=24, breakout_lookback=40,
    )

    # Zero out structure features for not-ready assets
    if np.any(~indicators_ready):
        not_ready = ~indicators_ready
        for struct in (struct_l4, struct_l23, struct_l1):
            struct["ema_9"][not_ready] = price_latest[not_ready]
            struct["ema_20"][not_ready] = price_latest[not_ready]
            struct["ema_26"][not_ready] = price_latest[not_ready]
            struct["ema_slope_9"][not_ready] = 0.0
            struct["ema_cross_distance_pct"][not_ready] = 0.0
            struct["range_pos_1h"][not_ready] = 0.5
            struct["range_pos_4h"][not_ready] = 0.5
            for key in ("ema9_above_ema20", "ema9_above_ema26", "price_above_ema20", "range_breakout_1h", "pivot_break", "pullback_hold"):
                struct[key][not_ready] = False
            struct["higher_low_count"][not_ready] = 0

    short_tf_min_bars_5m = int(os.getenv("WARMUP_MIN_BARS_5M", "6"))
    short_tf_min_bars_15m = int(os.getenv("WARMUP_MIN_BARS_15M", "4"))
    # 5m/15m short timeframe features
    if ohlc_5m_by_symbol:
        momentum_5m_list, trend_5m_list, short_tf_ready_5m = _compute_short_tf_features(
            ohlc_5m_by_symbol,
            symbols,
            lookback=5,
            min_bars=short_tf_min_bars_5m,
        )
    else:
        momentum_5m_list = [0.0] * n_assets
        trend_5m_list = [0] * n_assets
        short_tf_ready_5m = [False] * n_assets

    if ohlc_15m_by_symbol:
        momentum_15m_list, trend_15m_list, short_tf_ready_15m = _compute_short_tf_features(
            ohlc_15m_by_symbol,
            symbols,
            lookback=5,
            min_bars=short_tf_min_bars_15m,
        )
    else:
        momentum_15m_list = [0.0] * n_assets
        trend_15m_list = [0] * n_assets
        short_tf_ready_15m = [False] * n_assets

    # Compute VWRS using the 15m timeframe (approx 5h given 20 bars)
    btc_frame_15m: pd.DataFrame | None = None
    if ohlc_15m_by_symbol and "BTC/USD" in ohlc_15m_by_symbol:
        btc_frame_15m = ohlc_15m_by_symbol["BTC/USD"]
    
    # We use 15m frames for VWRS if available, fallback to primary frames if not
    frames_for_vwrs = []
    for sym in symbols:
        if ohlc_15m_by_symbol and sym in ohlc_15m_by_symbol:
            frames_for_vwrs.append(ohlc_15m_by_symbol[sym])
        else:
            frames_for_vwrs.append(ohlc_by_symbol.get(sym, pd.DataFrame()))
            
    vwrs_arr = _compute_vwrs(frames_for_vwrs, btc_frame_15m, lookback=20)

    # Per-symbol finbert scores
    finbert_per_symbol = [
        float(finbert_scores.get(sym, 0.0)) if finbert_scores else 0.0
        for sym in symbols
    ]

    # Per-symbol XGBoost scores
    xgb_per_symbol: list[float] = []
    for i, sym in enumerate(symbols):
        if xgb_model is not None:
            try:
                # Build a minimal feature dict for xgb prediction
                xgb_feat: dict[str, Any] = {
                    "momentum_5": float(momentum_5[i]),
                    "momentum_14": float(momentum_14[i]),
                    "momentum_30": float(momentum_30[i]),
                    "rsi": float(np.asarray(rsi, dtype=np.float64)[i]),
                    "atr": float(np.asarray(atr, dtype=np.float64)[i]),
                    "volume_surge": float(volume_surge[i]),
                    "book_imbalance": 0.0,
                    "rotation_score": float(rotation_score[i]),
                    "hurst": float(northstar["hurst"][i]),
                    "entropy": float(northstar["entropy"][i]),
                    "trend_1h": int(np.asarray(trend_1h, dtype=np.int64)[i]),
                    "finbert_score": finbert_per_symbol[i],
                    "autocorr": float(northstar["autocorr"][i]),
                    "price_zscore": float(price_zscore[i]),
                    "volume_ratio": float(volume_ratio[i]),
                }
                xgb_per_symbol.append(float(xgb_model.predict(xgb_feat)))
            except Exception:
                xgb_per_symbol.append(50.0)
        else:
            xgb_per_symbol.append(50.0)

    return {
        "symbols": symbols,
        "momentum": out_momentum,
        "momentum_1": momentum_1,
        "momentum_5": momentum_5,
        "momentum_14": momentum_14,
        "momentum_30": momentum_30,
        "rotation_score": rotation_score,
        "atr_expanding": atr_expanding,
        "volatility": out_volatility,
        "volume": volumes,
        "volume_ratio": volume_ratio,
        "volume_surge": volume_surge,
        "volume_surge_flag": volume_surge_flag,
        "price_zscore": price_zscore,
        "history_points": history_points,
        "indicators_ready": indicators_ready,
        "feature_status": feature_status.tolist(),
        "feature_failure_reason": feature_failure_reason,
        "rsi": rsi,
        "atr": atr,
        "atr_pct": atr_pct,
        "hurst": hurst,
        "entropy": entropy,
        "autocorr": autocorr,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_bandwidth": bb_bandwidth,
        "price": price_latest,
        "bar_ts": bar_ts,
        "bar_idx": bar_idx,
        "bar_interval_seconds": bar_interval_seconds,
        "correlation": correlation,
        "market_fingerprint": np.asarray(market_fingerprint["metrics"], dtype=np.float64),
        "market_regime": market_fingerprint,
        "trend_1h": np.asarray(trend_1h, dtype=np.int64),
        "regime_7d": regime_7d,
        "macro_30d": macro_30d,
        "ma_7": ma_7,
        "ma_26": ma_26,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "adx": adx,
        "obv_divergence": obv_div,
        "bullish_divergence": bull_div,
        "bearish_divergence": bear_div,
        "divergence_strength": div_strength,
        "divergence_age_bars": div_age,
        "rsi_divergence": rsi_div_sig,
        "rsi_divergence_strength": rsi_div_str,
        "rsi_divergence_age": rsi_div_ag,
        "vwio": vwio_arr,
        "trend_confirmed": trend_confirmed,
        "ranging_market": ranging_market,
        "volatility_percentile": volatility_percentile,
        "compression_score": compression_score,
        "expansion_score": expansion_score,
        "volatility_state": volatility_state,
        "vwrs": vwrs_arr,
        "momentum_5m": np.asarray(momentum_5m_list, dtype=np.float64),
        "trend_5m": np.asarray(trend_5m_list, dtype=np.int64),
        "short_tf_ready_5m": np.asarray(short_tf_ready_5m, dtype=bool),
        "momentum_15m": np.asarray(momentum_15m_list, dtype=np.float64),
        "trend_15m": np.asarray(trend_15m_list, dtype=np.int64),
        "short_tf_ready_15m": np.asarray(short_tf_ready_15m, dtype=bool),
        "finbert_score": np.asarray(finbert_per_symbol, dtype=np.float64),
        "xgb_score": np.asarray(xgb_per_symbol, dtype=np.float64),
        "struct_l1": struct_l1,
        "struct_l23": struct_l23,
        "struct_l4": struct_l4,
    }


def slice_features_for_asset(features_batch: dict[str, Any], asset_idx: int, *, lane_hint: str | None = None) -> dict[str, Any]:
    def _safe_batch_scalar(key: str, default: Any) -> Any:
        values = features_batch.get(key)
        if values is None:
            return default
        try:
            if len(values) <= asset_idx:
                return default
        except TypeError:
            return default
        try:
            return values[asset_idx]
        except Exception:
            return default

    symbol = features_batch["symbols"][asset_idx]
    resolved_lane = str(lane_hint or "").upper()
    features = {
        "symbol": symbol,
        "momentum": float(features_batch["momentum"][asset_idx]),
        "momentum_1": float(features_batch["momentum_1"][asset_idx]),
        "momentum_5": float(features_batch["momentum_5"][asset_idx]),
        "momentum_14": float(features_batch["momentum_14"][asset_idx]),
        "momentum_30": float(features_batch["momentum_30"][asset_idx]),
        "rotation_score": float(features_batch["rotation_score"][asset_idx]),
        "atr_expanding": bool(features_batch["atr_expanding"][asset_idx]),
        "volatility": float(features_batch["volatility"][asset_idx]),
        "volume": float(features_batch["volume"][asset_idx]),
        "volume_ratio": float(features_batch["volume_ratio"][asset_idx]),
        "volume_surge": float(features_batch["volume_surge"][asset_idx]),
        "volume_surge_flag": bool(features_batch["volume_surge_flag"][asset_idx]),
        "price_zscore": float(features_batch["price_zscore"][asset_idx]),
        "history_points": int(features_batch["history_points"][asset_idx]),
        "indicators_ready": bool(features_batch["indicators_ready"][asset_idx]),
        "feature_status": str(features_batch["feature_status"][asset_idx]),
        "feature_failure_reason": str(features_batch["feature_failure_reason"][asset_idx]),
        "rsi": float(features_batch["rsi"][asset_idx]),
        "atr": float(features_batch["atr"][asset_idx]),
        "atr_pct": float(features_batch["atr_pct"][asset_idx]),
        "hurst": float(features_batch["hurst"][asset_idx]),
        "entropy": float(features_batch["entropy"][asset_idx]),
        "autocorr": float(features_batch["autocorr"][asset_idx]),
        "bb_middle": float(features_batch["bb_middle"][asset_idx]),
        "bb_upper": float(features_batch["bb_upper"][asset_idx]),
        "bb_lower": float(features_batch["bb_lower"][asset_idx]),
        "bb_bandwidth": float(features_batch["bb_bandwidth"][asset_idx]),
        "price": float(features_batch["price"][asset_idx]),
        "bar_ts": features_batch["bar_ts"][asset_idx],
        "bar_idx": features_batch["bar_idx"][asset_idx],
        "bar_interval_seconds": features_batch["bar_interval_seconds"][asset_idx],
        "correlation_row": features_batch["correlation"][asset_idx],
        "correlation_symbols": features_batch["symbols"],
        "market_fingerprint": features_batch["market_fingerprint"],
        "market_regime": features_batch["market_regime"],
        "trend_1h": int(features_batch["trend_1h"][asset_idx]),
        "regime_7d": features_batch["regime_7d"][asset_idx],
        "macro_30d": features_batch["macro_30d"][asset_idx],
        "ma_7": float(features_batch["ma_7"][asset_idx]),
        "ma_26": float(features_batch["ma_26"][asset_idx]),
        "macd": float(features_batch["macd"][asset_idx]),
        "macd_signal": float(features_batch["macd_signal"][asset_idx]),
        "macd_hist": float(features_batch["macd_hist"][asset_idx]),
        "adx": float(features_batch["adx"][asset_idx]),
        "obv_divergence": int(features_batch["obv_divergence"][asset_idx]),
        "bullish_divergence": bool(features_batch["bullish_divergence"][asset_idx]),
        "bearish_divergence": bool(features_batch["bearish_divergence"][asset_idx]),
        "divergence_strength": float(features_batch["divergence_strength"][asset_idx]),
        "divergence_age_bars": int(features_batch["divergence_age_bars"][asset_idx]),
        "rsi_divergence": int(features_batch["rsi_divergence"][asset_idx]),
        "rsi_divergence_strength": float(features_batch["rsi_divergence_strength"][asset_idx]),
        "rsi_divergence_age": int(features_batch["rsi_divergence_age"][asset_idx]),
        "vwio": float(features_batch["vwio"][asset_idx]),
        "trend_confirmed": bool(features_batch["trend_confirmed"][asset_idx]),
        "ranging_market": bool(features_batch["ranging_market"][asset_idx]),
        "volatility_percentile": float(features_batch["volatility_percentile"][asset_idx]),
        "expansion_score": float(features_batch["expansion_score"][asset_idx]),
        "volatility_state": str(features_batch["volatility_state"][asset_idx]),
        "vwrs": float(features_batch["vwrs"][asset_idx]),
        "momentum_5m": float(_safe_batch_scalar("momentum_5m", 0.0)),
        "trend_5m": int(_safe_batch_scalar("trend_5m", 0)),
        "short_tf_ready_5m": bool(_safe_batch_scalar("short_tf_ready_5m", False)),
        "momentum_15m": float(_safe_batch_scalar("momentum_15m", 0.0)),
        "trend_15m": int(_safe_batch_scalar("trend_15m", 0)),
        "short_tf_ready_15m": bool(_safe_batch_scalar("short_tf_ready_15m", False)),
        "finbert_score": float(features_batch["finbert_score"][asset_idx]),
        "xgb_score": float(features_batch["xgb_score"][asset_idx]),
    }
    # Pick lane-appropriate structure features
    _lane = str(lane_hint or "").upper()
    if _lane == "L1":
        _struct = features_batch["struct_l1"]
    elif _lane == "L4":
        _struct = features_batch["struct_l4"]
    else:  # L2, L3, default
        _struct = features_batch["struct_l23"]
    features.update({
        "ema_9": float(_struct["ema_9"][asset_idx]),
        "ema_20": float(_struct["ema_20"][asset_idx]),
        "ema_26": float(_struct["ema_26"][asset_idx]),
        "ema9_above_ema20": bool(_struct["ema9_above_ema20"][asset_idx]),
        "ema9_above_ema26": bool(_struct["ema9_above_ema26"][asset_idx]),
        "price_above_ema20": bool(_struct["price_above_ema20"][asset_idx]),
        "ema_slope_9": float(_struct["ema_slope_9"][asset_idx]),
        "ema_cross_distance_pct": float(_struct["ema_cross_distance_pct"][asset_idx]),
        "range_pos_1h": float(_struct["range_pos_1h"][asset_idx]),
        "range_pos_4h": float(_struct["range_pos_4h"][asset_idx]),
        "range_breakout_1h": bool(_struct["range_breakout_1h"][asset_idx]),
        "higher_low_count": int(_struct["higher_low_count"][asset_idx]),
        "pivot_break": bool(_struct["pivot_break"][asset_idx]),
        "pullback_hold": bool(_struct["pullback_hold"][asset_idx]),
    })
    structure_build = bool(
        features["ema9_above_ema26"]
        and features["price_above_ema20"]
        and (
            features["pullback_hold"]
            or features["pivot_break"]
            or features["higher_low_count"] >= 3
        )
    )
    breakout_confirmed = bool(
        features["range_breakout_1h"]
        and (float(features.get("volume_ratio", 1.0) or 1.0) >= 1.15 or float(features.get("volume_surge", 0.0) or 0.0) >= 0.15)
        and bool(features["ema9_above_ema26"])
    )
    retest_confirmed = bool(
        features["pullback_hold"]
        and bool(features["ema9_above_ema26"])
        and int(features["higher_low_count"]) >= 2
    )
    overextended = bool(
        float(features["range_pos_4h"]) >= 0.92
        or abs(float(features.get("price_zscore", 0.0) or 0.0)) >= 2.2
        or float(features.get("rsi", 50.0) or 50.0) >= 74.0
    )
    breakout_failure_risk = bool(
        bool(features["range_breakout_1h"])
        and not bool(features["ema9_above_ema26"])
        and float(features["range_pos_1h"]) <= 0.55
    )
    structure_break_risk = bool(
        not bool(features["ema9_above_ema26"])
        and (
            int(features["higher_low_count"]) <= 1
            or float(features["range_pos_1h"]) <= 0.42
            or float(features["ema_slope_9"]) < -0.002
        )
    )
    features.update({
        "structure_build": structure_build,
        "breakout_confirmed": breakout_confirmed,
        "retest_confirmed": retest_confirmed,
        "overextended": overextended,
        "breakout_failure_risk": breakout_failure_risk,
        "structure_break_risk": structure_break_risk,
        "governing_timeframe": "1h" if _lane in {"L1", "L2"} else "15m",
        "macro_timeframe": "4h",
        "trigger_timeframe": "5m" if _lane == "L4" else "15m",
    })
    # Inject universe-assigned lane as universe_lane so it doesn't suppress runtime classification
    # pipeline.py: enriched["lane"] = enriched.get("lane") or classify_lane(...)
    if lane_hint:
        features["universe_lane"] = lane_hint.upper()
        resolved_lane = lane_hint.upper()
    else:
        resolved_lane = str(features.get("lane", "L3") or "L3").upper()
    coin_profile = _build_coin_profile(features_batch, asset_idx, lane=resolved_lane)
    features["coin_profile"] = coin_profile
    features.update(coin_profile)
    return apply_policy_pipeline(symbol, features)
