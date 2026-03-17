from __future__ import annotations

from typing import Any

import numpy as np


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, values.size):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def compute_trend_state(prices: np.ndarray, bb_bandwidth: np.ndarray) -> dict[str, Any]:
    price_matrix = np.asarray(prices, dtype=np.float64)
    n_assets, n_points = price_matrix.shape
    if n_assets == 0 or n_points == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_b = np.array([], dtype=bool)
        return {
            "ma_7": empty_f,
            "ma_26": empty_f,
            "macd": empty_f,
            "macd_signal": empty_f,
            "macd_hist": empty_f,
            "trend_confirmed": empty_b,
            "ranging_market": empty_b,
        }

    ma_7 = np.mean(price_matrix[:, -7:], axis=1) if n_points >= 7 else np.mean(price_matrix, axis=1)
    ma_26 = np.mean(price_matrix[:, -26:], axis=1) if n_points >= 26 else np.mean(price_matrix, axis=1)

    macd_vals = np.zeros(n_assets, dtype=np.float64)
    macd_signal_vals = np.zeros(n_assets, dtype=np.float64)
    macd_hist_vals = np.zeros(n_assets, dtype=np.float64)
    trend_confirmed = np.zeros(n_assets, dtype=bool)
    ranging_market = np.zeros(n_assets, dtype=bool)

    bandwidth = np.asarray(bb_bandwidth, dtype=np.float64)

    for idx in range(n_assets):
        close = price_matrix[idx]
        ema_12 = _ema(close, 12)
        ema_26 = _ema(close, 26)
        macd_line = ema_12 - ema_26
        signal_line = _ema(macd_line, 9)
        macd = float(macd_line[-1])
        macd_signal = float(signal_line[-1])
        macd_hist = macd - macd_signal
        macd_vals[idx] = macd
        macd_signal_vals[idx] = macd_signal
        macd_hist_vals[idx] = macd_hist

        price = float(close[-1])
        ma_gap_pct = abs(float(ma_7[idx] - ma_26[idx])) / price if price > 0 else 0.0
        trend_confirmed[idx] = bool(
            ma_7[idx] > ma_26[idx]
            and macd > macd_signal
            and macd_hist > 0.0
        )
        ranging_market[idx] = bool(
            ma_gap_pct <= 0.0035
            and abs(macd_hist) <= max(price * 0.0008, 1e-8)
            and float(bandwidth[idx]) <= 0.03
        )

    return {
        "ma_7": ma_7,
        "ma_26": ma_26,
        "macd": macd_vals,
        "macd_signal": macd_signal_vals,
        "macd_hist": macd_hist_vals,
        "trend_confirmed": trend_confirmed,
        "ranging_market": ranging_market,
    }
