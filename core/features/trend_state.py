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


def _compute_adx(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute ADX for each asset row. Returns the ADX value (0-100).

    ADX > 25 = trend in place. ADX < 20 = no trend (ranging).
    """
    n_assets, n_points = prices.shape
    adx_out = np.zeros(n_assets, dtype=np.float64)
    if n_points < period * 2 + 1:
        return adx_out

    for idx in range(n_assets):
        close = prices[idx]
        high = highs[idx]
        low = lows[idx]

        tr_arr = np.zeros(n_points - 1, dtype=np.float64)
        pdm_arr = np.zeros(n_points - 1, dtype=np.float64)
        ndm_arr = np.zeros(n_points - 1, dtype=np.float64)

        for i in range(1, n_points):
            h, l, c_prev = float(high[i]), float(low[i]), float(close[i - 1])
            h_prev, l_prev = float(high[i - 1]), float(low[i - 1])
            tr_arr[i - 1] = max(h - l, abs(h - c_prev), abs(l - c_prev))
            up_move = h - h_prev
            down_move = l_prev - l
            pdm_arr[i - 1] = up_move if (up_move > down_move and up_move > 0) else 0.0
            ndm_arr[i - 1] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder smoothing (RMA): initial = sum of first `period` values
        atr_s = float(np.sum(tr_arr[:period]))
        pdm_s = float(np.sum(pdm_arr[:period]))
        ndm_s = float(np.sum(ndm_arr[:period]))

        dx_list: list[float] = []
        for i in range(period, len(tr_arr)):
            atr_s = atr_s - atr_s / period + tr_arr[i]
            pdm_s = pdm_s - pdm_s / period + pdm_arr[i]
            ndm_s = ndm_s - ndm_s / period + ndm_arr[i]
            if atr_s < 1e-12:
                continue
            pdi = 100.0 * pdm_s / atr_s
            ndi = 100.0 * ndm_s / atr_s
            denom = pdi + ndi
            dx = 100.0 * abs(pdi - ndi) / denom if denom > 1e-12 else 0.0
            dx_list.append(dx)

        if len(dx_list) >= period:
            # Wilder smooth DX into ADX
            adx_val = float(np.mean(dx_list[:period]))
            for dx in dx_list[period:]:
                adx_val = (adx_val * (period - 1) + dx) / period
            adx_out[idx] = round(adx_val, 2)

    return adx_out


def compute_trend_state(
    prices: np.ndarray,
    bb_bandwidth: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
) -> dict[str, Any]:
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
            "adx": empty_f,
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

    if highs is not None and lows is not None:
        highs_matrix = np.asarray(highs, dtype=np.float64)
        lows_matrix = np.asarray(lows, dtype=np.float64)
        if highs_matrix.shape == price_matrix.shape and lows_matrix.shape == price_matrix.shape:
            adx_vals = _compute_adx(price_matrix, highs_matrix, lows_matrix)
        else:
            adx_vals = np.zeros(n_assets, dtype=np.float64)
    else:
        adx_vals = np.zeros(n_assets, dtype=np.float64)

    return {
        "ma_7": ma_7,
        "ma_26": ma_26,
        "macd": macd_vals,
        "macd_signal": macd_signal_vals,
        "macd_hist": macd_hist_vals,
        "adx": adx_vals,
        "trend_confirmed": trend_confirmed,
        "ranging_market": ranging_market,
    }
