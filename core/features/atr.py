from __future__ import annotations

import numpy as np


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, lookback: int = 14) -> np.ndarray:
    high_arr = np.asarray(highs, dtype=np.float64)
    low_arr = np.asarray(lows, dtype=np.float64)
    close_arr = np.asarray(closes, dtype=np.float64)
    if high_arr.ndim != 2 or low_arr.ndim != 2 or close_arr.ndim != 2:
        raise ValueError("inputs must have shape [n_assets x n_points]")
    n_assets, n_points = close_arr.shape
    out = np.zeros(n_assets, dtype=np.float64)
    if n_points <= lookback:
        return out

    prev_close = close_arr[:, :-1]
    tr = np.maximum.reduce(
        [
            high_arr[:, 1:] - low_arr[:, 1:],
            np.abs(high_arr[:, 1:] - prev_close),
            np.abs(low_arr[:, 1:] - prev_close),
        ]
    )
    out = tr[:, -lookback:].mean(axis=1)
    return out
