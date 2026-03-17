from __future__ import annotations

import numpy as np

import cuda_atr


def compute_atr_gpu(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, lookback: int = 14) -> np.ndarray:
    high_arr = np.asarray(highs, dtype=np.float64)
    low_arr = np.asarray(lows, dtype=np.float64)
    close_arr = np.asarray(closes, dtype=np.float64)
    if high_arr.ndim != 2 or low_arr.ndim != 2 or close_arr.ndim != 2:
        raise ValueError("inputs must have shape [n_assets x n_points]")
    n_assets, n_points = close_arr.shape
    out = cuda_atr.compute_atr_gpu(
        high_arr.flatten().tolist(),
        low_arr.flatten().tolist(),
        close_arr.flatten().tolist(),
        int(n_assets),
        int(n_points),
        int(lookback),
    )
    return np.asarray(out, dtype=np.float64)
