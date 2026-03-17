from __future__ import annotations

import numpy as np

import cuda_rsi


def compute_rsi_gpu(prices: np.ndarray, lookback: int = 14) -> np.ndarray:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    n_assets, n_points = arr.shape
    out = cuda_rsi.compute_rsi_gpu(arr.flatten().tolist(), int(n_assets), int(n_points), int(lookback))
    return np.asarray(out, dtype=np.float64)
