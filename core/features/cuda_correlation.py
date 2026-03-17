from __future__ import annotations

import numpy as np

import cuda_correlation


def compute_correlation_gpu(prices: np.ndarray) -> np.ndarray:
    """
    prices: [n_assets x n_points]
    returns correlation: [n_assets x n_assets]
    """
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")

    n_assets, n_points = arr.shape
    out = cuda_correlation.compute_correlation_gpu(
        arr.flatten().tolist(),
        int(n_assets),
        int(n_points),
    )
    return np.asarray(out, dtype=np.float64).reshape(n_assets, n_assets)
