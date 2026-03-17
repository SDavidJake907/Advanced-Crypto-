from __future__ import annotations

import numpy as np


def compute_correlation(prices: np.ndarray) -> np.ndarray:
    """
    prices: [n_assets x n_points]
    returns correlation: [n_assets x n_assets]
    """
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    if arr.shape[1] < 2:
        return np.eye(arr.shape[0], dtype=np.float64)
    return np.corrcoef(arr)
