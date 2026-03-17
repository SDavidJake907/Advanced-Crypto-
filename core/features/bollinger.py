from __future__ import annotations

import numpy as np


def compute_bollinger(prices: np.ndarray, lookback: int = 20, num_std: float = 2.0) -> dict[str, np.ndarray]:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    n_assets, n_points = arr.shape
    zeros = np.zeros(n_assets, dtype=np.float64)
    if n_points < lookback:
        return {"middle": zeros, "upper": zeros, "lower": zeros, "bandwidth": zeros}

    window = arr[:, -lookback:]
    middle = window.mean(axis=1)
    std = window.std(axis=1)
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    bandwidth = np.divide(upper - lower, middle, out=np.zeros_like(middle), where=middle != 0.0)
    return {"middle": middle, "upper": upper, "lower": lower, "bandwidth": bandwidth}
