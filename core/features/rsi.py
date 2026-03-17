from __future__ import annotations

import numpy as np


def compute_rsi(prices: np.ndarray, lookback: int = 14) -> np.ndarray:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    n_assets, n_points = arr.shape
    out = np.zeros(n_assets, dtype=np.float64)
    if n_points <= lookback:
        return out

    deltas = np.diff(arr, axis=1)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    avg_gain = gains[:, -lookback:].mean(axis=1)
    avg_loss = losses[:, -lookback:].mean(axis=1)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 0.0)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[avg_loss == 0.0] = 100.0
    return out
