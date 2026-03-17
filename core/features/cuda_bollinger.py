from __future__ import annotations

import numpy as np

import cuda_bollinger


def compute_bollinger_gpu(prices: np.ndarray, lookback: int = 20, num_std: float = 2.0) -> dict[str, np.ndarray]:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    n_assets, n_points = arr.shape
    out = cuda_bollinger.compute_bollinger_gpu(
        arr.flatten().tolist(),
        int(n_assets),
        int(n_points),
        int(lookback),
        float(num_std),
    )
    return {
        "middle": np.asarray(out.middle, dtype=np.float64),
        "upper": np.asarray(out.upper, dtype=np.float64),
        "lower": np.asarray(out.lower, dtype=np.float64),
        "bandwidth": np.asarray(out.bandwidth, dtype=np.float64),
    }
