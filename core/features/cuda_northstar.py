from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import cuda_northstar
    CUDA_NORTHSTAR_AVAILABLE = True
except Exception:
    cuda_northstar = None
    CUDA_NORTHSTAR_AVAILABLE = False
    if os.getenv("CUDA_NORTHSTAR_WARN", "true").lower() == "true":
        print("WARNING: cuda_northstar unavailable, using fallback features")


def compute_northstar_batch_features_gpu(prices: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    n_assets, n_points = arr.shape
    if not CUDA_NORTHSTAR_AVAILABLE:
        return {
            "hurst": np.full(n_assets, 0.5, dtype=np.float64),
            "entropy": np.full(n_assets, 0.5, dtype=np.float64),
            "autocorr": np.zeros(n_assets, dtype=np.float64),
        }
    out = cuda_northstar.compute_northstar_batch_features_gpu(
        arr.flatten().tolist(),
        int(n_assets),
        int(n_points),
    )
    return {
        "hurst": np.asarray(out.hurst, dtype=np.float64),
        "entropy": np.asarray(out.entropy, dtype=np.float64),
        "autocorr": np.asarray(out.autocorr, dtype=np.float64),
    }


def compute_northstar_fingerprint_gpu(
    prices: np.ndarray,
    *,
    btc_idx: int,
    eth_idx: int,
) -> dict[str, Any]:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prices must have shape [n_assets x n_points]")
    n_assets, n_points = arr.shape
    if not CUDA_NORTHSTAR_AVAILABLE:
        metrics = np.zeros(8, dtype=np.float64)
        return {
            "metrics": metrics,
            "r_mkt": 0.0,
            "r_btc": 0.0,
            "r_eth": 0.0,
            "breadth": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "rv_mkt": 0.0,
            "corr_avg": 0.0,
        }
    out = cuda_northstar.compute_northstar_fingerprint_gpu(
        arr.flatten().tolist(),
        int(n_assets),
        int(n_points),
        int(btc_idx),
        int(eth_idx),
    )
    metrics = np.asarray(out.metrics, dtype=np.float64)
    return {
        "metrics": metrics,
        "r_mkt": float(metrics[0]) if metrics.size > 0 else 0.0,
        "r_btc": float(metrics[1]) if metrics.size > 1 else 0.0,
        "r_eth": float(metrics[2]) if metrics.size > 2 else 0.0,
        "breadth": float(metrics[3]) if metrics.size > 3 else 0.0,
        "median": float(metrics[4]) if metrics.size > 4 else 0.0,
        "iqr": float(metrics[5]) if metrics.size > 5 else 0.0,
        "rv_mkt": float(metrics[6]) if metrics.size > 6 else 0.0,
        "corr_avg": float(metrics[7]) if metrics.size > 7 else 0.0,
    }
