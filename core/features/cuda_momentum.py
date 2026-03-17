from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.cuda_atr import compute_atr_gpu
from core.features.cuda_bollinger import compute_bollinger_gpu
from core.features.cuda_rsi import compute_rsi_gpu
import cuda_features


class CudaMomentumVolFeatureEngine:
    def __init__(
        self,
        lookback_mom: int = 14,
        lookback_vol: int = 20,
        lookback_rsi: int = 14,
        lookback_atr: int = 14,
        lookback_bollinger: int = 20,
        bollinger_num_std: float = 2.0,
    ):
        self.lookback_mom = lookback_mom
        self.lookback_vol = lookback_vol
        self.lookback_rsi = lookback_rsi
        self.lookback_atr = lookback_atr
        self.lookback_bollinger = lookback_bollinger
        self.bollinger_num_std = bollinger_num_std

    def compute(self, ohlc: pd.DataFrame) -> dict:
        close = ohlc["close"].to_numpy(dtype=np.float64)
        high = ohlc["high"].to_numpy(dtype=np.float64)
        low = ohlc["low"].to_numpy(dtype=np.float64)
        n_points = close.shape[0]
        bar_ts = None
        bar_interval_seconds = None
        bar_idx = None
        if "timestamp" in ohlc.columns and n_points > 0:
            last_ts = pd.Timestamp(ohlc["timestamp"].iloc[-1])
            bar_ts = last_ts.isoformat()
            if n_points > 1:
                prev_ts = pd.Timestamp(ohlc["timestamp"].iloc[-2])
                bar_interval_seconds = int((last_ts - prev_ts).total_seconds())
                if bar_interval_seconds > 0:
                    bar_idx = int(last_ts.timestamp() // bar_interval_seconds)
        if n_points == 0:
            return {
                "momentum": 0.0,
                "volatility": 0.0,
                "rsi": 0.0,
                "atr": 0.0,
                "bb_middle": 0.0,
                "bb_upper": 0.0,
                "bb_lower": 0.0,
                "bb_bandwidth": 0.0,
                "price": 0.0,
                "bar_ts": None,
                "bar_idx": None,
            }
        if n_points <= max(
            self.lookback_mom + 1,
            self.lookback_vol + 1,
            self.lookback_rsi + 1,
            self.lookback_atr + 1,
            self.lookback_bollinger,
        ):
            return {
                "momentum": 0.0,
                "volatility": 0.0,
                "rsi": 0.0,
                "atr": 0.0,
                "bb_middle": 0.0,
                "bb_upper": 0.0,
                "bb_lower": 0.0,
                "bb_bandwidth": 0.0,
                "price": float(close[-1]),
                "bar_ts": bar_ts,
                "bar_idx": bar_idx,
                "bar_interval_seconds": bar_interval_seconds,
            }

        prices = close.reshape(1, -1)
        highs = high.reshape(1, -1)
        lows = low.reshape(1, -1)
        cfg = cuda_features.FeatureConfig()
        cfg.lookback_mom = self.lookback_mom
        cfg.lookback_vol = self.lookback_vol

        out = cuda_features.compute_features_gpu(
            prices.flatten().tolist(),
            1,
            int(n_points),
            cfg,
        )
        rsi = compute_rsi_gpu(prices, lookback=self.lookback_rsi)
        atr = compute_atr_gpu(highs, lows, prices, lookback=self.lookback_atr)
        bollinger = compute_bollinger_gpu(
            prices,
            lookback=self.lookback_bollinger,
            num_std=self.bollinger_num_std,
        )

        return {
            "momentum": float(out.momentum[0]),
            "volatility": float(out.volatility[0]),
            "rsi": float(rsi[0]),
            "atr": float(atr[0]),
            "bb_middle": float(bollinger["middle"][0]),
            "bb_upper": float(bollinger["upper"][0]),
            "bb_lower": float(bollinger["lower"][0]),
            "bb_bandwidth": float(bollinger["bandwidth"][0]),
            "price": float(close[-1]),
            "bar_ts": bar_ts,
            "bar_idx": bar_idx,
            "bar_interval_seconds": bar_interval_seconds,
        }
