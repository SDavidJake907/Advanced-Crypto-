from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_1h_batch(ohlc_by_symbol: dict[str, pd.DataFrame], lookback: int = 8) -> dict[str, np.ndarray]:
    symbols = list(ohlc_by_symbol.keys())
    out = np.zeros(len(symbols), dtype=np.int64)
    for idx, symbol in enumerate(symbols):
        frame = ohlc_by_symbol[symbol]
        if len(frame) < lookback + 1:
            continue
        closes = frame["close"].to_numpy(dtype=np.float64)
        ret = (closes[-1] / closes[-(lookback + 1)]) - 1.0
        out[idx] = 1 if ret > 0.0 else (-1 if ret < 0.0 else 0)
    return {"symbols": symbols, "trend_1h": out}
