from __future__ import annotations

import numpy as np
import pandas as pd


def compute_macro_30d_batch(ohlc_by_symbol: dict[str, pd.DataFrame], lookback: int = 4) -> dict[str, list[str]]:
    symbols = list(ohlc_by_symbol.keys())
    out: list[str] = []
    for symbol in symbols:
        frame = ohlc_by_symbol[symbol]
        if len(frame) < lookback + 1:
            out.append("sideways")
            continue
        closes = frame["close"].to_numpy(dtype=np.float64)
        ret = (closes[-1] / closes[-(lookback + 1)]) - 1.0
        if ret > 0.05:
            out.append("bull")
        elif ret < -0.05:
            out.append("bear")
        else:
            out.append("sideways")
    return {"symbols": symbols, "macro_30d": out}
