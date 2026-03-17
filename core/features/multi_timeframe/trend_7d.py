from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_7d_batch(ohlc_by_symbol: dict[str, pd.DataFrame], lookback: int = 6) -> dict[str, list[str]]:
    symbols = list(ohlc_by_symbol.keys())
    out: list[str] = []
    for symbol in symbols:
        frame = ohlc_by_symbol[symbol]
        if len(frame) < lookback + 1:
            out.append("unknown")
            continue
        closes = frame["close"].to_numpy(dtype=np.float64)
        recent = closes[-(lookback + 1) :]
        momentum = abs((recent[-1] / recent[0]) - 1.0)
        noise = np.std(np.diff(recent)) if len(recent) > 2 else 0.0
        out.append("trending" if momentum > 0.03 and noise < recent[-1] * 0.03 else "choppy")
    return {"symbols": symbols, "regime_7d": out}
