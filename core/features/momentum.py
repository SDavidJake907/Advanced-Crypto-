from __future__ import annotations

import math

from pandas import DataFrame

from core.features.base import FeatureEngine


def risk_adjusted_momentum(closes: list[float]) -> float:
    # closes: oldest -> newest
    if len(closes) < 10:
        return float("-inf")
    if closes[0] <= 0:
        return float("-inf")
    ret = (closes[-1] / closes[0]) - 1.0
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))
    if len(rets) < 2:
        return float("-inf")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = math.sqrt(max(var, 1e-12))
    return ret / vol


class MomentumFeatureEngine(FeatureEngine):
    def __init__(self, lookback: int = 14) -> None:
        self.lookback = lookback

    def compute(self, ohlc: DataFrame) -> dict:
        if ohlc.empty:
            return {"momentum": 0.0, "price": 0.0}
        if len(ohlc) < self.lookback + 1:
            return {"momentum": 0.0, "price": float(ohlc["close"].iloc[-1])}

        close = ohlc["close"]
        mom = close.iloc[-1] / close.iloc[-self.lookback - 1] - 1.0
        return {"momentum": float(mom), "price": float(close.iloc[-1])}
