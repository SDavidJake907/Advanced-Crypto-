from __future__ import annotations

from abc import ABC, abstractmethod

from pandas import DataFrame


class FeatureEngine(ABC):
    @abstractmethod
    def compute(self, ohlc: DataFrame) -> dict:
        ...

