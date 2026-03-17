from __future__ import annotations

from collections import deque
from typing import Deque


class SignalSmoother:
    def __init__(self, window_bars: int = 2):
        self.window_bars = window_bars
        self._history: Deque[str] = deque(maxlen=window_bars)

    def update(self, raw_signal: str) -> str:
        self._history.append(raw_signal)

        if len(self._history) < self.window_bars:
            return "FLAT"

        if all(signal == "LONG" for signal in self._history):
            return "LONG"
        if all(signal == "SHORT" for signal in self._history):
            return "SHORT"
        return "FLAT"
