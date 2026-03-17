from __future__ import annotations

from abc import ABC, abstractmethod


class RiskEngine(ABC):
    @abstractmethod
    def check(self, signal: str, features: dict, state: dict) -> list[str]:
        ...

