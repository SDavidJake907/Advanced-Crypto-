from __future__ import annotations

from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    def execute(self, signal: str, symbol: str, features: dict, state: dict) -> dict:
        ...

