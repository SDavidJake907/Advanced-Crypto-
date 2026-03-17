from __future__ import annotations

from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, features: dict) -> str:
        ...

