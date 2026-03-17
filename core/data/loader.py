from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class CandleLoader:
    def __init__(self, path: str, symbol_col: Optional[str] = None, symbol: Optional[str] = None):
        self.path = Path(path)
        self.symbol_col = symbol_col
        self.symbol = symbol

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if self.symbol_col and self.symbol:
            df = df[df[self.symbol_col] == self.symbol]

        df = df.sort_values("timestamp")
        return df
