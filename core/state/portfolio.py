from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class PortfolioState:
    cash: float = 10_000.0
    positions: Dict[str, float] | None = None
    position_marks: Dict[str, float] | None = None
    pnl: float = 0.0
    initial_equity: float = 10_000.0
    last_fill_bar_ts: Optional[str] = None
    last_fill_bar_idx: Optional[int] = None
    last_fill_symbol: Optional[str] = None
    last_fill_side: Optional[str] = None

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = {}
        if self.position_marks is None:
            self.position_marks = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def apply_execution(self, exec_result: Dict[str, Any]) -> Dict[str, Any]:
        if exec_result.get("status") != "filled":
            return {}

        symbol = exec_result["symbol"]
        qty = exec_result["qty"]
        price = exec_result["price"]
        side = exec_result["side"]
        notional = qty * price
        fee = exec_result.get("fee", 0.0)

        if side == "BUY":
            self.cash -= notional + fee
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        elif side == "SELL":
            self.cash += notional - fee
            self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
        mark_price = float(exec_result.get("mark_price", price))
        if self.positions.get(symbol, 0.0) == 0.0:
            self.position_marks.pop(symbol, None)
        else:
            self.position_marks[symbol] = mark_price

        self.last_fill_bar_ts = exec_result.get("bar_ts")
        self.last_fill_bar_idx = exec_result.get("bar_idx")
        self.last_fill_symbol = symbol
        self.last_fill_side = "LONG" if side == "BUY" else "SHORT"
        self.pnl = self._compute_pnl_mark_to_market(mark_price)

        return {"cash": self.cash, "positions": dict(self.positions), "pnl": self.pnl}

    def _compute_pnl_mark_to_market(self, default_mark_price: float) -> float:
        total_pos_value = 0.0
        for symbol, qty in self.positions.items():
            mark_price = float(self.position_marks.get(symbol, default_mark_price))
            total_pos_value += qty * mark_price
        equity = self.cash + total_pos_value
        return equity - self.initial_equity
