from __future__ import annotations

from typing import List

from core.config.runtime import get_cooldown_bars, get_runtime_setting
from core.risk.base import RiskEngine


class BasicRiskEngine(RiskEngine):
    def __init__(
        self,
        max_position_notional: float | None = None,
        max_leverage: float | None = None,
        cooldown_bars: int | None = None,
    ):
        self.max_position_notional = (
            float(max_position_notional)
            if max_position_notional is not None
            else float(get_runtime_setting("RISK_MAX_POSITION_NOTIONAL"))
        )
        self.max_leverage = (
            float(max_leverage)
            if max_leverage is not None
            else float(get_runtime_setting("RISK_MAX_LEVERAGE"))
        )
        self.cooldown_bars = cooldown_bars if cooldown_bars is not None else get_cooldown_bars()

    def check(self, signal: str, features: dict, state: dict) -> List[str]:
        self.max_position_notional = float(get_runtime_setting("RISK_MAX_POSITION_NOTIONAL"))
        self.max_leverage = float(get_runtime_setting("RISK_MAX_LEVERAGE"))
        self.cooldown_bars = get_cooldown_bars()
        checks: List[str] = []

        if signal == "FLAT":
            checks.append("no_action")
            return checks

        price = features.get("price", 0.0)
        cash = state.get("cash", 0.0)
        positions = state.get("positions", {})
        symbol = features.get("symbol", None)
        current_bar_ts = features.get("bar_ts")
        bar_idx = features.get("bar_idx")
        last_fill_bar_ts = state.get("last_fill_bar_ts")
        last_fill_bar_idx = state.get("last_fill_bar_idx")
        last_fill_symbol = state.get("last_fill_symbol")
        last_fill_side = state.get("last_fill_side")
        runtime_health = features.get("runtime_health", {})

        current_notional = 0.0
        if symbol and symbol in positions:
            current_notional = positions[symbol] * price

        if symbol and current_bar_ts is not None and last_fill_bar_ts is not None and symbol == last_fill_symbol:
            if current_bar_ts == last_fill_bar_ts:
                checks.append("block_same_bar_entry")
                if last_fill_side is not None and signal != last_fill_side:
                    checks.append("block_same_bar_flip")

        cooldown = self.cooldown_bars
        if cooldown and last_fill_bar_idx is not None and bar_idx is not None:
            if bar_idx - last_fill_bar_idx < cooldown:
                checks.append("block_cooldown")

        equity = cash + current_notional
        if equity <= 0:
            checks.append("block_equity_non_positive")
            return checks

        if abs(current_notional) > self.max_position_notional:
            checks.append("block_max_notional")

        leverage = abs(current_notional) / equity if equity > 0 else 0.0
        if leverage > self.max_leverage:
            checks.append("block_max_leverage")

        if isinstance(runtime_health, dict):
            for reason in runtime_health.get("reasons", []):
                checks.append(str(reason))
        if not bool(features.get("book_valid", True)):
            if signal in {"LONG", "SHORT"}:
                checks.append("runtime_health_book_invalid")
                checks.append("block")
            else:
                checks.append("warn_book_invalid")

        if any(check.startswith("block") for check in checks):
            checks.append("block")

        return checks
