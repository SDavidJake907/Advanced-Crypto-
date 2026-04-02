from __future__ import annotations

from typing import Any

from core.config.runtime import get_symbol_lane
from core.execution.cpp_exec import CppExecutor
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState, evaluate_trade
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy


def strategy_decision(
    strategy: SimpleMomentumStrategy,
    features: dict[str, Any],
    *,
    reflex_decision: dict[str, Any] | None = None,
) -> str:
    if reflex_decision is not None:
        action = str(reflex_decision.get("action", "")).upper()
        if action == "HOLD":
            return "FLAT"
        override = reflex_decision.get("override_signal")
        if override in ("LONG", "SHORT", "FLAT"):
            return override
    signal = strategy.generate_signal(features)
    return signal


def risk_adjust(
    risk_engine: BasicRiskEngine,
    *,
    signal: str,
    features: dict[str, Any],
    portfolio_state: PortfolioState,
) -> list[str]:
    return risk_engine.check(signal, features, portfolio_state.to_dict())


def portfolio_evaluate(
    *,
    config: PortfolioConfig,
    positions: PositionState,
    symbol: str,
    signal: str,
    proposed_weight: float,
    features: dict[str, Any],
    symbols: list[str],
) -> dict[str, Any]:
    if signal not in ("LONG", "SHORT"):
        return {"decision": "allow", "size_factor": 1.0, "reasons": []}
    lane = str(features.get("lane") or get_symbol_lane(symbol))
    trend_1h = int(features.get("trend_1h", 0))
    trend_conflict = (lane == "L4" and ((signal == "LONG" and trend_1h == -1) or (signal == "SHORT" and trend_1h == 1)))
    return evaluate_trade(
        config=config,
        positions=positions,
        symbol=symbol,
        side=signal,
        proposed_weight=proposed_weight,
        correlation_row=features["correlation_row"],
        symbols=symbols,
        lane=lane,
        trend_conflict=trend_conflict,
        features=features,
    )


def execution_place_order(
    executor: CppExecutor,
    *,
    signal: str,
    symbol: str,
    features: dict[str, Any],
    portfolio_state: PortfolioState,
    size_factor: float,
) -> dict[str, Any]:
    if signal not in ("LONG", "SHORT") or float(size_factor) <= 0.0:
        return {
            "status": "no_trade",
            "bar_ts": features.get("bar_ts"),
            "bar_idx": features.get("bar_idx"),
        }
    exec_result = executor.execute(
        signal,
        symbol,
        features,
        portfolio_state.to_dict(),
        size_factor=size_factor,
    )
    exec_result["bar_ts"] = features.get("bar_ts")
    exec_result["bar_idx"] = features.get("bar_idx")
    return exec_result
