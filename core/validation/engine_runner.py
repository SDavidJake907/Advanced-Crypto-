from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from core.execution.mock_exec import MockExecutor
from core.llm.nemotron import NemotronStrategist
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig
from core.runtime.tools import execution_place_order, portfolio_evaluate, risk_adjust
from core.state.portfolio import PortfolioState
from core.state.position_state_store import merge_persisted_positions
from core.risk.portfolio import PositionState
from core.strategy.simple_momo import SimpleMomentumStrategy


def _env_flag(name: str, default: str = "false") -> bool:
    raw = str(os.getenv(name, default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class EngineComponents:
    engine_name: str
    strategy: SimpleMomentumStrategy
    risk_engine: BasicRiskEngine
    portfolio_config: PortfolioConfig
    executor: MockExecutor
    nemotron: NemotronStrategist | None = None


@dataclass
class EngineDecision:
    signal: str
    risk_checks: list[str]
    portfolio_decision: dict[str, Any]
    execution: dict[str, Any]
    timings: dict[str, float]


def build_engine_components(engine_name: str) -> EngineComponents:
    normalized = str(engine_name or "llm").strip().lower()
    strategy = SimpleMomentumStrategy()
    risk_engine = BasicRiskEngine()
    portfolio_config = PortfolioConfig.from_runtime()
    executor = MockExecutor(allow_short=_env_flag("REPLAY_ALLOW_SHORTS", "false"))
    nemotron = None
    if normalized == "llm":
        nemotron = NemotronStrategist(
            strategy=strategy,
            risk_engine=risk_engine,
            portfolio_config=portfolio_config,
            executor=executor,
        )
    return EngineComponents(
        engine_name=normalized,
        strategy=strategy,
        risk_engine=risk_engine,
        portfolio_config=portfolio_config,
        executor=executor,
        nemotron=nemotron,
    )


def clone_position_state(state: PositionState) -> PositionState:
    return merge_persisted_positions(PositionState(), state)


def clone_portfolio_state(state: PortfolioState) -> PortfolioState:
    return PortfolioState(
        cash=float(state.cash),
        positions=dict(state.positions or {}),
        position_marks=dict(state.position_marks or {}),
        pnl=float(state.pnl),
        initial_equity=float(state.initial_equity),
        last_fill_bar_ts=state.last_fill_bar_ts,
        last_fill_bar_idx=state.last_fill_bar_idx,
        last_fill_symbol=state.last_fill_symbol,
        last_fill_side=state.last_fill_side,
    )


def decide_with_engine(
    components: EngineComponents,
    *,
    symbol: str,
    features: dict[str, Any],
    portfolio_state: PortfolioState,
    positions_state: PositionState,
    symbols: list[str],
    proposed_weight: float,
) -> EngineDecision:
    if components.engine_name == "llm":
        assert components.nemotron is not None
        previous_validation_mode = os.environ.get("VALIDATION_MODE")
        os.environ["VALIDATION_MODE"] = "1"
        try:
            decision = components.nemotron.decide(
                symbol=symbol,
                features=features,
                portfolio_state=portfolio_state,
                positions_state=positions_state,
                symbols=symbols,
                proposed_weight=proposed_weight,
            )
        finally:
            if previous_validation_mode is None:
                os.environ.pop("VALIDATION_MODE", None)
            else:
                os.environ["VALIDATION_MODE"] = previous_validation_mode
        return EngineDecision(
            signal=decision.signal,
            risk_checks=list(decision.risk_checks),
            portfolio_decision=dict(decision.portfolio_decision),
            execution=dict(decision.execution),
            timings=dict(decision.timings),
        )

    signal = components.strategy.generate_signal(features)
    risk_checks = risk_adjust(
        components.risk_engine,
        signal=signal,
        features=features,
        portfolio_state=portfolio_state,
    )
    portfolio_decision = portfolio_evaluate(
        config=components.portfolio_config,
        positions=positions_state,
        symbol=symbol,
        signal=signal,
        proposed_weight=proposed_weight,
        features=features,
        symbols=symbols,
    )
    if portfolio_decision["reasons"]:
        risk_checks.extend(portfolio_decision["reasons"])
    if portfolio_decision["decision"] == "block" and "block" not in risk_checks:
        risk_checks.append("block")

    if "block" in risk_checks:
        execution = {
            "status": "blocked",
            "reason": list(risk_checks),
            "bar_ts": features.get("bar_ts"),
            "bar_idx": features.get("bar_idx"),
        }
    else:
        execution = execution_place_order(
            components.executor,
            signal=signal,
            symbol=symbol,
            features=features,
            portfolio_state=portfolio_state,
            size_factor=float(portfolio_decision.get("size_factor", 1.0) or 1.0),
        )
    return EngineDecision(
        signal=signal,
        risk_checks=risk_checks,
        portfolio_decision=portfolio_decision,
        execution=execution,
        timings={"phi3_ms": 0.0, "advisory_ms": 0.0, "nemotron_ms": 0.0, "execution_ms": 0.0, "total_ms": 0.0},
    )
