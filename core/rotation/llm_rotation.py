from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.llm.client import run_nemotron_tool_loop


@dataclass
class RotationConfig:
    rebalance_minutes: int = 30
    active_min: int = 2
    active_max: int = 8
    max_adds_per_rebalance: int = 2
    max_removes_per_rebalance: int = 2
    pair_cooldown_minutes: int = 60


@dataclass
class PairCandidate:
    symbol: str
    liquidity: float
    spread: float
    age_minutes: float
    momentum: float
    volatility: float
    trend_1h: int = 0
    regime_7d: str = "unknown"
    macro_30d: str = "sideways"
    expected_edge: float = 0.0
    correlation_to_book: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RotationState:
    active_symbols: list[str] = field(default_factory=list)
    cooldown_minutes_remaining: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_candidate(candidate: PairCandidate) -> dict[str, Any]:
    score = (
        candidate.momentum * 3.0
        - candidate.spread * 2.0
        - candidate.volatility * 0.25
        + candidate.expected_edge * 4.0
        - candidate.correlation_to_book
    )
    if candidate.trend_1h > 0:
        score += 0.5
    if candidate.regime_7d == "trending":
        score += 0.5
    if candidate.macro_30d == "bull":
        score += 0.25
    return {
        "symbol": candidate.symbol,
        "score": float(score),
        "reason": "scored_from_liquidity_spread_momentum_volatility_edge_and_correlation",
    }


def _apply_rotation_constraints(
    ranking: list[dict[str, Any]],
    *,
    state: RotationState,
    config: RotationConfig,
) -> dict[str, Any]:
    eligible = [
        item for item in ranking
        if state.cooldown_minutes_remaining.get(item["symbol"], 0) <= 0
    ]
    target = eligible[: config.active_max]
    kept = [item["symbol"] for item in target]
    adds = [symbol for symbol in kept if symbol not in state.active_symbols][: config.max_adds_per_rebalance]
    removes = [
        symbol for symbol in state.active_symbols if symbol not in kept
    ][: config.max_removes_per_rebalance]
    final_symbols = [symbol for symbol in state.active_symbols if symbol not in removes]
    for symbol in adds:
        if symbol not in final_symbols:
            final_symbols.append(symbol)
    final_symbols = final_symbols[: config.active_max]
    if len(final_symbols) < config.active_min:
        for item in eligible:
            if item["symbol"] not in final_symbols:
                final_symbols.append(item["symbol"])
            if len(final_symbols) >= config.active_min:
                break
    weight = 1.0 / max(len(final_symbols), 1)
    weights = {symbol: weight for symbol in final_symbols}
    return {
        "ranked": ranking,
        "adds": adds,
        "removes": removes,
        "weights": weights,
        "final_symbols": final_symbols,
    }


ROTATION_SYSTEM_PROMPT = """
You are Nemotron9B running the rotation loop for a deterministic trading system.

You evaluate candidate pairs and may use tools to:
- score candidates
- apply rebalance constraints

Return only JSON.
Either request a tool:
{"tool":"tool_name","args":{...}}
Or return a final decision object:
{
  "ranked": [...],
  "adds": [...],
  "removes": [...],
  "weights": {...},
  "reasons": {...}
}
No prose outside JSON.
""".strip()


def run_rotation_llm(
    *,
    candidates: list[PairCandidate],
    state: RotationState,
    config: RotationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RotationConfig()
    tool_registry = {
        "score_candidate": lambda candidate: _score_candidate(PairCandidate(**candidate)),
        "apply_rotation_constraints": lambda ranking: _apply_rotation_constraints(
            ranking,
            state=state,
            config=cfg,
        ),
    }
    initial_payload = {
        "candidates": [candidate.to_dict() for candidate in candidates],
        "state": state.to_dict(),
        "config": asdict(cfg),
    }
    response = run_nemotron_tool_loop(
        initial_payload,
        system=ROTATION_SYSTEM_PROMPT,
        tool_registry=tool_registry,
    )
    response.setdefault("ranked", [])
    response.setdefault("adds", [])
    response.setdefault("removes", [])
    response.setdefault("weights", {})
    response.setdefault("reasons", {})
    return response
