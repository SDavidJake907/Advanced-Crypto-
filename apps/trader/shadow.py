"""Shadow validation and LLM metrics for the trader loop.

Responsibilities:
- Run the shadow engine on copied portfolio/position state
- Build and emit shadow comparison payloads
- Extract LLM role metrics from execution results (approve/reject/defer per role)
"""

from __future__ import annotations

from typing import Any

from core.llm.contracts import classify_defer_reason
from core.validation.engine_runner import (
    decide_with_engine,
    clone_portfolio_state,
    clone_position_state,
)


# ---------------------------------------------------------------------------
# LLM role metrics
# ---------------------------------------------------------------------------

def _role_decision(role: str, contract: dict[str, Any], source: dict[str, Any]) -> str:
    if role == "trade_reviewer":
        action = str(source.get("action", "HOLD") or "HOLD").upper()
        if action in {"OPEN", "CLOSE"}:
            return "approve"
        return "defer"
    if role == "market_reviewer":
        market_state = str(source.get("market_state", "transition") or "transition").lower()
        lane_bias = str(source.get("lane_bias", "favor_selective") or "favor_selective").lower()
        if market_state == "trending":
            return "approve"
        if market_state == "ranging":
            if lane_bias == "reduce_trend_entries":
                return "reject"
            return "defer"
        return "defer"
    return "defer"


def _llm_role_metric(
    role: str, lane: object, symbol: str, source: dict[str, Any]
) -> dict[str, Any] | None:
    contract = source.get("contract", {})
    if not isinstance(contract, dict):
        return None
    reasons = contract.get("reasons", [])
    contradictions = contract.get("contradictions", [])
    risks = contract.get("risks", [])
    meta = contract.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    decision = _role_decision(role, contract, source)
    defer_reason = (
        (reasons[0] if isinstance(reasons, list) and reasons else str(source.get("reason", "")))
        if decision == "defer"
        else ""
    )
    defer_reason_category = classify_defer_reason(defer_reason) if defer_reason else ""
    return {
        "role": role,
        "lane": str(lane or "unknown"),
        "symbol": symbol,
        "decision": decision,
        "confidence": float(contract.get("confidence", 0.0) or 0.0),
        "contradiction_count": len(contradictions) if isinstance(contradictions, list) else 0,
        "risk_count": len(risks) if isinstance(risks, list) else 0,
        "schema_valid": bool(meta.get("schema_valid", True)),
        "normalized_field_count": int(meta.get("normalized_field_count", 0) or 0),
        "defer_reason": defer_reason,
        "defer_reason_category": defer_reason_category,
        "model": str(meta.get("model", "")),
        "provider": str(meta.get("provider", "")),
        "prompt_version": str(meta.get("prompt_version", "")),
        "contract_version": str(contract.get("version", "")),
    }


def extract_llm_metrics(
    symbol: str, lane: object, exec_result: dict[str, Any]
) -> dict[str, Any]:
    """Return approve/reject/defer counts and per-role details from an execution result."""
    nemotron = exec_result.get("nemotron", {})
    market_state = (
        exec_result.get("nemotron", {}).get("market_state_review", {})
        if isinstance(exec_result.get("nemotron", {}), dict)
        else {}
    )
    roles = []
    if isinstance(nemotron, dict):
        metric = _llm_role_metric("trade_reviewer", lane, symbol, nemotron)
        if metric:
            roles.append(metric)
    if isinstance(market_state, dict) and market_state:
        metric = _llm_role_metric("market_reviewer", lane, symbol, market_state)
        if metric:
            roles.append(metric)
    aggregate = {
        "approve": sum(1 for item in roles if item["decision"] == "approve"),
        "reject":  sum(1 for item in roles if item["decision"] == "reject"),
        "defer":   sum(1 for item in roles if item["decision"] == "defer"),
        "normalized_fields": sum(int(item["normalized_field_count"]) for item in roles),
    }
    return {"roles": roles, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Shadow engine execution
# ---------------------------------------------------------------------------

def prepare_shadow_state(portfolio, positions_state):
    """Clone portfolio and position state for isolated shadow evaluation."""
    return clone_portfolio_state(portfolio), clone_position_state(positions_state)


async def run_shadow_decision(
    run_blocking,
    shadow_components,
    *,
    symbol: str,
    features: dict[str, Any],
    shadow_portfolio,
    shadow_positions,
    all_symbols: list[str],
    proposed_weight: float,
):
    """Execute the shadow engine on cloned state and return the shadow decision."""
    return await run_blocking(
        decide_with_engine,
        shadow_components,
        symbol=symbol,
        features=features,
        portfolio_state=shadow_portfolio,
        positions_state=shadow_positions,
        symbols=all_symbols,
        proposed_weight=proposed_weight,
    )


def build_shadow_payload(
    *,
    ts: str,
    symbol: str,
    features: dict[str, Any],
    baseline_engine: str,
    shadow_components,
    baseline_signal: str,
    baseline_exec: dict[str, Any],
    shadow_decision,
) -> dict[str, Any]:
    """Build the shadow comparison dict that gets written to shadow_decisions.jsonl."""
    return {
        "ts": ts,
        "symbol": symbol,
        "lane": features.get("lane"),
        "coin_profile": features.get("coin_profile", {}),
        "baseline_engine": baseline_engine,
        "shadow_engine": shadow_components.engine_name,
        "baseline_signal": baseline_signal,
        "shadow_signal": shadow_decision.signal,
        "baseline_status": str(baseline_exec.get("status", "")),
        "shadow_status": str(shadow_decision.execution.get("status", "")),
        "signal_match": baseline_signal == shadow_decision.signal,
        "status_match": (
            str(baseline_exec.get("status", ""))
            == str(shadow_decision.execution.get("status", ""))
        ),
        "baseline_reason": baseline_exec.get(
            "reason", baseline_exec.get("nemotron", {}).get("reason", "")
        ),
        "shadow_reason": shadow_decision.execution.get(
            "reason", shadow_decision.execution.get("nemotron", {}).get("reason", "")
        ),
    }
