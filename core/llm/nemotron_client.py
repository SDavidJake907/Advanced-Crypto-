from __future__ import annotations

import logging
import os
from typing import Any

from core.config.runtime import get_runtime_setting, stage_runtime_override_proposal
from core.execution.cpp_exec import CppExecutor
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState
from core.runtime.tools import strategy_decision
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy

_LOGGER = logging.getLogger(__name__)

# Keys Nemo is allowed to adjust, with (type, min, max) bounds
_ADJUSTABLE_KEYS: dict[str, tuple[type, float, float]] = {
    "PORTFOLIO_MAX_OPEN_POSITIONS": (int, 1, 6),
    "MEME_MAX_OPEN_POSITIONS": (int, 0, 3),
    "EXEC_RISK_PER_TRADE_PCT": (float, 5.0, 75.0),
    "MEME_EXEC_RISK_PER_TRADE_PCT": (float, 5.0, 75.0),
}


class NemotronClient:
    def __init__(
        self,
        *,
        strategy: SimpleMomentumStrategy,
        risk: BasicRiskEngine,
        portfolio_config: PortfolioConfig,
        execution: CppExecutor,
        portfolio_state: PortfolioState,
        positions_state: PositionState,
        symbols: list[str],
    ) -> None:
        self.strategy = strategy
        self.risk = risk
        self.portfolio_config = portfolio_config
        self.execution = execution
        self.portfolio_state = portfolio_state
        self.positions_state = positions_state
        self.symbols = symbols
        self._current_features: dict[str, Any] = {}
        self.tools = {
            "strategy_decision": self._tool_strategy_decision,
            "portfolio_adjust": self._tool_portfolio_adjust,
        }

    def set_features(self, features: dict[str, Any]) -> None:
        """Store the full feature set so the tool always uses authoritative data."""
        self._current_features = features

    def _tool_strategy_decision(self, **kwargs: Any) -> dict[str, Any]:
        # Always use the full authoritative features — Nemotron often generates
        # a minimal tool call that only has symbol/momentum/rsi/atr/price.
        # Merge: kwargs base, overridden by stored full features.
        features = {**kwargs.get("features", {}), **self._current_features}
        reflex = kwargs.get("reflex")
        if not isinstance(reflex, dict):
            reflex = {
                "reflex": str(reflex or kwargs.get("micro_state") or "allow"),
                "micro_state": str(kwargs.get("micro_state") or "unknown"),
                "reason": "nemotron_tool_reflex_coerced",
            }

        # Derive a signal from the deterministic verifier output when available.
        # SimpleMomentumStrategy uses raw 1m momentum which may be flat during
        # corrections even when the entry verifier sees a valid BUY setup.
        entry_rec = str(features.get("entry_recommendation", "")).upper()
        entry_score = float(features.get("entry_score", 0.0) or 0.0)
        reversal_risk = str(features.get("reversal_risk", "")).upper()
        rotation_score = float(features.get("rotation_score", 0.0) or 0.0)

        verifier_signal: str | None = None
        if entry_rec in {"STRONG_BUY", "BUY"} and reversal_risk not in {"HIGH"} and entry_score >= 55.0 and rotation_score > 0.0:
            verifier_signal = "LONG"
        elif entry_rec == "WATCH" and reversal_risk == "LOW" and entry_score >= 62.0 and rotation_score > 0.0:
            verifier_signal = "LONG"

        if verifier_signal is not None:
            signal = verifier_signal
        else:
            signal = strategy_decision(self.strategy, features, reflex_decision=reflex)

        return {
            "tool": "strategy_decision",
            "result": {
                "signal": signal,
                "entry_score": round(entry_score, 1),
                "entry_recommendation": entry_rec,
                "reversal_risk": reversal_risk,
                "rotation_score": round(rotation_score, 3),
            },
        }

    def _tool_portfolio_adjust(self, **kwargs: Any) -> dict[str, Any]:
        """Stage a single portfolio parameter adjustment for replay/shadow/human review."""
        key = str(kwargs.get("key", "")).strip().upper()
        raw_value = kwargs.get("value")
        reason = str(kwargs.get("reason", "nemo_adjustment"))

        if key not in _ADJUSTABLE_KEYS:
            return {
                "success": False,
                "error": f"Key '{key}' is not adjustable. Allowed: {list(_ADJUSTABLE_KEYS)}",
            }

        target_type, min_val, max_val = _ADJUSTABLE_KEYS[key]
        try:
            value = target_type(raw_value)
        except (TypeError, ValueError):
            return {"success": False, "error": f"Invalid value {raw_value!r} for {key}"}

        # Clamp to safe bounds
        clamped = target_type(max(min_val, min(max_val, value)))

        if os.getenv("VALIDATION_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return {
                "success": True,
                "staged": False,
                "effective_immediately": False,
                "key": key,
                "proposed_value": clamped,
                "reason": reason,
                "proposal_id": "",
                "validation_mode": True,
            }

        try:
            proposal = stage_runtime_override_proposal(
                {key: clamped},
                source="nemotron_tool",
                summary=reason,
                context={"requested_value": raw_value},
            )
        except Exception as exc:
            _LOGGER.warning("portfolio_adjust: failed to stage override %s=%s: %s", key, clamped, exc)
            return {"success": False, "error": str(exc)}

        _LOGGER.info("portfolio_adjust: %s → %s (was %s) reason=%s", key, clamped, raw_value, reason)
        return {
            "success": True,
            "staged": True,
            "effective_immediately": False,
            "key": key,
            "proposed_value": clamped,
            "reason": reason,
            "proposal_id": proposal["id"],
        }
