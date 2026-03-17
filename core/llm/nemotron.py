from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from core.config.runtime import get_runtime_setting, get_runtime_snapshot
from core.execution.cpp_exec import CppExecutor
from core.llm.client import NEMOTRON_DEBUG_VERBOSE, run_nemotron_tool_loop, sanitize_for_json
from core.llm.orchestrator import build_advisory_bundle
from core.llm.nemotron_client import NemotronClient
from core.llm.prompts import NEMOTRON_STRATEGIST_SYSTEM_PROMPT
from core.policy.nemotron_gate import (
    load_universe_candidate_context,
    passes_deterministic_candidate_gate,
    should_run_nemotron,
)
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState
from core.runtime.tools import execution_place_order, portfolio_evaluate, risk_adjust, strategy_decision
from core.runtime.log_rotation import rotate_jsonl_if_needed
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy


@dataclass
class NemotronDecision:
    signal: str
    risk_checks: list[str]
    portfolio_decision: dict[str, Any]
    execution: dict[str, Any]
    reflex: dict[str, Any]
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NemotronStrategist:
    def __init__(
        self,
        *,
        strategy: SimpleMomentumStrategy,
        risk_engine: BasicRiskEngine,
        portfolio_config: PortfolioConfig,
        executor: CppExecutor,
    ) -> None:
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.portfolio_config = portfolio_config
        self.executor = executor
        self._decision_cache: dict[str, dict[str, Any]] = {}

    def _integrate_reflex(self, features: dict[str, Any], reflex: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": "HOLD",
            "reason": "nemotron_fallback_hold",
            "debug": {},
            "override_signal": "FLAT",
            "size_factor_hint": 1.0,
            "reflex": reflex.get("reflex", "allow"),
            "micro_state": reflex.get("micro_state", "unknown"),
        }

    def _reflex_is_hard_block(self, reflex: dict[str, Any]) -> bool:
        return str(reflex.get("micro_state", "")) == "data_integrity_issue"

    def _normalize_size_hint(self, raw_size: Any, proposed_weight: float) -> float:
        try:
            size = float(raw_size)
        except (TypeError, ValueError):
            return 1.0
        if size <= 0:
            return 1.0
        if size <= 1.0:
            return size
        if proposed_weight > 0:
            return max(min(size / proposed_weight, 2.0), 0.1)
        return 1.0

    def _validate_final_decision(
        self,
        *,
        symbol: str,
        final_decision: dict[str, Any],
        proposed_weight: float,
    ) -> dict[str, Any]:
        action = str(final_decision.get("action", "HOLD")).upper()
        raw_side = final_decision.get("side")
        side = str(raw_side).upper() if raw_side is not None else None
        if side == "BUY":
            side = "LONG"
        elif side == "SELL":
            side = "SHORT"
        reason = str(final_decision.get("reason", "nemotron_integrated"))
        debug = final_decision.get("debug", {}) if isinstance(final_decision.get("debug", {}), dict) else {}
        decision_symbol = str(final_decision.get("symbol", symbol) or symbol)

        if action not in {"OPEN", "CLOSE", "HOLD"}:
            return {
                "action": "HOLD",
                "reason": "invalid_final_decision_contract",
                "debug": {"contract_error": "invalid_action", "raw_action": action},
                "override_signal": "FLAT",
                "size_factor_hint": 1.0,
            }
        if decision_symbol != symbol and action != "HOLD":
            return {
                "action": "HOLD",
                "reason": "invalid_final_decision_contract",
                "debug": {"contract_error": "symbol_mismatch", "raw_symbol": decision_symbol},
                "override_signal": "FLAT",
                "size_factor_hint": 1.0,
            }
        if action == "OPEN":
            if side not in {"LONG", "SHORT"}:
                return {
                    "action": "HOLD",
                    "reason": "invalid_final_decision_contract",
                    "debug": {"contract_error": "invalid_open_side", "raw_side": raw_side},
                    "override_signal": "FLAT",
                    "size_factor_hint": 1.0,
                }
            size_hint = self._normalize_size_hint(final_decision.get("size", 1.0), proposed_weight)
            raw_size = final_decision.get("size", 1.0)
            try:
                raw_size_float = float(raw_size)
            except (TypeError, ValueError):
                raw_size_float = 0.0
            if raw_size_float <= 0:
                return {
                    "action": "HOLD",
                    "reason": "invalid_final_decision_contract",
                    "debug": {"contract_error": "non_positive_open_size", "raw_size": raw_size},
                    "override_signal": "FLAT",
                    "size_factor_hint": 1.0,
                }
            return {
                "action": action,
                "reason": reason,
                "debug": debug,
                "override_signal": side,
                "size_factor_hint": size_hint,
            }
        return {
            "action": action,
            "reason": reason,
            "debug": debug,
            "override_signal": "FLAT" if action == "HOLD" else side,
            "size_factor_hint": 1.0,
        }

    def _decision_fingerprint(
        self,
        *,
        symbol: str,
        features: dict[str, Any],
        portfolio_state: PortfolioState,
        positions_state: PositionState,
    ) -> tuple[Any, ...]:
        held_symbols = sorted(position.symbol for position in positions_state.all())
        return (
            symbol,
            str(features.get("lane", "")),
            round(float(features.get("entry_score", 0.0) or 0.0), 1),
            str(features.get("entry_recommendation", "")),
            str(features.get("reversal_risk", "")),
            round(float(features.get("rotation_score", 0.0) or 0.0), 3),
            round(float(features.get("momentum_5", 0.0) or 0.0), 4),
            round(float(features.get("momentum_14", 0.0) or 0.0), 4),
            round(float(features.get("volume_ratio", 0.0) or 0.0), 3),
            int(bool(features.get("trend_confirmed"))),
            str(features.get("regime_7d", "")),
            str(features.get("macro_30d", "")),
            round(float(features.get("price_zscore", 0.0) or 0.0), 3),
            round(float(portfolio_state.cash or 0.0), 2),
            tuple(held_symbols),
        )

    def _get_cached_decision(
        self,
        *,
        symbol: str,
        fingerprint: tuple[Any, ...],
    ) -> NemotronDecision | None:
        ttl_sec = float(get_runtime_setting("NEMOTRON_VERDICT_CACHE_TTL_SEC"))
        cached = self._decision_cache.get(symbol)
        if not cached:
            return None
        age_sec = time.time() - float(cached.get("ts", 0.0) or 0.0)
        if age_sec > ttl_sec or cached.get("fingerprint") != fingerprint:
            return None
        decision = copy.deepcopy(cached.get("decision"))
        if not isinstance(decision, NemotronDecision):
            return None
        decision.timings["nemotron_ms"] = 0.0
        decision.timings["execution_ms"] = 0.0
        decision.timings["total_ms"] = round(
            float(decision.timings.get("phi3_ms", 0.0)) + float(decision.timings.get("advisory_ms", 0.0)),
            2,
        )
        decision.execution = copy.deepcopy(decision.execution)
        decision.execution.setdefault("nemotron", {})
        decision.execution["nemotron"]["cache_hit"] = True
        return decision

    def _store_cached_decision(self, *, symbol: str, fingerprint: tuple[Any, ...], decision: NemotronDecision) -> None:
        if str(decision.execution.get("status", "")).lower() not in {"blocked", "no_trade", "rejected"}:
            return
        self._decision_cache[symbol] = {
            "ts": time.time(),
            "fingerprint": fingerprint,
            "decision": copy.deepcopy(decision),
        }

    def _build_payload(
        self,
        *,
        symbol: str,
        symbols: list[str],
        features: dict[str, Any],
        reflex: dict[str, Any],
        portfolio_state: PortfolioState,
        positions_state: PositionState,
        candidate_review: dict[str, Any] | None = None,
        market_state_review: dict[str, Any] | None = None,
        posture_review: dict[str, Any] | None = None,
        visual_review: dict[str, Any] | None = None,
        universe_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return sanitize_for_json(
            {
                "features": {
                    "symbol": features.get("symbol"),
                    "lane": features.get("lane"),
                    "momentum": features.get("momentum"),
                    "momentum_5": features.get("momentum_5"),
                    "momentum_14": features.get("momentum_14"),
                    "momentum_30": features.get("momentum_30"),
                    "rotation_score": features.get("rotation_score"),
                    "volatility": features.get("volatility"),
                    "volume": features.get("volume"),
                    "volume_ratio": features.get("volume_ratio"),
                    "volume_surge": features.get("volume_surge"),
                    "volume_surge_flag": features.get("volume_surge_flag"),
                    "price_zscore": features.get("price_zscore"),
                    "history_points": features.get("history_points"),
                    "indicators_ready": features.get("indicators_ready"),
                    "rsi": features.get("rsi"),
                    "atr": features.get("atr"),
                    "hurst": features.get("hurst"),
                    "entropy": features.get("entropy"),
                    "autocorr": features.get("autocorr"),
                    "bb_middle": features.get("bb_middle"),
                    "bb_upper": features.get("bb_upper"),
                    "bb_lower": features.get("bb_lower"),
                    "bb_bandwidth": features.get("bb_bandwidth"),
                    "price": features.get("price"),
                    "book_imbalance": features.get("book_imbalance"),
                    "book_wall_pressure": features.get("book_wall_pressure"),
                    "book_bid_depth": features.get("book_bid_depth"),
                    "book_ask_depth": features.get("book_ask_depth"),
                    "bar_ts": features.get("bar_ts"),
                    "bar_idx": features.get("bar_idx"),
                    "bar_interval_seconds": features.get("bar_interval_seconds"),
                    "correlation_row": features.get("correlation_row"),
                    "correlation_symbols": features.get("correlation_symbols"),
                    "market_fingerprint": features.get("market_fingerprint"),
                    "market_regime": features.get("market_regime"),
                    "trend_1h": features.get("trend_1h"),
                    "regime_7d": features.get("regime_7d"),
                    "macro_30d": features.get("macro_30d"),
                    "entry_score": features.get("entry_score"),
                    "entry_recommendation": features.get("entry_recommendation"),
                    "reversal_risk": features.get("reversal_risk"),
                    "entry_reasons": features.get("entry_reasons"),
                    "sentiment_fng_value": features.get("sentiment_fng_value"),
                    "sentiment_fng_label": features.get("sentiment_fng_label"),
                    "sentiment_btc_dominance": features.get("sentiment_btc_dominance"),
                    "sentiment_market_cap_change_24h": features.get("sentiment_market_cap_change_24h"),
                    "sentiment_symbol_trending": features.get("sentiment_symbol_trending"),
                    "ma_7": features.get("ma_7"),
                    "ma_26": features.get("ma_26"),
                    "macd": features.get("macd"),
                    "macd_signal": features.get("macd_signal"),
                    "macd_hist": features.get("macd_hist"),
                    "trend_confirmed": features.get("trend_confirmed"),
                    "ranging_market": features.get("ranging_market"),
                    "momentum_5m": features.get("momentum_5m"),
                    "trend_5m": features.get("trend_5m"),
                    "momentum_15m": features.get("momentum_15m"),
                    "trend_15m": features.get("trend_15m"),
                    "finbert_score": features.get("finbert_score"),
                    "xgb_score": features.get("xgb_score"),
                },
                "reflex": reflex,
                "micro_state": reflex.get("micro_state"),
                "symbol": symbol,
                "symbols": symbols,
                "runtime_config": get_runtime_snapshot().get("values", {}),
                "portfolio_config": asdict(self.portfolio_config),
                "portfolio_state": portfolio_state.to_dict(),
                "positions_state": [asdict(position) for position in positions_state.all()],
                "universe_context": universe_context or self._load_universe_candidate_context(symbol),
                "candidate_review": candidate_review or {},
                "market_state_review": market_state_review or {},
                "posture_review": posture_review or {},
                "visual_review": visual_review or {},
            }
        )

    def _build_gated_hold_decision(
        self,
        *,
        features: dict[str, Any],
        portfolio_state: PortfolioState,
        gate_reason: str,
        t_total_start: float,
    ) -> NemotronDecision:
        reflex = {
            "reflex": "allow",
            "micro_state": "pre_gate_skip",
            "reason": gate_reason,
        }
        integrated_reflex = {
            "action": "HOLD",
            "reason": "deterministic_candidate_gate",
            "debug": {"gate_reason": gate_reason},
            "override_signal": "FLAT",
            "size_factor_hint": 1.0,
            "reflex": reflex["reflex"],
            "micro_state": reflex["micro_state"],
            "candidate_review": {
                "promotion_decision": "demote",
                "priority": 0.2,
                "action_bias": "hold_preferred",
                "reason": gate_reason,
            },
            "market_state_review": {},
            "posture_review": {},
            "visual_review": {},
        }
        signal = strategy_decision(self.strategy, features, reflex_decision=integrated_reflex)
        risk_checks = risk_adjust(
            self.risk_engine,
            signal=signal,
            features=features,
            portfolio_state=portfolio_state,
        )
        execution = {
            "status": "no_trade",
            "bar_ts": features.get("bar_ts"),
            "bar_idx": features.get("bar_idx"),
            "reflex": reflex,
            "nemotron": integrated_reflex,
        }
        timings = {
            "phi3_ms": 0.0,
            "advisory_ms": 0.0,
            "nemotron_ms": 0.0,
            "execution_ms": 0.0,
            "total_ms": round((time.perf_counter() - t_total_start) * 1000.0, 2),
        }
        return NemotronDecision(
            signal=signal,
            risk_checks=risk_checks,
            portfolio_decision={"decision": "allow", "size_factor": 1.0, "reasons": []},
            execution=execution,
            reflex={"phi3": reflex, "nemotron": integrated_reflex},
            timings=timings,
        )

    def _build_filtered_hold_decision(
        self,
        *,
        features: dict[str, Any],
        portfolio_state: PortfolioState,
        reflex: dict[str, Any],
        market_state_review: dict[str, Any],
        candidate_review: dict[str, Any],
        posture_review: dict[str, Any],
        reason: str,
        phi3_ms: float,
        t_total_start: float,
    ) -> NemotronDecision:
        integrated_reflex = {
            "action": "HOLD",
            "reason": reason,
            "debug": {},
            "override_signal": "FLAT",
            "size_factor_hint": 1.0,
            "reflex": reflex.get("reflex", "allow"),
            "micro_state": reflex.get("micro_state", "unknown"),
            "visual_review": {},
            "market_state_review": market_state_review,
            "posture_review": posture_review,
            "candidate_review": candidate_review,
        }
        signal = strategy_decision(self.strategy, features, reflex_decision=integrated_reflex)
        risk_checks = risk_adjust(
            self.risk_engine,
            signal=signal,
            features=features,
            portfolio_state=portfolio_state,
        )
        execution = {
            "status": "no_trade",
            "bar_ts": features.get("bar_ts"),
            "bar_idx": features.get("bar_idx"),
            "reflex": reflex,
            "nemotron": integrated_reflex,
        }
        timings = {
            "phi3_ms": round(phi3_ms, 2),
            "advisory_ms": 0.0,
            "nemotron_ms": 0.0,
            "execution_ms": 0.0,
            "total_ms": round((time.perf_counter() - t_total_start) * 1000.0, 2),
        }
        return NemotronDecision(
            signal=signal,
            risk_checks=risk_checks,
            portfolio_decision={"decision": "allow", "size_factor": 1.0, "reasons": []},
            execution=execution,
            reflex={"phi3": reflex, "nemotron": integrated_reflex},
            timings=timings,
        )

    def _log_fallback(self, payload: dict[str, Any], exc: Exception) -> None:
        path = Path("logs/nemotron_debug.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rotate_jsonl_if_needed(path)
        event = {
            "event": "nemotron_fallback",
            "error": str(exc),
        }
        if NEMOTRON_DEBUG_VERBOSE:
            event["payload"] = payload
        event = sanitize_for_json(event)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _log_event(self, event: dict[str, Any]) -> None:
        path = Path("logs/nemotron_debug.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rotate_jsonl_if_needed(path)
        safe_event = sanitize_for_json(event)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_event) + "\n")

    def decide(
        self,
        *,
        symbol: str,
        features: dict[str, Any],
        portfolio_state: PortfolioState,
        positions_state: PositionState,
        symbols: list[str],
        proposed_weight: float,
    ) -> NemotronDecision:
        t_total_start = time.perf_counter()
        universe_context = load_universe_candidate_context(symbol)
        gate_passed, gate_reason = passes_deterministic_candidate_gate(
            symbol=symbol,
            positions_state=positions_state,
            features=features,
            universe_context=universe_context,
        )
        if not gate_passed:
            return self._build_gated_hold_decision(
                features=features,
                portfolio_state=portfolio_state,
                gate_reason=gate_reason,
                t_total_start=t_total_start,
            )
        advisory_bundle = build_advisory_bundle(
            symbol=symbol,
            features=features,
            universe_context=universe_context,
        )
        reflex = advisory_bundle.reflex
        market_state_review = advisory_bundle.market_state_review
        candidate_review = advisory_bundle.candidate_review
        posture_review = advisory_bundle.posture_review
        visual_review = advisory_bundle.visual_review
        phi3_ms = float(advisory_bundle.timings.get("phi3_ms", 0.0))
        advisory_ms = float(advisory_bundle.timings.get("advisory_ms", 0.0))
        entry_recommendation = str(features.get("entry_recommendation", "")).upper()
        buy_candidate = entry_recommendation in {"BUY", "STRONG_BUY"}
        if candidate_review.get("reason") == "light_advisory_pre_filter" and not buy_candidate:
            return self._build_filtered_hold_decision(
                features=features,
                portfolio_state=portfolio_state,
                reflex=reflex,
                market_state_review=market_state_review,
                candidate_review=candidate_review,
                posture_review=posture_review,
                reason="light_advisory_pre_filter",
                phi3_ms=phi3_ms,
                t_total_start=t_total_start,
            )
        if (
            candidate_review.get("promotion_decision") == "demote"
            and candidate_review.get("action_bias") == "hold_preferred"
            and str(candidate_review.get("reason", "")).strip().lower() == "weak_or_risky_setup"
            and not buy_candidate
        ):
            return self._build_filtered_hold_decision(
                features=features,
                portfolio_state=portfolio_state,
                reflex=reflex,
                market_state_review=market_state_review,
                candidate_review={
                    "promotion_decision": "demote",
                    "priority": candidate_review.get("priority", 0.2),
                    "action_bias": "hold_preferred",
                    "reason": "weak_or_risky_setup_filtered",
                },
                posture_review={
                    "posture": posture_review.get("posture", "neutral"),
                    "promotion_bias": posture_review.get("promotion_bias", "normal"),
                    "exit_bias": posture_review.get("exit_bias", "standard"),
                    "size_bias": posture_review.get("size_bias", "normal"),
                    "reason": "weak_or_risky_setup_filtered",
                },
                reason="weak_or_risky_setup_filtered",
                phi3_ms=phi3_ms,
                t_total_start=t_total_start,
            )
        if not bool(features.get("indicators_ready", True)):
            integrated_reflex = {
                "action": "HOLD",
                "reason": "nemotron_skipped_indicator_warmup",
                "debug": {},
                "override_signal": None,
                "size_factor_hint": 1.0,
                "reflex": reflex.get("reflex", "allow"),
                "micro_state": reflex.get("micro_state", "unknown"),
                "visual_review": visual_review,
                "market_state_review": market_state_review,
                "posture_review": posture_review,
                "candidate_review": candidate_review,
            }
            signal = strategy_decision(self.strategy, features, reflex_decision=integrated_reflex)
            risk_checks = risk_adjust(
                self.risk_engine,
                signal=signal,
                features=features,
                portfolio_state=portfolio_state,
            )
            execution = {
                "status": "no_trade",
                "bar_ts": features.get("bar_ts"),
                "bar_idx": features.get("bar_idx"),
                "reflex": reflex,
                "nemotron": integrated_reflex,
            }
            timings = {
                "phi3_ms": round(phi3_ms, 2),
                "advisory_ms": round(advisory_ms, 2),
                "nemotron_ms": 0.0,
                "execution_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_total_start) * 1000.0, 2),
            }
            return NemotronDecision(
                signal=signal,
                risk_checks=risk_checks,
                portfolio_decision={"decision": "allow", "size_factor": 1.0, "reasons": []},
                execution=execution,
                reflex={"phi3": reflex, "nemotron": integrated_reflex},
                timings=timings,
            )
        if not should_run_nemotron(
            symbol=symbol,
            features=features,
            positions_state=positions_state,
            universe_context=universe_context,
            candidate_review=candidate_review,
        ):
            return self._build_filtered_hold_decision(
                features=features,
                portfolio_state=portfolio_state,
                reflex=reflex,
                market_state_review=market_state_review,
                candidate_review=candidate_review,
                posture_review=posture_review,
                reason="phi3_candidate_manager",
                phi3_ms=phi3_ms,
                t_total_start=t_total_start,
            )
        fingerprint = self._decision_fingerprint(
            symbol=symbol,
            features=features,
            portfolio_state=portfolio_state,
            positions_state=positions_state,
        )
        cached_decision = self._get_cached_decision(symbol=symbol, fingerprint=fingerprint)
        if cached_decision is not None:
            return cached_decision
        client = NemotronClient(
            strategy=self.strategy,
            risk=self.risk_engine,
            portfolio_config=self.portfolio_config,
            execution=self.executor,
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            symbols=symbols,
        )
        payload = self._build_payload(
            symbol=symbol,
            symbols=symbols,
            features=features,
            reflex=reflex,
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            candidate_review=candidate_review,
            market_state_review=market_state_review,
            posture_review=posture_review,
            visual_review=visual_review,
            universe_context=universe_context,
        )
        self._log_event(
            {
                "event": "nemotron_decide_start",
                "symbol": symbol,
                "reflex": reflex,
                **({"payload": payload} if NEMOTRON_DEBUG_VERBOSE else {}),
            }
        )
        nemotron_ms = 0.0
        try:
            t_nemo_start = time.perf_counter()
            client.set_features(features)
            parsed = run_nemotron_tool_loop(
                payload,
                system=NEMOTRON_STRATEGIST_SYSTEM_PROMPT,
                tool_registry=client.tools,
            )
            nemotron_ms = (time.perf_counter() - t_nemo_start) * 1000.0
            self._log_event(
                {
                    "event": "nemotron_decide_success",
                    "symbol": symbol,
                    "parsed": parsed,
                }
            )
            final_decision = parsed.get("final_decision", {})
            integrated_reflex = {
                **self._validate_final_decision(
                    symbol=symbol,
                    final_decision=final_decision,
                    proposed_weight=proposed_weight,
                ),
                "reflex": reflex.get("reflex", "allow"),
                "micro_state": reflex.get("micro_state", "unknown"),
                "visual_review": visual_review,
                "candidate_review": candidate_review,
                "market_state_review": market_state_review,
                "posture_review": posture_review,
            }
            if (
                integrated_reflex.get("action") == "OPEN"
                and candidate_review.get("action_bias") == "hold_preferred"
                and str(features.get("entry_recommendation", "")).upper() == "WATCH"
            ):
                integrated_reflex = {
                    "action": "HOLD",
                    "reason": "candidate_review_hold_preferred",
                    "debug": {"candidate_review": candidate_review},
                    "override_signal": "FLAT",
                    "size_factor_hint": 1.0,
                    "reflex": reflex.get("reflex", "allow"),
                    "micro_state": reflex.get("micro_state", "unknown"),
                    "visual_review": visual_review,
                    "candidate_review": candidate_review,
                    "market_state_review": market_state_review,
                    "posture_review": posture_review,
                }
            elif integrated_reflex.get("action") == "OPEN" and candidate_review.get("action_bias") == "reduce_size":
                integrated_reflex["size_factor_hint"] = float(integrated_reflex.get("size_factor_hint", 1.0)) * 0.5
            if integrated_reflex.get("action") == "OPEN":
                size_factor_hint = float(integrated_reflex.get("size_factor_hint", 1.0))
                if posture_review.get("size_bias") == "reduce":
                    integrated_reflex["size_factor_hint"] = size_factor_hint * 0.75
                elif posture_review.get("size_bias") == "increase":
                    integrated_reflex["size_factor_hint"] = min(size_factor_hint * 1.1, 2.0)
        except Exception as exc:
            self._log_fallback(payload, exc)
            integrated_reflex = self._integrate_reflex(features, reflex)
            filtered_candidate_review = candidate_review
            filtered_posture_review = posture_review
            filtered_visual_review = visual_review
            if (
                candidate_review.get("promotion_decision") == "demote"
                and candidate_review.get("action_bias") == "hold_preferred"
            ):
                filtered_candidate_review = {
                    "promotion_decision": "demote",
                    "priority": candidate_review.get("priority", 0.2),
                    "action_bias": "hold_preferred",
                    "reason": "fallback_phi3_only",
                }
                filtered_posture_review = {
                    "posture": posture_review.get("posture", "neutral"),
                    "promotion_bias": posture_review.get("promotion_bias", "normal"),
                    "exit_bias": posture_review.get("exit_bias", "standard"),
                    "size_bias": posture_review.get("size_bias", "normal"),
                    "reason": "fallback_phi3_only",
                }
                filtered_visual_review = {}
                advisory_ms = 0.0
            integrated_reflex["candidate_review"] = filtered_candidate_review
            integrated_reflex["visual_review"] = filtered_visual_review
            integrated_reflex["market_state_review"] = market_state_review
            integrated_reflex["posture_review"] = filtered_posture_review
        signal = strategy_decision(self.strategy, features, reflex_decision=integrated_reflex)
        if self._reflex_is_hard_block(reflex):
            signal = "FLAT"
        risk_checks = risk_adjust(
            self.risk_engine,
            signal=signal,
            features=features,
            portfolio_state=portfolio_state,
        )
        portfolio_decision = portfolio_evaluate(
            config=self.portfolio_config,
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

        if self._reflex_is_hard_block(reflex):
            execution = {
                "status": "blocked",
                "reason": [reflex.get("reason", reflex.get("reflex"))],
                "bar_ts": features.get("bar_ts"),
                "bar_idx": features.get("bar_idx"),
            }
            execution_ms = 0.0
        elif "block" in risk_checks:
            execution = {
                "status": "blocked",
                "reason": risk_checks,
                "bar_ts": features.get("bar_ts"),
                "bar_idx": features.get("bar_idx"),
            }
            execution_ms = 0.0
        else:
            t_execution_start = time.perf_counter()
            execution = execution_place_order(
                self.executor,
                signal=signal,
                symbol=symbol,
                features=features,
                portfolio_state=portfolio_state,
                size_factor=portfolio_decision["size_factor"] * integrated_reflex.get("size_factor_hint", 1.0),
            )
            execution_ms = (time.perf_counter() - t_execution_start) * 1000.0
            if portfolio_decision["decision"] == "scale_down":
                execution["size_factor"] = portfolio_decision["size_factor"]
        execution["reflex"] = reflex
        execution["nemotron"] = integrated_reflex
        timings = {
            "phi3_ms": round(phi3_ms, 2),
            "advisory_ms": round(advisory_ms, 2),
            "nemotron_ms": round(nemotron_ms, 2),
            "execution_ms": round(execution_ms, 2),
            "total_ms": round((time.perf_counter() - t_total_start) * 1000.0, 2),
        }
        decision = NemotronDecision(
            signal=signal,
            risk_checks=risk_checks,
            portfolio_decision=portfolio_decision,
            execution=execution,
            reflex={"phi3": reflex, "nemotron": integrated_reflex},
            timings=timings,
        )
        self._store_cached_decision(symbol=symbol, fingerprint=fingerprint, decision=decision)
        return decision
