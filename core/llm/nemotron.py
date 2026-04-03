from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from core.config.runtime import get_runtime_setting
from core.execution.cpp_exec import CppExecutor
from core.llm.client import (
    NEMOTRON_DEBUG_VERBOSE,
    nemotron_chat,
    nemotron_provider_name,
    parse_json_response,
    sanitize_for_json,
)
from core.llm.contracts import (
    TRADE_ACTION_CLOSE,
    TRADE_ACTION_HOLD,
    TRADE_ACTION_OPEN,
    TRADE_ACTION_ROTATE,
    TRADE_ACTION_SCALE_IN,
    TRADE_ACTION_SCALE_OUT,
    TRADE_ACTION_SKIP,
    TRADE_ACTION_TIGHTEN,
    TRADE_ACTION_WATCH,
    normalize_trade_action,
    normalize_trade_reviewer_output,
)
from core.llm.orchestrator import build_advisory_bundle
from core.llm.prompts import get_nemotron_batch_strategist_prompt, get_nemotron_strategist_system_prompt
from core.memory.trade_memory import TradeMemoryStore
from core.policy.nemotron_gate import (
    _get_leader_metrics,
    load_universe_candidate_context,
    passes_deterministic_candidate_gate,
)
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState
from core.runtime.tools import execution_place_order, portfolio_evaluate, risk_adjust, strategy_decision
from core.runtime.log_rotation import rotate_jsonl_if_needed
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy

NEMOTRON_EVENT_LOG = os.getenv("NEMOTRON_EVENT_LOG", "false").strip().lower() in {"1", "true", "yes", "on"}
NEMOTRON_FALLBACK_LOG_DEDUPE_SEC = float(os.getenv("NEMOTRON_FALLBACK_LOG_DEDUPE_SEC", "60") or 60.0)


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
        self._memory_store = TradeMemoryStore()
        self._last_fallback_log_ts_by_error: dict[str, float] = {}

    def _build_portfolio_summary(
        self,
        *,
        portfolio_state: PortfolioState,
        positions_state: PositionState,
        current_price: float,
        current_symbol: str,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        open_count = positions_state.count()
        max_positions = self.portfolio_config.max_open_positions
        equity = portfolio_state.cash
        for sym, qty in (portfolio_state.positions or {}).items():
            mark = (portfolio_state.position_marks or {}).get(sym, 0.0)
            equity += float(qty) * float(mark)

        pos_details = []
        for p in positions_state.all():
            entry_price = p.entry_price or 0.0
            price = current_price if p.symbol == current_symbol else 0.0
            pnl_pct = None
            if entry_price > 0 and price > 0:
                if p.side == "LONG":
                    pnl_pct = round(((price / entry_price) - 1.0) * 100.0, 2)
                else:
                    pnl_pct = round(((entry_price / price) - 1.0) * 100.0, 2)
            hold_minutes = None
            if p.entry_bar_ts:
                try:
                    entry_dt = datetime.fromisoformat(str(p.entry_bar_ts).replace("Z", "+00:00"))
                    hold_minutes = round((now - entry_dt).total_seconds() / 60.0, 1)
                except Exception:
                    pass
            pos_details.append({
                "symbol": p.symbol,
                "side": p.side,
                "lane": p.lane or "L3",
                "entry_price": round(entry_price, 6),
                "stop_loss": round(p.stop_loss or 0.0, 6),
                "take_profit": round(p.take_profit or 0.0, 6),
                "pnl_pct": pnl_pct,
                "hold_minutes": hold_minutes,
                "exit_posture": p.exit_posture,
            })

        return {
            "open_positions_count": open_count,
            "max_open_positions": max_positions,
            "open_slots": max(0, max_positions - open_count),
            "equity_usd": round(equity, 2),
            "cash_usd": round(portfolio_state.cash, 2),
            "risk_per_trade_pct": float(get_runtime_setting("EXEC_RISK_PER_TRADE_PCT")),
            "positions": pos_details,
        }

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

    def _build_model_failure_fallback(
        self,
        *,
        symbol: str,
        features: dict[str, Any],
        positions_state: PositionState,
        universe_context: dict[str, Any],
        reflex: dict[str, Any],
    ) -> dict[str, Any]:
        integrated_reflex = self._integrate_reflex(features, reflex)
        passed_gate, gate_reason = passes_deterministic_candidate_gate(
            symbol=symbol,
            positions_state=positions_state,
            features=features,
            universe_context=universe_context,
        )
        if not passed_gate:
            integrated_reflex["debug"] = {"fallback_gate_reason": gate_reason}
            return integrated_reflex

        entry_recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
        reversal_risk = str(features.get("reversal_risk", "MEDIUM") or "MEDIUM").upper()
        if entry_recommendation not in {"BUY", "STRONG_BUY"} or reversal_risk == "HIGH":
            integrated_reflex["debug"] = {"fallback_gate_reason": gate_reason}
            return integrated_reflex

        return {
            "action": "OPEN",
            "reason": "model_parse_failed_deterministic_open",
            "debug": {"fallback_gate_reason": gate_reason},
            "override_signal": "LONG",
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
        action, normalized_fields = normalize_trade_action(final_decision.get("action", TRADE_ACTION_HOLD))
        raw_side = final_decision.get("side")
        side = str(raw_side).upper() if raw_side is not None else None
        if side == "BUY":
            side = "LONG"
        elif side == "SELL":
            side = "SHORT"
        reason = str(final_decision.get("reason", "nemotron_integrated"))
        debug = final_decision.get("debug", {}) if isinstance(final_decision.get("debug", {}), dict) else {}
        if normalized_fields:
            debug = dict(debug)
            debug["normalized_fields"] = sorted(set([*(debug.get("normalized_fields", []) if isinstance(debug.get("normalized_fields"), list) else []), *normalized_fields]))
        decision_symbol = str(final_decision.get("symbol", symbol) or symbol)
        # Model sometimes echoes a prompt placeholder or few-shot example symbol — treat as current symbol
        if decision_symbol in {"SYMBOL", "<current_symbol>", "CURRENT_SYMBOL", "LINK/USD", "BTC/USD", "ETH/USD"}:
            decision_symbol = symbol

        if decision_symbol != symbol and action != "HOLD":
            return {
                "action": "HOLD",
                "reason": "invalid_final_decision_contract",
                "debug": {"contract_error": "symbol_mismatch", "raw_symbol": decision_symbol},
                "override_signal": "FLAT",
                "size_factor_hint": 1.0,
            }
        if action == TRADE_ACTION_OPEN:
            if side not in {"LONG", "SHORT"}:
                return {
                    "action": TRADE_ACTION_HOLD,
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
                    "action": TRADE_ACTION_HOLD,
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
        if action in {TRADE_ACTION_WATCH, TRADE_ACTION_TIGHTEN, TRADE_ACTION_SCALE_IN, TRADE_ACTION_SCALE_OUT, TRADE_ACTION_ROTATE, TRADE_ACTION_SKIP}:
            return {
                "action": TRADE_ACTION_HOLD,
                "reason": reason,
                "debug": {**debug, "advisory_action": action},
                "override_signal": "FLAT",
                "size_factor_hint": 1.0,
            }
        return {
            "action": action,
            "reason": reason,
            "debug": debug,
            "override_signal": "FLAT" if action in {TRADE_ACTION_HOLD, TRADE_ACTION_CLOSE} else side,
            "size_factor_hint": 1.0,
        }

    def _phi_chart_support_level(self, *, features: dict[str, Any], market_state_review: dict[str, Any]) -> str:
        market_state = str(market_state_review.get("market_state", "transition") or "transition")
        lane_bias = str(market_state_review.get("lane_bias", "favor_selective") or "favor_selective")
        market_reason = str(market_state_review.get("reason", "") or "")
        trend_stage = str(market_state_review.get("trend_stage", "unclear") or "unclear")
        breakout_state = str(market_state_review.get("breakout_state", "unclear") or "unclear")
        volume_confirmation = str(market_state_review.get("volume_confirmation", "neutral") or "neutral")
        late_move_risk = str(market_state_review.get("late_move_risk", "moderate") or "moderate")
        pullback_quality = str(market_state_review.get("pullback_quality", "unclear") or "unclear")
        score = float(features.get("entry_score", 0.0) or 0.0)
        trend_confirmed = bool(features.get("trend_confirmed", False))
        recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
        if (
            market_state == "trending"
            and lane_bias == "favor_trend"
            and trend_stage in {"early", "emerging", "confirmed"}
            and breakout_state in {"fresh_breakout", "retest_holding"}
            and volume_confirmation == "supportive"
            and late_move_risk != "extended"
            and trend_confirmed
            and recommendation in {"BUY", "STRONG_BUY"}
            and score >= 65.0
        ):
            return "strong"
        if (
            market_state == "ranging"
            and lane_bias == "favor_selective"
            and market_reason == "range_breakout_attempt_with_ema_support"
            and breakout_state in {"fresh_breakout", "retest_holding", "breakout_attempt", "unclear"}
            and volume_confirmation in {"supportive", "neutral"}
            and late_move_risk != "extended"
            and recommendation in {"BUY", "STRONG_BUY"}
            and score >= 68.0
        ):
            return "strong"
        if (
            market_state in {"trending", "transition"}
            and trend_stage in {"early", "emerging", "confirmed"}
            and breakout_state in {"fresh_breakout", "retest_holding", "breakout_attempt"}
            and volume_confirmation in {"supportive", "neutral"}
            and late_move_risk != "extended"
            and (pullback_quality in {"clean_retest", "higher_low_support"} or recommendation in {"BUY", "STRONG_BUY"})
        ):
            return "medium"
        return "weak"

    def _phi_requires_explicit_override(self, *, features: dict[str, Any], market_state_review: dict[str, Any]) -> bool:
        pattern_explanation = market_state_review.get("pattern_explanation", {}) if isinstance(market_state_review.get("pattern_explanation", {}), dict) else {}
        candle_evidence = market_state_review.get("candle_evidence", {}) if isinstance(market_state_review.get("candle_evidence", {}), dict) else {}
        recommended = pattern_explanation.get("recommended_nemo_interpretation", {}) if isinstance(pattern_explanation.get("recommended_nemo_interpretation", {}), dict) else {}

        recommendation = str(features.get("entry_recommendation", "WATCH") or "WATCH").upper()
        reversal_risk = str(features.get("reversal_risk", "MEDIUM") or "MEDIUM").upper()
        structure_validity = str(pattern_explanation.get("structure_validity", "unclear") or "unclear")
        structure_confidence = float(pattern_explanation.get("structure_confidence", 0.0) or 0.0)
        prefer_action = str(recommended.get("prefer_action", "HOLD") or "HOLD").upper()
        skepticism_penalty = int(recommended.get("skepticism_penalty", 0) or 0)
        candle_confirmation = float(candle_evidence.get("confirmation_score", 0.0) or 0.0)
        candle_bias = str(candle_evidence.get("candle_bias", "neutral") or "neutral")
        late_move_risk = str(market_state_review.get("late_move_risk", "moderate") or "moderate")
        breakout_state = str(market_state_review.get("breakout_state", "unclear") or "unclear")
        score = float(features.get("entry_score", 0.0) or 0.0)

        return (
            recommendation in {"BUY", "STRONG_BUY"}
            and reversal_risk != "HIGH"
            and prefer_action == "OPEN"
            and structure_validity == "valid"
            and structure_confidence >= 0.6
            and candle_confirmation >= 0.65
            and candle_bias != "bearish"
            and skepticism_penalty <= 1
            and late_move_risk != "extended"
            and breakout_state in {"fresh_breakout", "retest_holding", "breakout_attempt"}
            and score >= 58.0
        )

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
        candidate_packet: dict[str, Any],
        reflex: dict[str, Any],
        portfolio_state: PortfolioState,
        positions_state: PositionState,
        market_state_review: dict[str, Any] | None = None,
        universe_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from core.policy.candidate_packet import build_local_nemotron_candidate_packet

        return sanitize_for_json(
            {
                "candidate": build_local_nemotron_candidate_packet(candidate_packet),
                "reflex": reflex,
                "portfolio_summary": self._build_portfolio_summary(
                    portfolio_state=portfolio_state,
                    positions_state=positions_state,
                    current_price=float(candidate_packet.get("price") or 0.0),
                    current_symbol=str(candidate_packet.get("symbol") or ""),
                ),
                "universe_context": universe_context or self._load_universe_candidate_context(str(candidate_packet.get("symbol") or "")),
                "market_state_review": market_state_review or {},
            }
        )

    def _build_filtered_hold_decision(
        self,
        *,
        features: dict[str, Any],
        portfolio_state: PortfolioState,
        reflex: dict[str, Any],
        market_state_review: dict[str, Any],
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
        error_text = str(exc)
        now = time.time()
        last_logged_at = self._last_fallback_log_ts_by_error.get(error_text, 0.0)
        if (now - last_logged_at) < NEMOTRON_FALLBACK_LOG_DEDUPE_SEC:
            return
        self._last_fallback_log_ts_by_error[error_text] = now
        path = Path("logs/nemotron_debug.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rotate_jsonl_if_needed(path)
        event = {
            "event": "nemotron_fallback",
            "error": error_text,
        }
        if NEMOTRON_DEBUG_VERBOSE:
            event["payload"] = payload
        event = sanitize_for_json(event)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _log_event(self, event: dict[str, Any]) -> None:
        if not NEMOTRON_EVENT_LOG:
            return
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
        from core.policy.candidate_packet import build_candidate_packet

        t_total_start = time.perf_counter()
        universe_context = load_universe_candidate_context(symbol)
        advisory_bundle = build_advisory_bundle(
            symbol=symbol,
            features=features,
            universe_context=universe_context,
        )
        reflex = advisory_bundle.reflex
        market_state_review = advisory_bundle.market_state_review
        visual_review = advisory_bundle.visual_review
        phi3_ms = float(advisory_bundle.timings.get("phi3_ms", 0.0))
        advisory_ms = float(advisory_bundle.timings.get("advisory_ms", 0.0))
        # All pre-gates disabled — once a symbol reaches single-symbol strategist review,
        # Nemo should make the call instead of being short-circuited by legacy advisory gating.
        leader_urgency, leader_takeover = _get_leader_metrics(symbol, universe_context)
        lane = str(features.get("lane", "L3") or "L3").upper()
        fingerprint = self._decision_fingerprint(
            symbol=symbol,
            features=features,
            portfolio_state=portfolio_state,
            positions_state=positions_state,
        )
        cached_decision = self._get_cached_decision(symbol=symbol, fingerprint=fingerprint)
        if cached_decision is not None:
            return cached_decision
        lesson_summary = self._memory_store.build_symbol_lesson_block(symbol)
        behavior_score = self._memory_store.build_behavior_score_block(lookback=50) or None
        candidate_packet = build_candidate_packet(
            features=features,
            positions_state=positions_state,
            portfolio_state=portfolio_state,
            lesson_summary=lesson_summary,
            behavior_score=behavior_score,
        )
        payload = self._build_payload(
            candidate_packet=candidate_packet,
            reflex=reflex,
            portfolio_state=portfolio_state,
            positions_state=positions_state,
            market_state_review=market_state_review,
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
            raw = nemotron_chat(
                payload,
                system=get_nemotron_strategist_system_prompt(),
            )
            parsed = parse_json_response(raw)
            nemotron_ms = (time.perf_counter() - t_nemo_start) * 1000.0
            self._log_event(
                {
                    "event": "nemotron_decide_success",
                    "symbol": symbol,
                    "parsed": parsed,
                }
            )
            parsed = normalize_trade_reviewer_output(parsed, symbol=symbol)
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
                "market_state_review": market_state_review,
            }
        except Exception as exc:
            self._log_fallback(payload, exc)
            integrated_reflex = self._build_model_failure_fallback(
                symbol=symbol,
                features=features,
                positions_state=positions_state,
                universe_context=universe_context,
                reflex=reflex,
            )
            filtered_visual_review = visual_review
            filtered_visual_review = {}
            advisory_ms = 0.0
            integrated_reflex["visual_review"] = filtered_visual_review
            integrated_reflex["market_state_review"] = market_state_review
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
        elif portfolio_decision["decision"] == "replace":
            execution = {
                "status": "no_trade",
                "reason": ["replacement_required"],
                "replace_symbol": portfolio_decision.get("replace_symbol"),
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

    def batch_decide(
        self,
        *,
        candidates: list[dict[str, Any]],
        portfolio_state: "PortfolioState",
        positions_state: PositionState,
        symbols: list[str],
        market_state_summary: str = "",
    ) -> "dict[str, NemotronDecision]":
        """Single Nemo call for all candidates — comparative ranking instead of 15 serial calls."""
        if not candidates:
            return {}

        t_start = time.perf_counter()
        open_count = positions_state.count()
        max_positions = self.portfolio_config.max_open_positions
        open_slots = max(0, max_positions - open_count)
        cash = float(portfolio_state.cash)
        risk_pct = float(get_runtime_setting("EXEC_RISK_PER_TRADE_PCT"))
        max_trade = cash * risk_pct / 100.0

        positions_lines = [
            f"  HOLDING {sym}"
            for sym, qty in (portfolio_state.positions or {}).items()
            if float(qty or 0) > 0
        ]
        positions_block = "\n".join(positions_lines) if positions_lines else "  (none)"

        rows = []
        for c in candidates:
            sym = c["symbol"]
            f = c["features"]
            phi3 = str(c.get("reflex", {}).get("reflex", "allow"))
            market_state_review = c.get("market_state_review", {}) if isinstance(c.get("market_state_review", {}), dict) else {}
            lane_supervision = c.get("lane_supervision", {}) if isinstance(c.get("lane_supervision", {}), dict) else {}
            pattern_candidate = f.get("pattern_candidate", {}) if isinstance(f.get("pattern_candidate", {}), dict) else {}
            pattern_verification = f.get("pattern_verification", {}) if isinstance(f.get("pattern_verification", {}), dict) else {}
            lane = str(f.get("lane", "L3") or "L3")
            score = float(f.get("entry_score", 0.0) or 0.0)
            rec = str(f.get("entry_recommendation", "WATCH") or "WATCH")[:10]
            risk = str(f.get("reversal_risk", "MEDIUM") or "MEDIUM")[:6]
            m5 = float(f.get("momentum_5", 0.0) or 0.0) * 100.0
            rsi = float(f.get("rsi", 50.0) or 50.0)
            vol = float(f.get("volume_ratio", 1.0) or 1.0)
            macd_h = float(f.get("macd_hist", 0.0) or 0.0)
            adx = float(f.get("adx", 0.0) or 0.0)
            trend_ok = bool(f.get("trend_confirmed", False))
            ranging = bool(f.get("ranging_market", False))
            structure_quality = float(f.get("structure_quality", 0.0) or 0.0)
            trade_quality = float(f.get("trade_quality", 0.0) or 0.0)
            risk_quality = float(f.get("risk_quality", 0.0) or 0.0)
            ema9 = bool(f.get("ema9_above_ema20", False))
            brk = bool(f.get("range_breakout_1h", False))
            pb = bool(f.get("pullback_hold", False))
            hl = int(f.get("higher_low_count", 0) or 0)
            vs = float(f.get("volume_surge", 0.0) or 0.0) * 100.0
            m14 = float(f.get("momentum_14", 0.0) or 0.0) * 100.0
            rot = float(f.get("rotation_score", 0.0) or 0.0)
            _pb = f.get("point_breakdown") or {}
            net_edge = float(_pb.get("net_edge_pct", 0.0) or 0.0)
            cost_penalty = float(_pb.get("cost_penalty_pts", 0.0) or 0.0)
            pattern_name = str(pattern_candidate.get("pattern", "none") or "none")
            pattern_validity = str(pattern_verification.get("validity", "unclear") or "unclear")
            pattern_quality = float(pattern_verification.get("pattern_quality_score", 0.0) or 0.0)
            extension_risk = float(pattern_verification.get("extension_risk_score", 0.0) or 0.0)
            market_state = str(market_state_review.get("market_state", "transition") or "transition")
            lane_bias = str(market_state_review.get("lane_bias", "favor_selective") or "favor_selective")
            market_confidence = float(market_state_review.get("confidence", 0.0) or 0.0)
            breakout_state = str(market_state_review.get("breakout_state", "unclear") or "unclear")
            trend_stage = str(market_state_review.get("trend_stage", "unclear") or "unclear")
            volume_confirmation = str(market_state_review.get("volume_confirmation", "neutral") or "neutral")
            pullback_quality = str(market_state_review.get("pullback_quality", "unclear") or "unclear")
            late_move_risk = str(market_state_review.get("late_move_risk", "moderate") or "moderate")
            pattern_explanation = market_state_review.get("pattern_explanation", {}) if isinstance(market_state_review.get("pattern_explanation", {}), dict) else {}
            candle_evidence = market_state_review.get("candle_evidence", {}) if isinstance(market_state_review.get("candle_evidence", {}), dict) else {}
            structure_pattern = str(pattern_explanation.get("structure_pattern", "none") or "none")
            structure_validity = str(pattern_explanation.get("structure_validity", "unclear") or "unclear")
            structure_confidence = float(pattern_explanation.get("structure_confidence", 0.0) or 0.0)
            recommended_interpretation = pattern_explanation.get("recommended_nemo_interpretation", {}) if isinstance(pattern_explanation.get("recommended_nemo_interpretation", {}), dict) else {}
            pattern_prefer_action = str(recommended_interpretation.get("prefer_action", "HOLD") or "HOLD")
            structure_bonus = int(recommended_interpretation.get("structure_bonus", 0) or 0)
            skepticism_penalty = int(recommended_interpretation.get("skepticism_penalty", 0) or 0)
            primary_candle = str(candle_evidence.get("primary_candle", "none") or "none")
            candle_bias = str(candle_evidence.get("candle_bias", "neutral") or "neutral")
            candle_strength = float(candle_evidence.get("candle_strength", 0.0) or 0.0)
            candle_confirmation_score = float(candle_evidence.get("confirmation_score", 0.0) or 0.0)
            lane_candidate = str(lane_supervision.get("lane_candidate", "") or "")
            lane_conflict = bool(lane_supervision.get("lane_conflict", False))
            universe_lane = str(lane_supervision.get("universe_lane", "") or "")
            rows.append(
                f"{sym:<14}|{lane:<3}|score:{score:>4.0f}|{rec:<10}|{risk:<6}"
                f"|m5:{m5:>+5.1f}%|m14:{m14:>+5.1f}%|rsi:{rsi:>4.0f}|vol:{vol:>4.1f}x|vs:{vs:>4.1f}%"
                f"|macd_h:{macd_h:>+7.4f}|adx:{adx:>4.0f}|rot:{rot:>4.2f}"
                f"|sq:{structure_quality:>4.0f}|tq:{trade_quality:>4.0f}|rq:{risk_quality:>4.0f}"
                f"|trend:{'Y' if trend_ok else 'N'}|chop:{'Y' if ranging else 'N'}"
                f"|ema:{'Y' if ema9 else 'N'}|brk:{'Y' if brk else 'N'}|pb:{'Y' if pb else 'N'}|hl:{hl}"
                f"|net_edge:{net_edge:>+5.2f}%|cost_pen:{cost_penalty:>+4.0f}pts"
                f"|phi3:{phi3}|mkt:{market_state}|bias:{lane_bias}|mkt_cf:{market_confidence:>3.2f}"
                f"|sp:{structure_pattern}|sval:{structure_validity}|scf:{structure_confidence:>3.2f}"
                f"|brk_state:{breakout_state}|trend_stage:{trend_stage}|pbq:{pullback_quality}|vol_cf:{volume_confirmation}|late:{late_move_risk}"
                f"|nemo_act:{pattern_prefer_action}|sbonus:{structure_bonus}|skep:{skepticism_penalty}"
                f"|candle:{primary_candle}|cbias:{candle_bias}|cstr:{candle_strength:>3.2f}|ccf:{candle_confirmation_score:>3.2f}"
                f"|lane_sup:{lane_candidate or '-'}|lane_conflict:{'Y' if lane_conflict else 'N'}|univ_lane:{universe_lane or '-'}"
                f"|pat:{pattern_name}|pver:{pattern_validity}|pqs:{pattern_quality:>3.2f}|ext:{extension_risk:>3.2f}"
            )
        table = "\n".join(rows)

        payload = {
            "open_slots": open_slots,
            "max_positions": max_positions,
            "cash_usd": round(cash, 2),
            "risk_pct": round(risk_pct, 1),
            "max_trade_usd": round(max_trade, 2),
            "open_positions": positions_block,
            "market_context": market_state_summary or "unknown",
            "candidates": table,
        }

        # Inject learned lessons so Nemo improves from past trades
        lessons_block = self._memory_store.build_lessons_block(max_lessons=8)
        if lessons_block:
            payload["learned_lessons"] = lessons_block

        # Inject AI behavior score so Nemo self-calibrates entry strictness
        behavior_block = self._memory_store.build_behavior_score_block(lookback=50)
        if behavior_block:
            payload["behavior_score"] = behavior_block

        t_nemo = time.perf_counter()
        reasoning = ""
        decision_map: dict[str, dict[str, Any]] = {}
        batch_parse_failed = False
        raw = ""

        def _safe_batch_model_failure_decision(symbol: str, features: dict[str, Any], reflex: dict[str, Any]) -> dict[str, Any]:
            fallback = self._build_model_failure_fallback(
                symbol=symbol,
                features=features,
                positions_state=positions_state,
                universe_context=load_universe_candidate_context(symbol),
                reflex=reflex,
            )
            if str(fallback.get("action", "HOLD")).upper() == "OPEN":
                return {
                    "action": "OPEN",
                    "side": "LONG",
                    "reason": str(fallback.get("reason", "model_parse_failed_deterministic_open")),
                    "size": max_trade,
                    "debug": fallback.get("debug", {}),
                }
            return {
                "action": "HOLD",
                "reason": "batch_parse_fallback_hold",
                "size": 0,
                "debug": fallback.get("debug", {}),
            }

        def _extract_batch_decisions(parsed: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
            reasoning_text = str(parsed.get("reasoning", ""))
            decisions = parsed.get("decisions")
            if isinstance(decisions, list):
                return reasoning_text, [d for d in decisions if isinstance(d, dict)]

            tool_result = parsed.get("tool_result")
            if isinstance(tool_result, dict):
                nested_result = tool_result.get("result")
                if isinstance(nested_result, dict):
                    nested_reasoning = str(nested_result.get("reasoning", reasoning_text))
                    nested_decisions = nested_result.get("decisions")
                    if isinstance(nested_decisions, list):
                        return nested_reasoning, [d for d in nested_decisions if isinstance(d, dict)]

            args = parsed.get("args")
            if isinstance(args, dict):
                nested_reasoning = str(args.get("reasoning", reasoning_text))
                nested_decisions = args.get("decisions")
                if isinstance(nested_decisions, list):
                    return nested_reasoning, [d for d in nested_decisions if isinstance(d, dict)]

            raise ValueError("batch response missing decisions list")

        def _deterministic_batch_fallback(features: dict[str, Any]) -> dict[str, Any]:
            # Never open positions when Nemo fails JSON — only real Nemo decisions may open trades.
            # The old deterministic OPEN path was causing entries during model failures/downtrends.
            return {
                "action": "HOLD",
                "reason": "batch_parse_fallback_hold",
                "size": 0,
            }
        try:
            batch_max_tokens = max(600, int(os.getenv("NEMOTRON_BATCH_MAX_TOKENS", "1800") or 1800))
            raw = nemotron_chat(
                payload,
                system=get_nemotron_batch_strategist_prompt(),
                max_tokens=batch_max_tokens,
            )
            nemo_ms = (time.perf_counter() - t_nemo) * 1000.0
            # Strip chain-of-thought prefix — find the JSON object
            parsed = parse_json_response(raw)
            reasoning, parsed_decisions = _extract_batch_decisions(parsed)
            for d in parsed_decisions:
                sym = str(d.get("symbol", ""))
                if sym:
                    decision_map[sym] = d
        except Exception as exc:
            nemo_ms = (time.perf_counter() - t_nemo) * 1000.0
            reasoning = f"batch_error:{exc}"
            batch_parse_failed = True
            self._log_event(
                {
                    "event": "nemotron_batch_failure",
                    "provider": nemotron_provider_name(),
                    "candidate_count": len(candidates),
                    "raw_response": raw,
                    "error": str(exc),
                }
            )

        if (
            batch_parse_failed
            and nemotron_provider_name() == "local"
            and os.getenv("NEMOTRON_LOCAL_BATCH_SINGLE_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}
        ):
            return {
                c["symbol"]: self.decide(
                    symbol=c["symbol"],
                    features=c["features"],
                    portfolio_state=portfolio_state,
                    positions_state=positions_state,
                    symbols=symbols,
                    proposed_weight=float(c.get("proposed_weight", 0.1) or 0.1),
                )
                for c in candidates
            }

        results: dict[str, NemotronDecision] = {}
        slots_used = 0

        for c in candidates:
            sym = c["symbol"]
            features = c["features"]
            proposed_weight = float(c.get("proposed_weight", 0.1))
            reflex = c.get("reflex", {"reflex": "allow", "micro_state": "stable", "reason": "batch"})
            market_state_review = c.get("market_state_review", {}) if isinstance(c.get("market_state_review", {}), dict) else {}
            lane_supervision = c.get("lane_supervision", {}) if isinstance(c.get("lane_supervision", {}), dict) else {}
            t_sym = time.perf_counter()

            if sym in decision_map:
                d = decision_map[sym]
            elif batch_parse_failed:
                d = _safe_batch_model_failure_decision(sym, features, reflex)
            else:
                d = {"action": "HOLD", "reason": "not_ranked_by_batch"}
            action, _normalized_action_fields = normalize_trade_action(d.get("action", TRADE_ACTION_HOLD))
            side_raw = d.get("side")
            side = str(side_raw).upper() if side_raw else None
            if side == "BUY":
                side = "LONG"
            elif side == "SELL":
                side = "SHORT"
            reason = str(d.get("reason", "batch_hold"))
            size_raw = d.get("size", 0)
            if action in {TRADE_ACTION_WATCH, TRADE_ACTION_TIGHTEN, TRADE_ACTION_SCALE_IN, TRADE_ACTION_SCALE_OUT, TRADE_ACTION_ROTATE, TRADE_ACTION_SKIP, TRADE_ACTION_CLOSE}:
                action = TRADE_ACTION_HOLD
            phi_chart_support = self._phi_chart_support_level(
                features=features,
                market_state_review=market_state_review,
            )
            phi_requires_explicit_override = self._phi_requires_explicit_override(
                features=features,
                market_state_review=market_state_review,
            )

            if (
                action == "HOLD"
                and reason in {"m5_zero", "hold_unspecified", "not_ranked_by_batch", "batch_hold"}
                and (phi_chart_support == "strong" or phi_requires_explicit_override)
                and str(reflex.get("reflex", "allow")) == "allow"
                and str(features.get("entry_recommendation", "WATCH") or "WATCH").upper() in {"BUY", "STRONG_BUY"}
                and str(features.get("reversal_risk", "MEDIUM") or "MEDIUM").upper() != "HIGH"
            ):
                action = "OPEN"
                side = "LONG"
                reason = "phi_structure_confirmed"
                size_raw = round(max_trade * 0.6, 2)

            # Enforce slot limit across batch
            if action == "OPEN" and slots_used >= open_slots:
                action = "HOLD"
                reason = "batch_slots_exhausted"
                side = None

            # Cooldown: block re-entry if this symbol was recently closed
            if action == "OPEN":
                cooldown_min = float(get_runtime_setting("REENTRY_COOLDOWN_MIN") if hasattr(get_runtime_setting("REENTRY_COOLDOWN_MIN"), "__float__") else 240.0)
                try:
                    cooldown_min = float(get_runtime_setting("REENTRY_COOLDOWN_MIN"))
                except Exception:
                    cooldown_min = 240.0
                if self._memory_store.is_in_cooldown(sym, cooldown_minutes=cooldown_min):
                    action = "HOLD"
                    reason = "reentry_cooldown"
                    side = None

            integrated_reflex = {
                "action": action,
                "reason": reason,
                "debug": {
                    "batch": True,
                    "batch_reasoning": reasoning[:120],
                    "phi_chart_support": phi_chart_support,
                    "phi_requires_explicit_override": phi_requires_explicit_override,
                    "primary_blocker": reason if action == "HOLD" else "",
                },
                "override_signal": side if action == "OPEN" and side in {"LONG", "SHORT"} else "FLAT",
                "size_factor_hint": 1.0,
                "reflex": reflex.get("reflex", "allow"),
                "micro_state": reflex.get("micro_state", "stable"),
                "market_state_review": market_state_review,
                "lane_supervision": lane_supervision,
            }

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
                symbol=sym,
                signal=signal,
                proposed_weight=proposed_weight,
                features=features,
                symbols=symbols,
            )
            if portfolio_decision["reasons"]:
                risk_checks.extend(portfolio_decision["reasons"])
            if portfolio_decision["decision"] == "block" and "block" not in risk_checks:
                risk_checks.append("block")

            if (
                action == "OPEN"
                and side in {"LONG", "SHORT"}
                and not self._reflex_is_hard_block(reflex)
                and portfolio_decision["decision"] != "replace"
                and "block" not in risk_checks
            ):
                try:
                    size_hint = float(size_raw) / max(max_trade, 1.0) if float(size_raw or 0) > 0 else 1.0
                except (TypeError, ValueError):
                    size_hint = 1.0
                execution = execution_place_order(
                    self.executor,
                    signal=signal,
                    symbol=sym,
                    features=features,
                    portfolio_state=portfolio_state,
                    size_factor=portfolio_decision["size_factor"] * min(size_hint, 2.0),
                )
                if execution.get("status") == "filled":
                    slots_used += 1
            else:
                execution = {
                    "status": "no_trade",
                    "bar_ts": features.get("bar_ts"),
                    "bar_idx": features.get("bar_idx"),
                }
                if portfolio_decision["decision"] == "replace":
                    execution["reason"] = ["replacement_required"]
                    execution["replace_symbol"] = portfolio_decision.get("replace_symbol")

            execution["reflex"] = reflex
            execution["nemotron"] = integrated_reflex
            amortized_ms = round(nemo_ms / max(len(candidates), 1), 2)
            timings = {
                "phi3_ms": round(float(c.get("phi3_ms", 0.0)), 2),
                "advisory_ms": 0.0,
                "nemotron_ms": amortized_ms,
                "execution_ms": round((time.perf_counter() - t_sym) * 1000.0, 2),
                "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            }
            results[sym] = NemotronDecision(
                signal=signal,
                risk_checks=risk_checks,
                portfolio_decision=portfolio_decision,
                execution=execution,
                reflex={"phi3": reflex, "nemotron": integrated_reflex},
                timings=timings,
            )

        return results
