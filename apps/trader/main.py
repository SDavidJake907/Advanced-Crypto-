from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv

from apps.trader.account_sync_state import build_degraded_account_sync_payload, build_synced_account_sync_payload
from apps.trader.boot import (
    ACCOUNT_SYNC_INTERVAL_SEC,
    CANDLES_PATH,
    CANDLES_PATH_1H,
    CANDLES_PATH_30D,
    CANDLES_PATH_7D,
    CANDLES_PATH_TEMPLATE,
    CANDLES_PATH_TEMPLATE_1H,
    CANDLES_PATH_TEMPLATE_30D,
    CANDLES_PATH_TEMPLATE_7D,
    DATA_SOURCE,
    DECISION_ENGINE,
    INTRABAR_PRICE_THRESHOLD,
    LOOP_ALIVE_LOG,
    NEMOTRON_BATCH_MODE,
    NEMOTRON_BATCH_SKIP_PHI3,
    NEMOTRON_BATCH_TOP_N,
    NEMOTRON_MAX_PER_CYCLE,
    RUN_ONCE_ON_STATIC,
    SHADOW_DECISION_ENGINE,
    SHADOW_VALIDATION_ENABLED,
    SYMBOLS,
    SYMBOL_STATE_RETENTION_SEC,
    USE_ACTIVE_UNIVERSE,
    TraderComponents,
    boot_trader,
    effective_nemotron_batch_mode,
    load_trading_symbols,
    read_json_file,
    _resolve_candles_path,
    _symbol_token,
)
from core.config.runtime import get_proposed_weight, get_runtime_setting
from core.data.account_sync import bootstrap_account_state
from core.data.kraken_rest import KrakenRestClient
from apps.trader.cycle import (
    apply_lane_supervision,
    fetch_live_ohlc,
    fetch_ohlc,
    format_decision_timings,
    frame_last_bar_key,
    frame_last_price,
    load_lane_supervision_map,
    market_sentiment_fallback,
    run_blocking_decision,
    select_symbols_for_cycle,
)
from core.data import warmup
from core.features.batch import compute_features_batch, slice_features_for_asset
from core.features.pattern_engine import detect_top_pattern_from_frame
from core.llm.client import advisory_provider_name
from core.llm.phi3_pattern_verifier import verify_pattern_evidence
from core.llm.micro_prompts import nemotron_review_outcome, phi3_review_exit_posture
from core.llm.client import unload_nemotron_model
from core.memory.trade_memory import build_outcome_record
from apps.trader.positions import (
    PositionAction,
    compute_hold_minutes,
    compute_pnl_pct,
    execute_replacement_exit,
    handle_open_position,
    prune_symbol_tracking_state,
)
from core.risk.exits import build_exit_plan
from core.risk.runtime_health import evaluate_runtime_health
from core.risk.portfolio import PortfolioConfig, Position, PositionState, build_opportunity_snapshot, evaluate_trade
from core.runtime.cleanup import cleanup_tasks
from core.runtime.trader_lock import TraderAlreadyRunningError, TraderSingletonLock, build_trader_instance_id
from core.policy.trade_plan import build_trade_plan_metadata
from core.policy.nemo_payload_merge import merge_candidate_with_phi3
from core.state.portfolio import PortfolioState
from core.state.position_state_store import load_position_state, merge_persisted_positions, save_position_state
from apps.trader.logging_sink import (
    ACCOUNT_SYNC_LOG_PATH,
    DEBUG_LOG_PATH,
    OUTCOME_REVIEW_LOG_PATH,
    SHADOW_LOG_PATH,
    TRACE_LOG_PATH,
    console_event,
    log_account_sync,
    log_decision_debug,
    log_outcome_review,
    log_shadow_decision,
    log_trace,
    now_iso,
)
from core.state.store import load_universe, read_reload_signal
from core.state.trace import DecisionTrace
from core.validation.engine_runner import build_engine_components
from core.state.open_orders import reconcile_open_orders, reconcile_with_positions
from core.health.watchdog import check_all_endpoints, log_watchdog_status
from apps.trader.shadow import (
    build_shadow_payload,
    extract_llm_metrics,
    prepare_shadow_state,
    run_shadow_decision,
)

load_dotenv()

WARMUP_LOG_PATH = Path("logs/warmup.jsonl")
COLLECTOR_TELEMETRY_PATH = Path("logs/collector_telemetry.json")


def _build_phi3_reflex_cache_key(
    *,
    symbol: str,
    bar_key: str,
    features: dict[str, object],
) -> str:
    return "|".join(
        [
            str(symbol or "").upper(),
            str(bar_key or ""),
            str(features.get("bar_ts", "") or ""),
        ]
    )


def _compute_phi3_reflex_dict(features: dict[str, object]) -> dict[str, object]:
    from core.llm.phi3_reflex import phi3_reflex as _phi3_reflex_fn

    try:
        return _phi3_reflex_fn(features).to_dict()
    except Exception:
        return {"reflex": "allow", "micro_state": "stable", "reason": "reflex_error"}


_format_decision_timings = format_decision_timings









async def trader_loop(steps: int | None = None, *, instance_id: str | None = None) -> None:
    # --- Boot: initialise all components ---
    c = await boot_trader(instance_id)
    trader_instance_id  = c.trader_instance_id
    strategy            = c.strategy
    risk_engine         = c.risk_engine
    portfolio_config    = c.portfolio_config
    executor            = c.executor
    nemotron            = c.nemotron
    nemotron_needed     = c.nemotron_needed
    shadow_components   = c.shadow_components
    positions_state     = c.positions_state
    portfolio           = c.portfolio
    memory_store        = c.memory_store
    finbert_service     = c.finbert_service
    xgb_model           = c.xgb_model
    sentiment_feed      = c.sentiment_feed
    live_feed           = c.live_feed
    collector_task      = c.collector_task
    current_symbols     = c.current_symbols
    current_account_sync_status = c.current_account_sync_status
    last_xgb_train_ts   = c.last_xgb_train_ts

    last_bar_keys: dict[str, str] = {}
    last_prices: dict[str, float] = {}
    last_exit_ts: dict[str, float] = {}
    last_ticker_snapshot: dict = {}
    last_ticker_ts: float = 0.0
    phi3_reflex_cache: dict[str, dict[str, object]] = {}
    phi3_pattern_cache: dict[str, dict[str, object]] = {}
    market_client = KrakenRestClient()

    def _filter_symbol_map(mapping: dict[str, pd.DataFrame], symbols: list[str]) -> dict[str, pd.DataFrame]:
        return {symbol: mapping[symbol] for symbol in symbols if symbol in mapping}

    def _attach_pattern_context(
        *,
        symbol: str,
        features: dict[str, object],
        bar_key: str,
        ohlc_15m: pd.DataFrame | None,
        ohlc_1m: pd.DataFrame | None,
    ) -> dict[str, object]:
        frame = ohlc_15m if ohlc_15m is not None and not ohlc_15m.empty else ohlc_1m
        timeframe = "15m" if frame is ohlc_15m and frame is not None and not frame.empty else "1m"
        if frame is None or frame.empty:
            return features
        pattern_candidate = detect_top_pattern_from_frame(symbol=symbol, timeframe=timeframe, frame=frame)
        if not isinstance(pattern_candidate, dict) or not pattern_candidate:
            return features
        pattern_key = "|".join(
            [
                symbol,
                bar_key,
                timeframe,
                str(pattern_candidate.get("pattern", "none")),
                f"{float(pattern_candidate.get('confidence_raw', 0.0) or 0.0):.4f}",
            ]
        )
        cached = phi3_pattern_cache.get(symbol)
        if cached and str(cached.get("key", "")) == pattern_key:
            verification = dict(cached.get("verification", {}))
        else:
            verification = verify_pattern_evidence(pattern_candidate)
            phi3_pattern_cache[symbol] = {
                "key": pattern_key,
                "verification": dict(verification),
            }
        merged = merge_candidate_with_phi3(features, verification)
        merged["pattern_candidate"] = pattern_candidate
        merged["pattern_verification"] = verification
        return merged

    completed_steps = 0
    was_warm = DATA_SOURCE != "live"
    last_warmup_snapshot = None
    last_account_sync_ts = 0.0
    last_full_order_reconcile_ts = 0.0
    last_reload_token = read_reload_signal()
    last_watchdog_ts = 0.0
    last_watchdog_results: dict[str, object] = {}
    watchdog_interval = float(os.getenv("WATCHDOG_INTERVAL_SEC", "300"))
    if not positions_state.all():
        positions_state = load_position_state()
    try:
        while steps is None or completed_steps < steps:
            now_ts = time.time()
            if now_ts - last_watchdog_ts >= watchdog_interval:
                last_watchdog_results = check_all_endpoints()
                log_watchdog_status(last_watchdog_results)
                last_watchdog_ts = now_ts
            current_reload_token = read_reload_signal()
            if current_reload_token != last_reload_token:
                last_reload_token = current_reload_token
                current_symbols = load_trading_symbols()
                portfolio_config = PortfolioConfig.from_runtime()
                nemotron.portfolio_config = portfolio_config
                if DATA_SOURCE == "live" and live_feed is not None:
                    live_feed.set_symbols(current_symbols)
                last_account_sync_ts = 0.0
                prune_symbol_tracking_state(
                    current_symbols,
                    positions_state,
                    last_prices=last_prices,
                    last_bar_keys=last_bar_keys,
                    last_exit_ts=last_exit_ts,
                    retention_sec=SYMBOL_STATE_RETENTION_SEC,
                )
                console_event({"ts": now_iso(), "event": "reload_applied", "symbols": current_symbols})
            if DATA_SOURCE == "live" and executor.mode == "live" and (time.time() - last_account_sync_ts) >= ACCOUNT_SYNC_INTERVAL_SEC:
                try:
                    bootstrap = bootstrap_account_state(client=KrakenRestClient(), symbols=current_symbols)
                    portfolio = bootstrap.portfolio_state
                    positions_state = merge_persisted_positions(bootstrap.positions_state, load_position_state())
                    save_position_state(positions_state)
                    reconcile_with_positions(portfolio.positions)
                    # Dust sweep: clean up bot-tracked positions that fell below the dust threshold.
                    # Only attempt if the position is still above Kraken's min notional — anything
                    # truly unsellable gets removed from positions_state so it stops blocking slot counts.
                    dust_min_notional = float(get_runtime_setting("EXEC_MIN_NOTIONAL_USD"))
                    for dust_sym, dust_qty in bootstrap.ignored_dust_qty.items():
                        dust_usd = float(bootstrap.ignored_dust.get(dust_sym, 0.0))
                        if dust_qty <= 0.0 or positions_state.get(dust_sym) is None:
                            continue
                        if dust_usd >= dust_min_notional:
                            try:
                                dust_price = dust_usd / dust_qty
                                executor.execute_exit(
                                    symbol=dust_sym,
                                    side="LONG",
                                    qty=dust_qty,
                                    price=dust_price,
                                    features={},
                                    exit_reason="dust_sweep",
                                )
                                console_event({"ts": now_iso(), "event": "dust_sweep", "symbol": dust_sym, "qty": dust_qty, "usd": dust_usd})
                            except Exception:
                                pass
                        else:
                            # Too small to sell — remove from tracking so it doesn't count as an open position
                            positions_state.remove(dust_sym)
                            save_position_state(positions_state)
                            console_event({"ts": now_iso(), "event": "dust_dropped", "symbol": dust_sym, "usd": dust_usd})
                    sync_payload = build_synced_account_sync_payload(bootstrap.to_dict())
                    log_account_sync(sync_payload)
                    current_account_sync_status = sync_payload
                    last_account_sync_ts = time.time()
                except Exception as exc:
                    previous_sync_payload = current_account_sync_status or read_json_file(ACCOUNT_SYNC_LOG_PATH)
                    current_account_sync_status = build_degraded_account_sync_payload(
                        error=exc,
                        previous_payload=previous_sync_payload,
                    )
                    log_account_sync(current_account_sync_status)
            # Full open-order reconcile every 2 min: cancels stale orders across all symbols,
            # including coins that have rotated out of the active universe.
            if DATA_SOURCE == "live" and executor.mode == "live" and (time.time() - last_full_order_reconcile_ts) >= 120.0:
                try:
                    reconcile_open_orders(
                        client=KrakenRestClient(),
                        symbol=None,
                        timeout_sec=int(get_runtime_setting("ORDER_OPEN_TTL_SEC")),
                    )
                    last_full_order_reconcile_ts = time.time()
                except Exception:
                    pass
            try:
                await sentiment_feed.maybe_update()
            except Exception:
                pass
            # Cache universe once per loop — avoids 255 disk reads (load_universe called per symbol)
            _cached_universe = load_universe()
            prune_symbol_tracking_state(
                current_symbols,
                positions_state,
                last_prices=last_prices,
                last_bar_keys=last_bar_keys,
                last_exit_ts=last_exit_ts,
                retention_sec=SYMBOL_STATE_RETENTION_SEC,
            )
            # Compute FinBERT sentiment scores for active symbols
            finbert_scores: dict[str, float] = {}
            try:
                sentiment_snapshot = sentiment_feed.snapshot
                trending_headlines = [
                    f"{coin.symbol} trending" for coin in (sentiment_snapshot.trending or [])
                ]
                for _sym in current_symbols:
                    finbert_scores[_sym] = finbert_service.score_symbol(
                        _sym,
                        trending_headlines,
                        fallback_score=market_sentiment_fallback(_sym, sentiment_snapshot),
                    )
            except Exception:
                pass
            # Background XGBoost retraining every 6 hours
            if time.time() - last_xgb_train_ts > 21600:
                try:
                    xgb_model.train_from_memory(
                        str(memory_store.outcomes_path),
                        "logs/decision_debug.jsonl",
                    )
                    last_xgb_train_ts = time.time()
                except Exception:
                    pass
            if DATA_SOURCE == "live" and live_feed is not None:
                live_feed.set_symbols(current_symbols)
            if DATA_SOURCE == "live":
                warmup_info = warmup.check_status(live_feed, was_ready=was_warm)
                if not warmup_info.ready:
                    warmup_state = {
                        "warmup_status": "in_progress",
                        "symbols_ready": warmup_info.symbols_ready,
                        "symbols_pending": warmup_info.symbols_pending,
                        "hard_timeframes": warmup_info.hard_timeframes,
                        "soft_timeframes": warmup_info.soft_timeframes,
                        "timeframes": warmup_info.timeframe_progress,
                    }
                    if warmup_state != last_warmup_snapshot:
                        warmup_payload = {"ts": now_iso(), **warmup_state}
                        console_event(warmup_payload)
                        warmup.emit_warmup_status(WARMUP_LOG_PATH, warmup_payload)
                        last_warmup_snapshot = warmup_state
                    await asyncio.sleep(1)
                    continue
                ready_symbols = warmup_info.symbols_ready
                # Always include symbols with open positions — stop losses must fire
                # even if the symbol hasn't re-warmed after a restart.
                held_symbols = [p.symbol for p in positions_state.all()]
                for _held in held_symbols:
                    if _held not in ready_symbols:
                        ready_symbols = list(ready_symbols) + [_held]
                if warmup_info.just_completed:
                    warmup_payload = {
                        "ts": now_iso(),
                        "warmup_status": "complete",
                        "hard_timeframes": warmup_info.hard_timeframes,
                        "soft_timeframes": warmup_info.soft_timeframes,
                        "symbols_ready": warmup_info.symbols_ready,
                        "symbols_pending": warmup_info.symbols_pending,
                        "ready_for_trading": True,
                    }
                    console_event(warmup_payload)
                    warmup.emit_warmup_status(WARMUP_LOG_PATH, warmup_payload)
                    was_warm = True
                if not ready_symbols:
                    await asyncio.sleep(1)
                    continue
                eval_symbols = ready_symbols
                _ohlc_tfs = ["1m", "5m", "15m", "1h", "7d", "30d"]
                _ohlc_results = await asyncio.gather(*[
                    asyncio.gather(*[fetch_live_ohlc(live_feed, s, tf) for s in eval_symbols])
                    for tf in _ohlc_tfs
                ])
                ohlc_by_symbol     = dict(zip(eval_symbols, _ohlc_results[0]))
                ohlc_5m_by_symbol  = dict(zip(eval_symbols, _ohlc_results[1]))
                ohlc_15m_by_symbol = dict(zip(eval_symbols, _ohlc_results[2]))
                ohlc_1h_by_symbol  = dict(zip(eval_symbols, _ohlc_results[3]))
                ohlc_7d_by_symbol  = dict(zip(eval_symbols, _ohlc_results[4]))
                ohlc_30d_by_symbol = dict(zip(eval_symbols, _ohlc_results[5]))
            else:
                eval_symbols = current_symbols
                ohlc_by_symbol = {symbol: await fetch_ohlc(symbol) for symbol in current_symbols}
                ohlc_5m_by_symbol: dict[str, pd.DataFrame] = {}
                ohlc_15m_by_symbol: dict[str, pd.DataFrame] = {}
                ohlc_1h_by_symbol = {
                    symbol: await fetch_ohlc(symbol, base_path=CANDLES_PATH_1H, template=CANDLES_PATH_TEMPLATE_1H)
                    for symbol in current_symbols
                }
                ohlc_7d_by_symbol = {
                    symbol: await fetch_ohlc(symbol, base_path=CANDLES_PATH_7D, template=CANDLES_PATH_TEMPLATE_7D)
                    for symbol in current_symbols
                }
                ohlc_30d_by_symbol = {
                    symbol: await fetch_ohlc(symbol, base_path=CANDLES_PATH_30D, template=CANDLES_PATH_TEMPLATE_30D)
                    for symbol in current_symbols
                }

            if time.time() - last_ticker_ts >= 10.0:
                try:
                    last_ticker_snapshot = market_client.get_ticker_snapshot(eval_symbols)
                    last_ticker_ts = time.time()
                except Exception:
                    pass
            ticker_snapshot = last_ticker_snapshot

            requested_symbols = len(eval_symbols)
            valid_symbols = [symbol for symbol, frame in ohlc_by_symbol.items() if not frame.empty]
            if not valid_symbols:
                await asyncio.sleep(1)
                continue
            missing_symbols: list[str] = []
            if len(valid_symbols) != len(eval_symbols):
                missing_symbols = sorted(set(eval_symbols) - set(valid_symbols))
                if LOOP_ALIVE_LOG:
                    log_decision_debug(
                        {
                            "ts": now_iso(),
                            "loop": "missing_ohlc",
                            "missing_symbols": missing_symbols,
                            "total_eval": len(eval_symbols),
                            "total_valid": len(valid_symbols),
                        }
                    )
                eval_symbols = valid_symbols
                ohlc_by_symbol = _filter_symbol_map(ohlc_by_symbol, valid_symbols)
                ohlc_5m_by_symbol = _filter_symbol_map(ohlc_5m_by_symbol, valid_symbols)
                ohlc_15m_by_symbol = _filter_symbol_map(ohlc_15m_by_symbol, valid_symbols)
                ohlc_1h_by_symbol = _filter_symbol_map(ohlc_1h_by_symbol, valid_symbols)
                ohlc_7d_by_symbol = _filter_symbol_map(ohlc_7d_by_symbol, valid_symbols)
                ohlc_30d_by_symbol = _filter_symbol_map(ohlc_30d_by_symbol, valid_symbols)
            eval_symbols = valid_symbols
            if LOOP_ALIVE_LOG:
                log_decision_debug(
                    {
                        "ts": now_iso(),
                        "loop": "heartbeat",
                        "instance_id": trader_instance_id,
                        "pid": os.getpid(),
                        "decision_engine": DECISION_ENGINE,
                        "requested_symbols": requested_symbols,
                        "valid_symbols": len(valid_symbols),
                        "missing_symbols": missing_symbols,
                        "stale_symbols": [],
                    }
                )
            for symbol in eval_symbols:
                if symbol not in last_prices:
                    last_prices[symbol] = frame_last_price(ohlc_by_symbol[symbol])
            active_eval_symbols, static_symbols, intrabar_prices = select_symbols_for_cycle(
                eval_symbols,
                ohlc_by_symbol,
                last_bar_keys=last_bar_keys,
                last_prices=last_prices,
                intrabar_price_threshold=INTRABAR_PRICE_THRESHOLD,
            )

            if LOOP_ALIVE_LOG:
                log_decision_debug(
                    {
                        "ts": now_iso(),
                        "loop": "cycle_activity",
                        "active_symbols": len(active_eval_symbols),
                        "static_symbols": static_symbols,
                    }
                )

            if not active_eval_symbols:

                if LOOP_ALIVE_LOG:
                    log_decision_debug(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "loop": "static_bar",
                            "static_symbols": static_symbols,
                            "prices": intrabar_prices,
                            "intrabar_triggered": False,
                        }
                    )

                if RUN_ONCE_ON_STATIC:
                    break
                await asyncio.sleep(1)
                continue

            for symbol in active_eval_symbols:
                bar_key = frame_last_bar_key(ohlc_by_symbol[symbol])
                if bar_key is not None:
                    last_bar_keys[symbol] = bar_key
            features_started = time.perf_counter()
            features_batch = compute_features_batch(
                _filter_symbol_map(ohlc_by_symbol, active_eval_symbols),
                ohlc_5m_by_symbol=_filter_symbol_map(ohlc_5m_by_symbol, active_eval_symbols),
                ohlc_15m_by_symbol=_filter_symbol_map(ohlc_15m_by_symbol, active_eval_symbols),
                ohlc_1h_by_symbol=_filter_symbol_map(ohlc_1h_by_symbol, active_eval_symbols),
                ohlc_7d_by_symbol=_filter_symbol_map(ohlc_7d_by_symbol, active_eval_symbols),
                ohlc_30d_by_symbol=_filter_symbol_map(ohlc_30d_by_symbol, active_eval_symbols),
                finbert_scores=finbert_scores,
                xgb_model=xgb_model,
            )
            features_ms = (time.perf_counter() - features_started) * 1000.0

            lane_supervision_map = load_lane_supervision_map()

            runtime_health = evaluate_runtime_health(
                now_ts=now_ts,
                collector_telemetry=read_json_file(COLLECTOR_TELEMETRY_PATH),
                account_sync=current_account_sync_status or read_json_file(ACCOUNT_SYNC_LOG_PATH),
                watchdog_results=last_watchdog_results,
                telemetry_dir=DEBUG_LOG_PATH.parent,
                require_account_sync=DATA_SOURCE == "live" and executor.mode == "live",
                llm_required=DECISION_ENGINE == "llm",
            ).to_dict()

            # Cap Nemo evaluations per cycle — rank non-held symbols by entry_score,
            # only allow top NEMOTRON_MAX_PER_CYCLE through the full decide() call.
            _held_syms = {
                sym for sym in features_batch["symbols"]
                if positions_state.get(sym) is not None or float(portfolio.positions.get(sym, 0.0)) > 0.0
            }
            # Phi-3 scan watchlist boost: symbols Phi-3 flagged get up to +10 pts on entry_score
            # so they rank higher and are more likely to make the top-15 Nemo batch.
            _phi3_scan_boost: dict[str, float] = {}
            try:
                _scan_meta = (_cached_universe.get("meta") or {}).get("phi3_scan") or {}
                for _item in (_scan_meta.get("watchlist") or []):
                    _sym = str(_item.get("symbol", ""))
                    _conf = float(_item.get("confidence", 0.0) or 0.0)
                    if _sym:
                        _phi3_scan_boost[_sym] = _conf * 10.0
            except Exception:
                pass
            _score_rank: list[tuple[tuple[float, ...], str]] = []
            _structure_syms: set[str] = set()
            _warmup_syms: set[str] = set()
            for _ai, _sym in enumerate(features_batch["symbols"]):
                if _sym not in _held_syms:
                    _sf = slice_features_for_asset(features_batch, _ai)
                    if not bool(_sf.get("indicators_ready", True)):
                        _warmup_syms.add(_sym)
                        continue
                    _base = float(_sf.get("entry_score", 0.0) or 0.0)
                    _boost = _phi3_scan_boost.get(_sym, 0.0)
                    _ema_ok = bool(_sf.get("ema9_above_ema20", False))
                    _brk = bool(_sf.get("range_breakout_1h", False))
                    _pb = bool(_sf.get("pullback_hold", False))
                    _hl = int(_sf.get("higher_low_count", 0) or 0)
                    _promo_tier = str(_sf.get("promotion_tier", "skip") or "skip").lower()
                    _promo_rank = 2.0 if _promo_tier == "promote" else 1.0 if _promo_tier == "probe" else 0.0
                    _short_tf_ready = bool(_sf.get("short_tf_ready_5m", False) or _sf.get("short_tf_ready_15m", False))
                    _overextended = bool(_sf.get("overextended", False))
                    _range_pos_1h = float(_sf.get("range_pos_1h", 0.5) or 0.5)
                    _reversal = str(_sf.get("reversal_risk", "MEDIUM") or "MEDIUM").upper()
                    _risk_rank = 2.0 if _reversal == "LOW" else 1.0 if _reversal == "MEDIUM" else 0.0
                    _fresh_structure = 1.0 if (_ema_ok and (_pb or _brk or _hl >= 3) and not _overextended) else 0.0
                    _retest_bonus = 1.0 if _pb else 0.0
                    _breakout_bonus = 1.0 if _brk else 0.0
                    _short_tf_bonus = 1.0 if _short_tf_ready else 0.0
                    _not_extended = 1.0 if (_range_pos_1h <= 0.88 and not _overextended) else 0.0
                    _score_rank.append((
                        (
                            _fresh_structure,
                            _promo_rank,
                            _retest_bonus,
                            _breakout_bonus,
                            _short_tf_bonus,
                            _not_extended,
                            _risk_rank,
                            _base + _boost,
                        ),
                        _sym,
                    ))
                    # Elite structure bypass — channel_breakout/retest always gets through
                    _promo = str(_sf.get("promotion_reason", "") or "")
                    if (_ema_ok and (_brk or _pb)) or _promo in {"channel_breakout", "channel_retest"}:
                        _structure_syms.add(_sym)
            _score_rank.sort(reverse=True)
            _entry_nemo_limit = max(1, min(NEMOTRON_MAX_PER_CYCLE, NEMOTRON_BATCH_TOP_N))
            _ranked_entry_syms = {sym for _, sym in _score_rank[:_entry_nemo_limit]}
            _nemo_allowed = _held_syms | _ranked_entry_syms | _structure_syms
            _use_batch_nemo = effective_nemotron_batch_mode()

            # Batch mode: run ONE Nemo call for all top candidates instead of N serial calls
            _batch_decisions: dict[str, object] = {}
            if DECISION_ENGINE == "llm" and _use_batch_nemo:
                from core.features.market_trend import compute_market_trend as _compute_market_trend
                # Collect candidates — non-held symbols only, with a small ranked batch and
                # elite structure bypass. Phi reflex is cached by symbol+bar to avoid
                # recomputing the same advisory on every loop pass.
                _batch_candidates = []
                _phi3_cache_hits = 0
                _phi3_skipped = 0
                _phi3_uncached = 0
                _phi3_stage_ms = 0.0
                _pending_phi3_refs: list[dict[str, object]] = []
                for _ai, _sym in enumerate(features_batch["symbols"]):
                    if _sym in _held_syms:
                        continue
                    if _sym in _warmup_syms:
                        continue
                    if _sym not in _ranked_entry_syms and _sym not in _structure_syms:
                        continue
                    _bf = slice_features_for_asset(features_batch, _ai)
                    if not bool(_bf.get("indicators_ready", True)):
                        continue
                    # Inject live market data (bid/ask/book_valid) into batch features
                    # so execution_place_order can price limit orders correctly
                    _msnap = ticker_snapshot.get(_sym, {})
                    if DATA_SOURCE == "live" and live_feed is not None:
                        try:
                            _lsnap = live_feed.get_market_snapshot(_sym)
                            if bool(_lsnap.get("book_valid", False)):
                                _msnap = {**_msnap, **_lsnap}
                            else:
                                _msnap.update({k: _lsnap.get(k, 0.0) for k in ("book_bid_depth", "book_ask_depth", "book_imbalance", "book_wall_pressure")})
                        except KeyError:
                            pass
                    if _msnap:
                        _bf["bid"] = float(_msnap.get("bid", 0.0) or 0.0)
                        _bf["ask"] = float(_msnap.get("ask", 0.0) or 0.0)
                        _bf["spread_pct"] = float(_msnap.get("spread_pct", _bf.get("spread_pct", 0.0)) or 0.0)
                        _bv_rest = _bf["bid"] > 0.0 and _bf["ask"] > 0.0 and _bf["ask"] >= _bf["bid"]
                        _bf["book_valid"] = bool(_msnap.get("book_valid", _bv_rest))
                    _bf = _attach_pattern_context(
                        symbol=_sym,
                        features=_bf,
                        bar_key=str(last_bar_keys.get(_sym, "")),
                        ohlc_15m=ohlc_15m_by_symbol.get(_sym),
                        ohlc_1m=ohlc_by_symbol.get(_sym),
                    )
                    _bar_key = str(last_bar_keys.get(_sym, ""))
                    _cache_key = _build_phi3_reflex_cache_key(
                        symbol=_sym,
                        bar_key=_bar_key,
                        features=_bf,
                    )
                    if NEMOTRON_BATCH_SKIP_PHI3:
                        _reflex = {"reflex": "allow", "micro_state": "stable", "reason": "batch_phi3_skipped"}
                        _phi3_ms = 0.0
                        _phi3_skipped += 1
                    else:
                        _cached_reflex = phi3_reflex_cache.get(_sym)
                        if _cached_reflex and str(_cached_reflex.get("key", "")) == _cache_key:
                            _reflex = dict(_cached_reflex.get("reflex", {}))
                            _phi3_ms = float(_cached_reflex.get("phi3_ms", 0.0) or 0.0)
                            _phi3_cache_hits += 1
                        else:
                            _reflex = {"reflex": "allow", "micro_state": "pending", "reason": "batch_phi3_pending"}
                            _phi3_ms = 0.0
                            _phi3_uncached += 1
                    _rank_score = float(_bf.get("entry_score", 0.0) or 0.0) + _phi3_scan_boost.get(_sym, 0.0)
                    _elite = _sym in _structure_syms
                    _promo_tier = str(_bf.get("promotion_tier", "skip") or "skip").lower()
                    _promo_rank = 2.0 if _promo_tier == "promote" else 1.0 if _promo_tier == "probe" else 0.0
                    _reversal = str(_bf.get("reversal_risk", "MEDIUM") or "MEDIUM").upper()
                    _risk_rank = 2.0 if _reversal == "LOW" else 1.0 if _reversal == "MEDIUM" else 0.0
                    _short_tf_ready = bool(_bf.get("short_tf_ready_5m", False) or _bf.get("short_tf_ready_15m", False))
                    _overextended = bool(_bf.get("overextended", False))
                    _range_pos_1h = float(_bf.get("range_pos_1h", 0.5) or 0.5)
                    _batch_candidates.append({
                        "symbol": _sym,
                        "features": _bf,
                        "reflex": _reflex,
                        "phi3_ms": _phi3_ms,
                        "proposed_weight": float(get_proposed_weight(_sym, _bf.get("lane"))),
                        "_rank_score": _rank_score,
                        "_elite": _elite,
                        "_promo_rank": _promo_rank,
                        "_retest_bonus": 1.0 if bool(_bf.get("pullback_hold", False)) else 0.0,
                        "_breakout_bonus": 1.0 if bool(_bf.get("range_breakout_1h", False)) else 0.0,
                        "_short_tf_bonus": 1.0 if _short_tf_ready else 0.0,
                        "_not_extended": 1.0 if (_range_pos_1h <= 0.88 and not _overextended) else 0.0,
                        "_risk_rank": _risk_rank,
                        "_phi3_cache_key": _cache_key,
                    })
                    if not NEMOTRON_BATCH_SKIP_PHI3 and _phi3_ms <= 0.0:
                        _pending_phi3_refs.append(_batch_candidates[-1])
                if _pending_phi3_refs:
                    _phi3_stage_start = time.perf_counter()
                    _phi3_concurrency = max(1, int(os.getenv("PHI3_BATCH_MAX_CONCURRENCY", "2")))
                    _phi3_sem = asyncio.Semaphore(_phi3_concurrency)

                    async def _resolve_pending_phi3(_candidate: dict[str, object]) -> None:
                        async with _phi3_sem:
                            _t_phi3 = time.perf_counter()
                            _reflex = await run_blocking_decision(_compute_phi3_reflex_dict, _candidate["features"])
                            _phi3_ms_local = (time.perf_counter() - _t_phi3) * 1000.0
                            _candidate["reflex"] = _reflex
                            _candidate["phi3_ms"] = _phi3_ms_local
                            phi3_reflex_cache[str(_candidate["symbol"])] = {
                                "key": str(_candidate.get("_phi3_cache_key", "")),
                                "reflex": dict(_reflex),
                                "phi3_ms": _phi3_ms_local,
                            }

                    await asyncio.gather(*[_resolve_pending_phi3(_cand) for _cand in _pending_phi3_refs])
                    _phi3_stage_ms = (time.perf_counter() - _phi3_stage_start) * 1000.0
                if _batch_candidates:
                    _batch_candidates.sort(
                        key=lambda item: (
                            1 if bool(item.get("_elite")) else 0,
                            float(item.get("_promo_rank", 0.0) or 0.0),
                            float(item.get("_retest_bonus", 0.0) or 0.0),
                            float(item.get("_breakout_bonus", 0.0) or 0.0),
                            float(item.get("_short_tf_bonus", 0.0) or 0.0),
                            float(item.get("_not_extended", 0.0) or 0.0),
                            float(item.get("_risk_rank", 0.0) or 0.0),
                            float(item.get("_rank_score", 0.0) or 0.0),
                        ),
                        reverse=True,
                    )
                    _batch_candidates = _batch_candidates[: max(1, _entry_nemo_limit + len(_structure_syms))]
                    # Deterministic market trend state using BTC as market-wide barometer.
                    # Advisory only — adjusts entry scores so Nemo has accurate context.
                    _mts = _compute_market_trend(
                        features_batch,
                        list(features_batch["symbols"]),
                        bear_score_penalty=float(get_runtime_setting("MARKET_TREND_BEAR_SCORE_PENALTY")),
                        bull_score_boost=float(get_runtime_setting("MARKET_TREND_BULL_SCORE_BOOST")),
                    )
                    _market_summary = _mts.to_nemo_context()
                    # Apply score bias to each candidate so Nemo sees adjusted scores
                    if _mts.score_bias != 0.0:
                        for _cand in _batch_candidates:
                            _cand["features"] = dict(_cand["features"])
                            _cand["features"]["entry_score"] = float(
                                _cand["features"].get("entry_score", 0.0) or 0.0
                            ) + _mts.score_bias
                    _batch_decisions = await run_blocking_decision(
                        nemotron.batch_decide,
                        candidates=_batch_candidates,
                        portfolio_state=portfolio,
                        positions_state=positions_state,
                        symbols=features_batch["symbols"],
                        market_state_summary=_market_summary,
                    )
                    log_decision_debug({
                        "ts": now_iso(),
                        "loop": "batch_nemo",
                        "candidates": len(_batch_candidates),
                        "held_bypass_count": len(_held_syms),
                        "ranked_entry_limit": _entry_nemo_limit,
                        "structure_bypass_count": len(_structure_syms),
                        "warmup_skip_count": len(_warmup_syms),
                        "phi3_cache_hits": _phi3_cache_hits,
                        "phi3_skipped_count": _phi3_skipped,
                        "phi3_uncached_count": _phi3_uncached,
                        "phi3_stage_ms": round(_phi3_stage_ms, 2),
                        "market_trend": _mts.trend,
                        "market_trend_strength": _mts.strength,
                        "market_trend_score_bias": _mts.score_bias,
                        "market_context": _market_summary,
                        "decisions": {
                            sym: dec.signal
                            for sym, dec in _batch_decisions.items()
                        },
                    })

            for asset_idx, symbol in enumerate(features_batch["symbols"]):
                lane_supervision = lane_supervision_map.get(symbol)
                universe_lane = (lane_supervision or {}).get("universe_lane")
                features = slice_features_for_asset(features_batch, asset_idx, lane_hint=universe_lane)
                if lane_supervision:
                    # Store both lanes for logging — runtime may differ from universe
                    features["universe_lane"] = universe_lane or features.get("lane")
                    features = apply_lane_supervision(features, lane_supervision)
                market_snapshot = ticker_snapshot.get(symbol, {})
                if DATA_SOURCE == "live" and live_feed is not None:
                    try:
                        live_market_snapshot = live_feed.get_market_snapshot(symbol)
                    except KeyError:
                        live_market_snapshot = {}
                    if live_market_snapshot:
                        if bool(live_market_snapshot.get("book_valid", False)):
                            market_snapshot = {**market_snapshot, **live_market_snapshot}
                        else:
                            market_snapshot.update(
                                {
                                    key: live_market_snapshot.get(key, 0.0)
                                    for key in ("book_bid_depth", "book_ask_depth", "book_imbalance", "book_wall_pressure")
                                }
                            )
                if market_snapshot:
                    features["bid"] = float(market_snapshot.get("bid", 0.0) or 0.0)
                    features["ask"] = float(market_snapshot.get("ask", 0.0) or 0.0)
                    features["spread_pct"] = float(market_snapshot.get("spread_pct", 0.0) or 0.0)
                    _rest_book_valid = features["bid"] > 0.0 and features["ask"] > 0.0 and features["ask"] >= features["bid"]
                    features["book_valid"] = bool(market_snapshot.get("book_valid", _rest_book_valid))
                    features["quote_volume"] = float(market_snapshot.get("quote_volume", 0.0) or 0.0)
                    features["book_bid_depth"] = float(market_snapshot.get("book_bid_depth", 0.0) or 0.0)
                    features["book_ask_depth"] = float(market_snapshot.get("book_ask_depth", 0.0) or 0.0)
                    features["book_imbalance"] = float(market_snapshot.get("book_imbalance", 0.0) or 0.0)
                    features["book_wall_pressure"] = float(market_snapshot.get("book_wall_pressure", 0.0) or 0.0)
                else:
                    features["book_valid"] = False
                features["runtime_health"] = runtime_health
                features = _attach_pattern_context(
                    symbol=symbol,
                    features=features,
                    bar_key=str(last_bar_keys.get(symbol, "")),
                    ohlc_15m=ohlc_15m_by_symbol.get(symbol),
                    ohlc_1m=ohlc_by_symbol.get(symbol),
                )
                sentiment_snapshot = sentiment_feed.snapshot
                features["sentiment_fng_value"] = int(sentiment_snapshot.fng_value)
                features["sentiment_fng_label"] = str(sentiment_snapshot.fng_label)
                features["sentiment_btc_dominance"] = float(sentiment_snapshot.btc_dominance)
                features["sentiment_market_cap_change_24h"] = float(sentiment_snapshot.market_cap_change_24h)
                features["sentiment_trending_symbols"] = [coin.symbol for coin in sentiment_snapshot.trending[:5]]
                features["sentiment_symbol_trending"] = symbol.split("/")[0].upper() in set(features["sentiment_trending_symbols"])
                features["proposed_weight"] = get_proposed_weight(symbol, features.get("lane"))
                proposed_weight = float(features["proposed_weight"])
                features["kelly_fraction"] = memory_store.get_kelly_fraction(symbol)
                features["symbol_performance"] = memory_store.get_symbol_performance(symbol)
                existing_position = positions_state.get(symbol)
                live_qty = float(portfolio.positions.get(symbol, 0.0))

                if not bool(features.get("indicators_ready", False)) and existing_position is None and live_qty <= 0.0:
                    log_decision_debug(
                        {
                            "ts": now_iso(),
                            "symbol": symbol,
                            "phase": "feature_warmup",
                            "action": "HOLD",
                            "feature_status": features.get("feature_status"),
                            "history_points": features.get("history_points"),
                            "reason": features.get("feature_failure_reason") or "feature_warmup",
                        }
                    )
                    last_prices[symbol] = float(features.get("price", 0.0) or 0.0)
                    continue

                if existing_position is not None and existing_position.entry_price is None and live_qty > 0.0:
                    if float(features.get("atr", 0.0)) > 0.0:
                        positions_state.add_or_update(
                            build_exit_plan(
                                symbol=symbol,
                                side=existing_position.side,
                                weight=existing_position.weight,
                                entry_price=float(features.get("price", 0.0)),
                                atr=float(features.get("atr", 0.0)),
                                entry_bar_ts=features.get("bar_ts"),
                                entry_bar_idx=features.get("bar_idx"),
                                lane=features.get("lane"),
                            )
                        )
                        save_position_state(positions_state)
                        existing_position = positions_state.get(symbol)

                if existing_position is not None and live_qty > 0.0:
                    pos_action = handle_open_position(
                        symbol, features, existing_position, live_qty,
                        executor=executor,
                        portfolio=portfolio,
                        positions_state=positions_state,
                        memory_store=memory_store,
                        last_exit_ts=last_exit_ts,
                    )
                    if pos_action == PositionAction.EXITED:
                        last_prices[symbol] = float(ohlc_by_symbol[symbol]["close"].iloc[-1])
                    if pos_action != PositionAction.PROCEED:
                        continue

                # Symbol cooldown: block re-entry for 90 min after any exit
                _symbol_cooldown_min = 90.0
                _last_exit = last_exit_ts.get(symbol, 0.0)
                if existing_position is None and _last_exit > 0.0 and (time.time() - _last_exit) < _symbol_cooldown_min * 60.0:
                    log_decision_debug({"ts": now_iso(), "symbol": symbol, "phase": "entry", "action": "HOLD", "reason": "symbol_cooldown", "cooldown_remaining_min": round((_symbol_cooldown_min * 60.0 - (time.time() - _last_exit)) / 60.0, 1)})
                    continue

                shadow_portfolio, shadow_positions = (
                    prepare_shadow_state(portfolio, positions_state)
                    if shadow_components is not None
                    else (None, None)
                )

                if DECISION_ENGINE == "llm" and symbol not in _nemo_allowed:
                    log_decision_debug({
                        "ts": now_iso(), "symbol": symbol, "phase": "entry",
                        "action": "HOLD", "reason": "nemo_cap_skip",
                        "entry_score": features.get("entry_score"),
                        "nemo_budget": NEMOTRON_MAX_PER_CYCLE,
                    })
                    last_prices[symbol] = float(features.get("price", 0.0) or 0.0)
                    continue

                if DECISION_ENGINE == "llm" and _use_batch_nemo and symbol in _batch_decisions:
                    nemotron_decision = _batch_decisions[symbol]
                elif DECISION_ENGINE == "llm":
                    nemotron_decision = await run_blocking_decision(
                        nemotron.decide,
                        symbol=symbol,
                        features=features,
                        portfolio_state=portfolio,
                        positions_state=positions_state,
                        symbols=features_batch["symbols"],
                        proposed_weight=proposed_weight,
                    )
                else:
                    nemotron_decision = None

                if nemotron_decision is not None:
                    signal = nemotron_decision.signal
                    risk_checks = nemotron_decision.risk_checks
                    exec_result = nemotron_decision.execution
                    timings = dict(nemotron_decision.timings)
                    state_change = {}
                    portfolio_decision = nemotron_decision.portfolio_decision
                    portfolio_opportunity = build_opportunity_snapshot(
                        positions=positions_state,
                        candidate_symbol=symbol,
                        features=features,
                    )
                    if portfolio_decision.get("decision") == "replace":
                        replace_symbol = str(portfolio_decision.get("replace_symbol") or "")
                        replacement_exec, replacement_state_change = execute_replacement_exit(
                            target_symbol=replace_symbol,
                            replacement_symbol=symbol,
                            executor=executor,
                            portfolio=portfolio,
                            positions_state=positions_state,
                            last_exit_ts=last_exit_ts,
                        )
                        exec_result = dict(exec_result)
                        exec_result["replacement"] = replacement_exec
                        if replacement_exec.get("status") == "filled":
                            state_change = replacement_state_change
                    if exec_result.get("status") == "filled":
                        state_change = portfolio.apply_execution(exec_result)
                        weight = proposed_weight * portfolio_decision["size_factor"]
                        trade_plan = build_trade_plan_metadata(signal, features, exec_result)
                        positions_state.add_or_update(
                            build_exit_plan(
                                symbol=symbol,
                                side=signal,
                                weight=weight,
                                entry_price=float(exec_result["price"]),
                                atr=float(features.get("atr", 0.0)),
                                entry_bar_ts=features.get("bar_ts"),
                                entry_bar_idx=features.get("bar_idx"),
                                entry_reasons=[
                                    str(exec_result.get("nemotron", {}).get("reason", "")),
                                    f"lane:{features.get('lane', 'main')}",
                                ],
                                lane=features.get("lane"),
                                entry_thesis=trade_plan["entry_thesis"],
                                expected_hold_style=trade_plan["expected_hold_style"],
                                invalidate_on=trade_plan["invalidate_on"],
                                expected_edge_pct=float((features.get("point_breakdown") or {}).get("net_edge_pct", 0.0)),
                            )
                        )
                        save_position_state(positions_state)
                else:
                    timings = {
                        "features_ms": round(features_ms, 2),
                        "phi3_ms": 0.0,
                        "advisory_ms": 0.0,
                        "nemotron_ms": 0.0,
                        "execution_ms": 0.0,
                        "total_ms": 0.0,
                    }
                    signal = strategy.generate_signal(features)
                    risk_checks = risk_engine.check(signal, features, portfolio.to_dict())
                    portfolio_decision = {
                        "decision": "allow",
                        "size_factor": 1.0,
                        "reasons": [],
                        "replace_symbol": None,
                        "replace_reason": None,
                    }
                    portfolio_opportunity = build_opportunity_snapshot(
                        positions=positions_state,
                        candidate_symbol=symbol,
                        features=features,
                    )
                    if signal in ("LONG", "SHORT"):
                        trend_1h = int(features.get("trend_1h", 0))
                        trend_conflict = (
                            features["lane"] == "L4"
                            and ((signal == "LONG" and trend_1h == -1) or (signal == "SHORT" and trend_1h == 1))
                        )
                        portfolio_decision = evaluate_trade(
                            config=portfolio_config,
                            positions=positions_state,
                            symbol=symbol,
                            side=signal,
                            proposed_weight=proposed_weight,
                            correlation_row=features["correlation_row"],
                            symbols=features_batch["symbols"],
                            lane=features["lane"],
                            trend_conflict=trend_conflict,
                            features=features,
                        )
                    if portfolio_decision["reasons"]:
                        risk_checks.extend(portfolio_decision["reasons"])
                    if portfolio_decision["decision"] == "block" and "block" not in risk_checks:
                        risk_checks.append("block")

                    if portfolio_decision["decision"] == "replace":
                        replace_symbol = str(portfolio_decision.get("replace_symbol") or "")
                        replacement_exec, state_change = execute_replacement_exit(
                            target_symbol=replace_symbol,
                            replacement_symbol=symbol,
                            executor=executor,
                            portfolio=portfolio,
                            positions_state=positions_state,
                            last_exit_ts=last_exit_ts,
                        )
                        exec_result = {
                            "status": "no_trade",
                            "reason": ["replacement_exit_submitted"],
                            "bar_ts": features.get("bar_ts"),
                            "bar_idx": features.get("bar_idx"),
                            "replacement": replacement_exec,
                        }
                    elif "block" in risk_checks:
                        exec_result = {
                            "status": "blocked",
                            "reason": risk_checks,
                            "bar_ts": features.get("bar_ts"),
                            "bar_idx": features.get("bar_idx"),
                        }
                        state_change = {}
                    else:
                        exec_result = executor.execute(
                            signal,
                            symbol,
                            features,
                            portfolio.to_dict(),
                            size_factor=portfolio_decision["size_factor"],
                        )
                        exec_result["bar_ts"] = features.get("bar_ts")
                        exec_result["bar_idx"] = features.get("bar_idx")
                        if portfolio_decision["decision"] == "scale_down":
                            exec_result["size_factor"] = portfolio_decision["size_factor"]
                        state_change = portfolio.apply_execution(exec_result)
                        if exec_result.get("status") == "filled":
                            weight = proposed_weight * portfolio_decision["size_factor"]
                            trade_plan = build_trade_plan_metadata(signal, features, exec_result)
                            positions_state.add_or_update(
                                build_exit_plan(
                                    symbol=symbol,
                                    side=signal,
                                    weight=weight,
                                    entry_price=float(exec_result["price"]),
                                    atr=float(features.get("atr", 0.0)),
                                    entry_bar_ts=features.get("bar_ts"),
                                    entry_bar_idx=features.get("bar_idx"),
                                    entry_reasons=[signal.lower(), f"lane:{features.get('lane', 'main')}"],
                                    lane=features.get("lane"),
                                    entry_thesis=trade_plan["entry_thesis"],
                                    expected_hold_style=trade_plan["expected_hold_style"],
                                    invalidate_on=trade_plan["invalidate_on"],
                                    expected_edge_pct=float((features.get("point_breakdown") or {}).get("net_edge_pct", 0.0)),
                                )
                            )
                            save_position_state(positions_state)

                if shadow_components is not None and shadow_portfolio is not None and shadow_positions is not None:
                    shadow_decision = await run_shadow_decision(
                        run_blocking_decision,
                        shadow_components,
                        symbol=symbol,
                        features=features,
                        shadow_portfolio=shadow_portfolio,
                        shadow_positions=shadow_positions,
                        all_symbols=features_batch["symbols"],
                        proposed_weight=proposed_weight,
                    )
                    log_shadow_decision(build_shadow_payload(
                        ts=now_iso(),
                        symbol=symbol,
                        features=features,
                        baseline_engine=DECISION_ENGINE,
                        shadow_components=shadow_components,
                        baseline_signal=signal,
                        baseline_exec=exec_result,
                        shadow_decision=shadow_decision,
                    ))

                trace = DecisionTrace(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    features=features,
                    signal=signal,
                    risk_checks=risk_checks,
                    execution=exec_result,
                    state_change=state_change,
                )
                log_trace(trace)
                log_decision_debug(
                    {
                        "ts": now_iso(),
                        "symbol": symbol,
                        "lane": features.get("lane"),
                        "universe_lane": features.get("universe_lane"),
                        "lane_drift": features.get("lane") != features.get("universe_lane") and features.get("universe_lane") is not None,
                        "price": features.get("price"),
                        "rotation_score": features.get("rotation_score"),
                        "entry_score": features.get("entry_score"),
                        "entry_recommendation": features.get("entry_recommendation"),
                        "reversal_risk": features.get("reversal_risk"),
                        "coin_profile": features.get("coin_profile", {}),
                        "trend_confirmed": features.get("trend_confirmed"),
                        "ranging_market": features.get("ranging_market"),
                        "ma_7": features.get("ma_7"),
                        "ma_26": features.get("ma_26"),
                        "macd": features.get("macd"),
                        "macd_signal": features.get("macd_signal"),
                        "macd_hist": features.get("macd_hist"),
                        "reflex": exec_result.get("reflex", {}),
                        "nemotron": exec_result.get("nemotron", {}),
                        "market_state": exec_result.get("nemotron", {}).get("market_state_review", {}),
                        "posture": exec_result.get("nemotron", {}).get("posture_review", {}),
                        "timings": format_decision_timings(features_ms, timings),
                        "signal": signal,
                        "risk_checks": risk_checks,
                        "portfolio_decision": nemotron_decision.portfolio_decision if DECISION_ENGINE == "llm" else portfolio_decision,
                        "portfolio_opportunity": portfolio_opportunity,
                        "execution_status": exec_result.get("status"),
                        "execution_reason": exec_result.get("reason"),
                        "order_type": exec_result.get("order_type"),
                        "limit_price": exec_result.get("limit_price"),
                        "spread_pct": exec_result.get("spread_pct", features.get("spread_pct")),
                        "book_imbalance": features.get("book_imbalance"),
                        "book_wall_pressure": features.get("book_wall_pressure"),
                        "short_tf_ready_5m": features.get("short_tf_ready_5m"),
                        "short_tf_ready_15m": features.get("short_tf_ready_15m"),
                        "finbert_score": features.get("finbert_score"),
                        "xgb_score": features.get("xgb_score"),
                        "runtime_health": features.get("runtime_health", {}),
                        "cost": exec_result.get("cost"),
                        "llm_metrics": extract_llm_metrics(symbol, features.get("lane"), exec_result),
                    }
                )
                last_prices[symbol] = float(ohlc_by_symbol[symbol]["close"].iloc[-1])

            completed_steps += 1
            await asyncio.sleep(1)
    finally:
        if nemotron_needed:
            unload_nemotron_model()   # Release GPU VRAM and allow clean lock file deletion on restart
        await cleanup_tasks([collector_task])


def main() -> None:
    lock_path = Path(os.getenv("TRADER_LOCK_PATH", "logs/trader_instance.lock"))
    instance_id = build_trader_instance_id()
    try:
        with TraderSingletonLock(lock_path, instance_id=instance_id) as lock:
            console_event(
                {
                    "ts": now_iso(),
                    "event": "trader_lock_acquired",
                    **lock.metadata,
                },
                force=True,
            )
            asyncio.run(trader_loop(instance_id=instance_id))
    except TraderAlreadyRunningError as exc:
        console_event(
            {
                "ts": now_iso(),
                "event": "trader_lock_conflict",
                "lock_path": str(exc.path),
                "owner": exc.owner_metadata,
                "attempted_instance_id": instance_id,
                "pid": os.getpid(),
            },
            force=True,
        )
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
