from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv

from apps.collector.main import collector_loop, seed_feed_from_snapshots
from core.config.runtime import get_proposed_weight
from core.data.account_sync import bootstrap_account_state
from core.data.loader import CandleLoader
from core.data.live_buffer import LiveMarketDataFeed
from core.data.kraken_rest import KrakenRestClient
from core.data import warmup
from core.data.news_sentiment import NewsSentimentFeed
from core.execution.cpp_exec import CppExecutor
from core.features.batch import compute_features_batch, slice_features_for_asset
from core.llm.micro_prompts import nemotron_review_outcome, phi3_review_exit_posture
from core.llm.nemotron import NemotronStrategist
from core.memory.trade_memory import TradeMemoryStore, build_outcome_record
from core.risk.basic_risk import BasicRiskEngine
from core.risk.exits import (
    build_exit_execution,
    build_exit_plan,
    evaluate_exit,
    maybe_apply_exit_posture,
    maybe_arm_break_even,
    maybe_update_trailing,
)
from core.risk.portfolio import PortfolioConfig, Position, PositionState, evaluate_trade
from core.runtime.cleanup import cleanup_tasks
from core.runtime.log_rotation import rotate_jsonl_if_needed
from core.policy.trade_plan import build_trade_plan_metadata
from core.state.portfolio import PortfolioState
from core.state.position_state_store import load_position_state, merge_persisted_positions, save_position_state
from core.state.store import load_synced_position_symbols, load_universe, read_reload_signal
from core.state.trace import DecisionTrace
from core.strategy.simple_momo import SimpleMomentumStrategy
from core.state.open_orders import reconcile_with_positions
from core.sentiment.finbert_service import FinBertService
from core.models.xgb_entry import XGBEntryModel

load_dotenv()

SYMBOLS = [s.strip() for s in os.getenv("TRADER_SYMBOLS", "ETH/USD").split(",") if s.strip()]
USE_ACTIVE_UNIVERSE = os.getenv("USE_ACTIVE_UNIVERSE", "true").lower() == "true"
CANDLES_PATH = os.getenv("CANDLES_PATH", "logs/candles_ETHUSD.csv")
CANDLES_PATH_TEMPLATE = os.getenv("CANDLES_PATH_TEMPLATE", "")
CANDLES_PATH_1H = os.getenv("CANDLES_PATH_1H", "logs/candles_ETHUSD_1h.csv")
CANDLES_PATH_TEMPLATE_1H = os.getenv("CANDLES_PATH_TEMPLATE_1H", "")
CANDLES_PATH_7D = os.getenv("CANDLES_PATH_7D", "logs/candles_ETHUSD_7d.csv")
CANDLES_PATH_TEMPLATE_7D = os.getenv("CANDLES_PATH_TEMPLATE_7D", "")
CANDLES_PATH_30D = os.getenv("CANDLES_PATH_30D", "logs/candles_ETHUSD_30d.csv")
CANDLES_PATH_TEMPLATE_30D = os.getenv("CANDLES_PATH_TEMPLATE_30D", "")
TRACE_LOG_PATH = Path("logs/decision_traces.jsonl")
DEBUG_LOG_PATH = Path("logs/decision_debug.jsonl")
WARMUP_LOG_PATH = Path("logs/warmup.jsonl")
ACCOUNT_SYNC_LOG_PATH = Path("logs/account_sync.json")
OUTCOME_REVIEW_LOG_PATH = Path("logs/outcome_reviews.jsonl")
RUN_ONCE_ON_STATIC = os.getenv("RUN_ONCE_ON_STATIC", "false").lower() == "true"
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv").lower()
DECISION_ENGINE = os.getenv("TRADER_DECISION_ENGINE", "classic").lower()
INTRABAR_PRICE_THRESHOLD = float(os.getenv("INTRABAR_PRICE_THRESHOLD", "0.001"))
LOOP_ALIVE_LOG = os.getenv("LOOP_ALIVE_LOG", "false").lower() == "true"
ACCOUNT_SYNC_INTERVAL_SEC = float(os.getenv("ACCOUNT_SYNC_INTERVAL_SEC", "30"))


def log_trace(trace: DecisionTrace) -> None:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(TRACE_LOG_PATH)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(), default=str) + "\n")


def log_decision_debug(payload: dict[str, object]) -> None:
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(DEBUG_LOG_PATH)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def log_account_sync(payload: dict[str, object]) -> None:
    ACCOUNT_SYNC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNT_SYNC_LOG_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def log_outcome_review(payload: dict[str, object]) -> None:
    OUTCOME_REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(OUTCOME_REVIEW_LOG_PATH)
    with OUTCOME_REVIEW_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _resolve_candles_path(symbol: str, base_path: str, template: str) -> str:
    if template:
        return template.format(symbol=_symbol_token(symbol))
    return base_path


def _load_trading_symbols() -> list[str]:
    if USE_ACTIVE_UNIVERSE:
        active = load_universe().get("active_pairs", [])
        resolved = []
        for symbol in active:
            value = str(symbol).strip().upper()
            if not value:
                continue
            if value.startswith("XBT/"):
                value = value.replace("XBT/", "BTC/", 1)
            resolved.append(value)
    else:
        resolved = list(SYMBOLS)
    merged = list(dict.fromkeys(resolved + load_synced_position_symbols()))
    return merged or SYMBOLS


def _compute_hold_minutes(entry_bar_ts: str | None, exit_bar_ts: str | None) -> float:
    if not entry_bar_ts or not exit_bar_ts:
        return 0.0
    try:
        start = pd.Timestamp(entry_bar_ts)
        end = pd.Timestamp(exit_bar_ts)
        return max((end - start).total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0


def _compute_pnl_pct(side: str, entry_price: float | None, exit_price: float) -> float:
    if entry_price is None or entry_price <= 0.0 or exit_price <= 0.0:
        return 0.0
    if side == "LONG":
        return (exit_price / entry_price) - 1.0
    return (entry_price / exit_price) - 1.0


def _market_sentiment_fallback(symbol: str, sentiment_snapshot: object) -> float:
    fng_value = float(getattr(sentiment_snapshot, "fng_value", 50) or 50.0)
    market_cap_change = float(getattr(sentiment_snapshot, "market_cap_change_24h", 0.0) or 0.0)
    trending = [str(getattr(item, "symbol", "")).upper() for item in getattr(sentiment_snapshot, "trending", [])[:10]]
    base_symbol = symbol.split("/")[0].upper()
    score = max(min((fng_value - 50.0) / 50.0, 1.0), -1.0) * 0.35
    score += max(min(market_cap_change / 4.0, 1.0), -1.0) * 0.45
    if base_symbol in trending:
        score += 0.2
    return max(min(score, 1.0), -1.0)


def _load_lane_supervision_map() -> dict[str, dict[str, object]]:
    universe = load_universe()
    meta = universe.get("meta", {}) if isinstance(universe.get("meta", {}), dict) else {}
    lane_supervision = meta.get("lane_supervision", [])
    if not isinstance(lane_supervision, list):
        return {}
    mapping: dict[str, dict[str, object]] = {}
    for item in lane_supervision:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        mapping[symbol] = {
            "lane_candidate": item.get("lane_candidate"),
            "lane_confidence": item.get("lane_confidence"),
            "lane_reason": item.get("lane_reason"),
            "lane_conflict": item.get("lane_conflict"),
            "narrative_tag": item.get("narrative_tag"),
        }
    return mapping


async def fetch_ohlc(symbol: str, *, base_path: str = CANDLES_PATH, template: str = CANDLES_PATH_TEMPLATE) -> pd.DataFrame:
    loader = CandleLoader(_resolve_candles_path(symbol, base_path, template))
    if not loader.path.exists():
        raise FileNotFoundError(f"Candles CSV not found: {loader.path}")
    df = loader.load()
    return df.tail(200)


async def fetch_live_ohlc(feed: LiveMarketDataFeed, symbol: str, timeframe: str) -> pd.DataFrame:
    return feed.get_ohlc(symbol, timeframe, limit=200)


async def trader_loop(steps: int | None = None) -> None:
    strategy = SimpleMomentumStrategy()
    risk_engine = BasicRiskEngine()
    portfolio_config = PortfolioConfig.from_runtime()
    executor = CppExecutor()
    nemotron = NemotronStrategist(
        strategy=strategy,
        risk_engine=risk_engine,
        portfolio_config=portfolio_config,
        executor=executor,
    )
    positions_state = PositionState()
    portfolio = PortfolioState()
    memory_store = TradeMemoryStore()
    finbert_service = FinBertService()
    xgb_model = XGBEntryModel()
    xgb_model.load_or_init("models/xgb_entry.pkl")
    last_xgb_train_ts = 0.0
    last_bar_keys = None
    last_prices: dict[str, float] = {}
    live_feed = None
    collector_task = None
    market_client = KrakenRestClient()
    sentiment_feed = NewsSentimentFeed(fetch_interval_sec=float(os.getenv("NEWS_SENTIMENT_INTERVAL_SEC", "300")))
    if DATA_SOURCE == "live":
        active_symbols = _load_trading_symbols()
        print(
            {
                "ts": now_iso(),
                "event": "trader_symbols_loaded",
                "use_active_universe": USE_ACTIVE_UNIVERSE,
                "symbols": active_symbols,
            }
        )
        if os.getenv("SYNC_KRAKEN_ACCOUNT_ON_START", "true").lower() == "true":
            try:
                bootstrap = bootstrap_account_state(client=KrakenRestClient(), symbols=active_symbols)
                portfolio = bootstrap.portfolio_state
                positions_state = merge_persisted_positions(bootstrap.positions_state, load_position_state())
                save_position_state(positions_state)
                sync_payload = {
                    "ts": now_iso(),
                    "status": "synced",
                    **bootstrap.to_dict(),
                }
            except Exception as exc:
                sync_payload = {
                    "ts": now_iso(),
                    "status": "sync_failed",
                    "error": str(exc),
                    "portfolio_state": portfolio.to_dict(),
                    "positions_state": [],
                    "cash_usd": portfolio.cash,
                    "initial_equity_usd": portfolio.initial_equity,
                    "ignored_dust": {},
                    "synced_positions_usd": {},
                    "diagnostics": {"account_sync_disabled": True},
                }
            print(sync_payload)
            log_account_sync(sync_payload)
        live_feed = LiveMarketDataFeed(active_symbols)
        seed_feed_from_snapshots(live_feed, active_symbols)
        collector_task = asyncio.create_task(collector_loop(live_feed))

    completed_steps = 0
    was_warm = DATA_SOURCE != "live"
    last_warmup_snapshot = None
    last_account_sync_ts = 0.0
    last_reload_token = read_reload_signal()
    if not positions_state.all():
        positions_state = load_position_state()
    try:
        while steps is None or completed_steps < steps:
            current_reload_token = read_reload_signal()
            if current_reload_token != last_reload_token:
                last_reload_token = current_reload_token
                current_symbols = _load_trading_symbols()
                if DATA_SOURCE == "live" and live_feed is not None:
                    live_feed.set_symbols(current_symbols)
                last_account_sync_ts = 0.0
                print({"ts": now_iso(), "event": "reload_applied", "symbols": current_symbols})
            portfolio_config = PortfolioConfig.from_runtime()
            nemotron.portfolio_config = portfolio_config
            current_symbols = _load_trading_symbols()
            if DATA_SOURCE == "live" and executor.mode == "live" and (time.time() - last_account_sync_ts) >= ACCOUNT_SYNC_INTERVAL_SEC:
                try:
                    bootstrap = bootstrap_account_state(client=KrakenRestClient(), symbols=current_symbols)
                    portfolio = bootstrap.portfolio_state
                    positions_state = merge_persisted_positions(bootstrap.positions_state, load_position_state())
                    save_position_state(positions_state)
                    reconcile_with_positions(portfolio.positions)
                    sync_payload = {
                        "ts": now_iso(),
                        "status": "synced",
                        **bootstrap.to_dict(),
                    }
                    log_account_sync(sync_payload)
                    last_account_sync_ts = time.time()
                except Exception as exc:
                    log_account_sync({"ts": now_iso(), "status": "sync_failed", "error": str(exc)})
            try:
                await sentiment_feed.maybe_update()
            except Exception:
                pass
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
                        fallback_score=_market_sentiment_fallback(_sym, sentiment_snapshot),
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
                        print(warmup_payload)
                        warmup.emit_warmup_status(WARMUP_LOG_PATH, warmup_payload)
                        last_warmup_snapshot = warmup_state
                    await asyncio.sleep(1)
                    continue
                ready_symbols = warmup_info.symbols_ready
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
                    print(warmup_payload)
                    warmup.emit_warmup_status(WARMUP_LOG_PATH, warmup_payload)
                    was_warm = True
                if not ready_symbols:
                    await asyncio.sleep(1)
                    continue
                eval_symbols = ready_symbols
                ohlc_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "1m") for symbol in eval_symbols}
                ohlc_5m_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "5m") for symbol in eval_symbols}
                ohlc_15m_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "15m") for symbol in eval_symbols}
                ohlc_1h_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "1h") for symbol in eval_symbols}
                ohlc_7d_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "7d") for symbol in eval_symbols}
                ohlc_30d_by_symbol = {symbol: await fetch_live_ohlc(live_feed, symbol, "30d") for symbol in eval_symbols}
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

            try:
                ticker_snapshot = market_client.get_ticker_snapshot(eval_symbols)
            except Exception:
                ticker_snapshot = {}

            if any(frame.empty for frame in ohlc_by_symbol.values()):
                await asyncio.sleep(1)
                continue
            for symbol in eval_symbols:
                if symbol not in last_prices:
                    last_prices[symbol] = float(ohlc_by_symbol[symbol]["close"].iloc[-1])
            current_bar_keys = tuple(
                pd.Timestamp(ohlc_by_symbol[symbol]["timestamp"].iloc[-1]).isoformat() for symbol in eval_symbols
            )

            if last_bar_keys is not None and current_bar_keys == last_bar_keys:
                intrabar_move = False
                intrabar_prices: dict[str, float] = {}

                for symbol in eval_symbols:
                    price = float(ohlc_by_symbol[symbol]["close"].iloc[-1])
                    intrabar_prices[symbol] = price
                    last_price = last_prices.get(symbol)
                    if last_price is None or last_price <= 0.0:
                        continue

                    change = abs(price - last_price) / last_price
                    if change >= INTRABAR_PRICE_THRESHOLD:
                        intrabar_move = True
                        break

                if not intrabar_move:
                    if LOOP_ALIVE_LOG:
                        print(
                            {
                                "loop": "alive",
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "bar_keys": current_bar_keys,
                                "prices": intrabar_prices,
                                "intrabar_triggered": False,
                            }
                        )
                    if RUN_ONCE_ON_STATIC:
                        break
                    await asyncio.sleep(1)
                    continue

            last_bar_keys = current_bar_keys
            features_started = time.perf_counter()
            features_batch = compute_features_batch(
                ohlc_by_symbol,
                ohlc_5m_by_symbol=ohlc_5m_by_symbol,
                ohlc_15m_by_symbol=ohlc_15m_by_symbol,
                ohlc_1h_by_symbol=ohlc_1h_by_symbol,
                ohlc_7d_by_symbol=ohlc_7d_by_symbol,
                ohlc_30d_by_symbol=ohlc_30d_by_symbol,
                finbert_scores=finbert_scores,
                xgb_model=xgb_model,
            )
            features_ms = (time.perf_counter() - features_started) * 1000.0
            lane_supervision_map = _load_lane_supervision_map()

            for asset_idx, symbol in enumerate(features_batch["symbols"]):
                features = slice_features_for_asset(features_batch, asset_idx)
                lane_supervision = lane_supervision_map.get(symbol)
                if lane_supervision:
                    features.update(lane_supervision)
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
                    features["quote_volume"] = float(market_snapshot.get("quote_volume", 0.0) or 0.0)
                    features["book_bid_depth"] = float(market_snapshot.get("book_bid_depth", 0.0) or 0.0)
                    features["book_ask_depth"] = float(market_snapshot.get("book_ask_depth", 0.0) or 0.0)
                    features["book_imbalance"] = float(market_snapshot.get("book_imbalance", 0.0) or 0.0)
                    features["book_wall_pressure"] = float(market_snapshot.get("book_wall_pressure", 0.0) or 0.0)
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
                    updated_position = maybe_arm_break_even(existing_position, float(features.get("price", 0.0)))
                    updated_position = maybe_update_trailing(
                        updated_position,
                        float(features.get("price", 0.0)),
                        float(features.get("atr", 0.0)),
                    )
                    hold_minutes = _compute_hold_minutes(updated_position.entry_bar_ts, features.get("bar_ts"))
                    posture = phi3_review_exit_posture(
                        {
                            "symbol": symbol,
                            "lane": updated_position.lane,
                            "side": updated_position.side,
                            "price": float(features.get("price", 0.0)),
                            "entry_price": float(updated_position.entry_price or 0.0),
                            "pnl_pct": _compute_pnl_pct(
                                updated_position.side,
                                updated_position.entry_price,
                                float(features.get("price", 0.0)),
                            )
                            * 100.0,
                            "hold_minutes": hold_minutes,
                            "atr": float(features.get("atr", 0.0)),
                            "rsi": float(features.get("rsi", 50.0)),
                            "momentum": float(features.get("momentum", 0.0)),
                            "momentum_5": float(features.get("momentum_5", 0.0)),
                            "momentum_14": float(features.get("momentum_14", 0.0)),
                            "trend_1h": int(features.get("trend_1h", 0)),
                            "regime_7d": str(features.get("regime_7d", "unknown")),
                            "macro_30d": str(features.get("macro_30d", "unknown")),
                            "entry_thesis": updated_position.entry_thesis,
                            "expected_hold_style": updated_position.expected_hold_style,
                            "invalidate_on": updated_position.invalidate_on,
                        }
                    )
                    updated_position = maybe_apply_exit_posture(
                        updated_position,
                        price=float(features.get("price", 0.0)),
                        atr=float(features.get("atr", 0.0)),
                        posture=posture,
                    )
                    positions_state.add_or_update(updated_position)
                    save_position_state(positions_state)
                    exit_reason = evaluate_exit(updated_position, float(features.get("price", 0.0)))
                    if exit_reason:
                        exec_result = executor.execute_exit(
                            symbol=symbol,
                            side=updated_position.side,
                            qty=abs(live_qty),
                            price=float(features.get("price", 0.0)),
                            features=features,
                            exit_reason=exit_reason,
                        )
                        exec_result["bar_ts"] = features.get("bar_ts")
                        exec_result["bar_idx"] = features.get("bar_idx")
                        state_change = portfolio.apply_execution(exec_result)
                        if exec_result.get("status") == "filled":
                            positions_state.remove(symbol)
                            save_position_state(positions_state)
                            pnl_pct = _compute_pnl_pct(
                                updated_position.side,
                                updated_position.entry_price,
                                float(exec_result["price"]),
                            )
                            hold_minutes = _compute_hold_minutes(updated_position.entry_bar_ts, features.get("bar_ts"))
                            memory_store.append_outcome(
                                build_outcome_record(
                                    symbol=symbol,
                                    side=updated_position.side,
                                    pnl_pct=pnl_pct,
                                    hold_minutes=hold_minutes,
                                    exit_reason=exit_reason,
                                    entry_reasons=list(updated_position.entry_reasons),
                                    regime_label=str(features.get("regime_7d", "unknown")),
                                )
                            )
                            outcome_review = nemotron_review_outcome(
                                {
                                    "symbol": symbol,
                                    "lane": features.get("lane"),
                                    "side": updated_position.side,
                                    "pnl_pct": pnl_pct,
                                    "hold_minutes": hold_minutes,
                                    "exit_reason": exit_reason,
                                    "entry_reasons": list(updated_position.entry_reasons),
                                    "regime_label": str(features.get("regime_7d", "unknown")),
                                    "entry_score": features.get("entry_score"),
                                    "entry_recommendation": features.get("entry_recommendation"),
                                    "reversal_risk": features.get("reversal_risk"),
                                    "exit_posture": updated_position.exit_posture,
                                    "exit_posture_reason": updated_position.exit_posture_reason,
                                    "exit_posture_confidence": updated_position.exit_posture_confidence,
                                    "entry_thesis": updated_position.entry_thesis,
                                    "expected_hold_style": updated_position.expected_hold_style,
                                    "invalidate_on": updated_position.invalidate_on,
                                }
                            ).to_dict()
                            log_outcome_review(
                                {
                                    "ts": now_iso(),
                                    "symbol": symbol,
                                    "lane": features.get("lane"),
                                    "side": updated_position.side,
                                    "pnl_pct": pnl_pct,
                                    "hold_minutes": hold_minutes,
                                    "exit_reason": exit_reason,
                                    "exit_posture": updated_position.exit_posture,
                                    "exit_posture_reason": updated_position.exit_posture_reason,
                                    "exit_posture_confidence": updated_position.exit_posture_confidence,
                                    "entry_thesis": updated_position.entry_thesis,
                                    "expected_hold_style": updated_position.expected_hold_style,
                                    "invalidate_on": updated_position.invalidate_on,
                                    "review": outcome_review,
                                }
                            )
                        trace = DecisionTrace(
                            timestamp=datetime.utcnow(),
                            symbol=symbol,
                            features=features,
                            signal="EXIT",
                            risk_checks=[exit_reason],
                            execution=exec_result,
                            state_change=state_change,
                        )
                        print(trace.to_dict())
                        log_trace(trace)
                        log_decision_debug(
                            {
                                "ts": now_iso(),
                                "symbol": symbol,
                                "lane": features.get("lane"),
                                "phase": "exit",
                                "price": features.get("price"),
                                "atr": features.get("atr"),
                                "entry_price": updated_position.entry_price,
                                "stop_loss": updated_position.stop_loss,
                                "take_profit": updated_position.take_profit,
                                "trail_stop": updated_position.trail_stop,
                                "exit_posture": updated_position.exit_posture,
                                "exit_posture_reason": updated_position.exit_posture_reason,
                                "exit_posture_confidence": updated_position.exit_posture_confidence,
                                "entry_thesis": updated_position.entry_thesis,
                                "expected_hold_style": updated_position.expected_hold_style,
                                "invalidate_on": updated_position.invalidate_on,
                                "exit_reason": exit_reason,
                                "execution_status": exec_result.get("status"),
                            }
                        )
                        last_prices[symbol] = float(ohlc_by_symbol[symbol]["close"].iloc[-1])
                        continue
                    log_decision_debug(
                        {
                            "ts": now_iso(),
                            "symbol": symbol,
                            "lane": updated_position.lane,
                            "phase": "hold_manager",
                            "price": features.get("price"),
                            "entry_price": updated_position.entry_price,
                            "hold_minutes": hold_minutes,
                            "trail_stop": updated_position.trail_stop,
                            "stop_loss": updated_position.stop_loss,
                            "take_profit": updated_position.take_profit,
                            "exit_posture": updated_position.exit_posture,
                            "exit_posture_reason": updated_position.exit_posture_reason,
                            "exit_posture_confidence": updated_position.exit_posture_confidence,
                            "entry_thesis": updated_position.entry_thesis,
                            "expected_hold_style": updated_position.expected_hold_style,
                            "invalidate_on": updated_position.invalidate_on,
                        }
                    )

                if DECISION_ENGINE == "llm":
                    nemotron_decision = nemotron.decide(
                        symbol=symbol,
                        features=features,
                        portfolio_state=portfolio,
                        positions_state=positions_state,
                        symbols=features_batch["symbols"],
                        proposed_weight=proposed_weight,
                    )
                    signal = nemotron_decision.signal
                    risk_checks = nemotron_decision.risk_checks
                    exec_result = nemotron_decision.execution
                    timings = dict(nemotron_decision.timings)
                    state_change = {}
                    if exec_result.get("status") == "filled":
                        state_change = portfolio.apply_execution(exec_result)
                        weight = proposed_weight * nemotron_decision.portfolio_decision["size_factor"]
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
                    portfolio_decision = {"decision": "allow", "size_factor": 1.0, "reasons": []}
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
                        )
                    if portfolio_decision["reasons"]:
                        risk_checks.extend(portfolio_decision["reasons"])
                    if portfolio_decision["decision"] == "block" and "block" not in risk_checks:
                        risk_checks.append("block")

                    if "block" in risk_checks:
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
                                )
                            )
                            save_position_state(positions_state)

                trace = DecisionTrace(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    features=features,
                    signal=signal,
                    risk_checks=risk_checks,
                    execution=exec_result,
                    state_change=state_change,
                )
                print(trace.to_dict())
                log_trace(trace)
                log_decision_debug(
                    {
                        "ts": now_iso(),
                        "symbol": symbol,
                        "lane": features.get("lane"),
                        "price": features.get("price"),
                        "rotation_score": features.get("rotation_score"),
                        "entry_score": features.get("entry_score"),
                        "entry_recommendation": features.get("entry_recommendation"),
                        "reversal_risk": features.get("reversal_risk"),
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
                        "timings": {
                            "features_ms": round(features_ms, 2),
                            "phi3_ms": timings.get("phi3_ms", 0.0),
                            "advisory_ms": timings.get("advisory_ms", 0.0),
                            "nemotron_ms": timings.get("nemotron_ms", 0.0),
                            "execution_ms": timings.get("execution_ms", 0.0),
                            "total_ms": max(round(features_ms, 2), 0.0) + timings.get("total_ms", 0.0),
                        },
                        "signal": signal,
                        "risk_checks": risk_checks,
                        "portfolio_decision": nemotron_decision.portfolio_decision if DECISION_ENGINE == "llm" else portfolio_decision,
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
                        "cost": exec_result.get("cost"),
                    }
                )
                last_prices[symbol] = float(ohlc_by_symbol[symbol]["close"].iloc[-1])

            completed_steps += 1
            await asyncio.sleep(1)
    finally:
        await cleanup_tasks([collector_task])


def main() -> None:
    asyncio.run(trader_loop())


if __name__ == "__main__":
    main()
