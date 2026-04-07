"""Trader bootstrap — env config, dependency wiring, and live-feed startup.

Responsible for:
- All module-level env var constants (single source of truth for the trader)
- Symbol resolution (_load_trading_symbols)
- Component initialization (strategy, risk, Nemo, executor, state stores)
- Live feed startup and initial account sync
- Returning a TraderComponents snapshot that trader_loop unpacks into locals
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from apps.collector.main import collector_loop, seed_feed_from_snapshots, seed_feed_from_rest
from apps.trader.account_sync_state import build_degraded_account_sync_payload, build_synced_account_sync_payload
from apps.trader.logging_sink import console_event, log_account_sync, now_iso
from core.data.account_sync import bootstrap_account_state
from core.data.kraken_rest import KrakenRestClient
from core.data.live_buffer import LiveMarketDataFeed
from core.data.news_sentiment import NewsSentimentFeed
from core.execution.cpp_exec import CppExecutor
from core.llm.client import cleanup_nemotron_lock, nemotron_provider_name, warm_nemotron_model
from core.llm.client import advisory_provider_name
from core.llm.nemotron import NemotronStrategist
from core.memory.trade_memory import TradeMemoryStore
from core.models.xgb_entry import XGBEntryModel
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState
from core.runtime.trader_lock import build_trader_instance_id
from core.sentiment.finbert_service import FinBertService
from core.state.portfolio import PortfolioState
from core.state.position_state_store import (
    load_position_state,
    merge_persisted_positions,
    save_position_state,
)
from core.state.store import load_synced_position_symbols, load_universe
from core.strategy.simple_momo import SimpleMomentumStrategy
from core.validation.engine_runner import build_engine_components

# ---------------------------------------------------------------------------
# Environment constants — single source of truth for the trader process
# ---------------------------------------------------------------------------

SYMBOLS: list[str] = [
    s.strip() for s in os.getenv("TRADER_SYMBOLS", "ETH/USD").split(",") if s.strip()
]
USE_ACTIVE_UNIVERSE       = os.getenv("USE_ACTIVE_UNIVERSE", "true").lower() == "true"
CANDLES_PATH              = os.getenv("CANDLES_PATH", "logs/candles_ETHUSD.csv")
CANDLES_PATH_TEMPLATE     = os.getenv("CANDLES_PATH_TEMPLATE", "")
CANDLES_PATH_1H           = os.getenv("CANDLES_PATH_1H", "logs/candles_ETHUSD_1h.csv")
CANDLES_PATH_TEMPLATE_1H  = os.getenv("CANDLES_PATH_TEMPLATE_1H", "")
CANDLES_PATH_7D           = os.getenv("CANDLES_PATH_7D", "logs/candles_ETHUSD_7d.csv")
CANDLES_PATH_TEMPLATE_7D  = os.getenv("CANDLES_PATH_TEMPLATE_7D", "")
CANDLES_PATH_30D          = os.getenv("CANDLES_PATH_30D", "logs/candles_ETHUSD_30d.csv")
CANDLES_PATH_TEMPLATE_30D = os.getenv("CANDLES_PATH_TEMPLATE_30D", "")

RUN_ONCE_ON_STATIC        = os.getenv("RUN_ONCE_ON_STATIC", "false").lower() == "true"
DATA_SOURCE               = os.getenv("DATA_SOURCE", "csv").lower()
DECISION_ENGINE           = os.getenv("TRADER_DECISION_ENGINE", "classic").lower()
SHADOW_VALIDATION_ENABLED = os.getenv("SHADOW_VALIDATION_ENABLED", "false").lower() == "true"
SHADOW_DECISION_ENGINE    = os.getenv("SHADOW_DECISION_ENGINE", "llm").lower()
INTRABAR_PRICE_THRESHOLD  = float(os.getenv("INTRABAR_PRICE_THRESHOLD", "0.001"))
LOOP_ALIVE_LOG            = os.getenv("LOOP_ALIVE_LOG", "false").lower() == "true"
ACCOUNT_SYNC_INTERVAL_SEC = float(os.getenv("ACCOUNT_SYNC_INTERVAL_SEC", "30"))
SYMBOL_STATE_RETENTION_SEC = float(os.getenv("SYMBOL_STATE_RETENTION_SEC", "21600"))
NEMOTRON_MAX_PER_CYCLE    = int(os.getenv("NEMOTRON_MAX_PER_CYCLE", "15"))
NEMOTRON_BATCH_MODE       = os.getenv("NEMOTRON_BATCH_MODE", "true").lower() == "true"
NEMOTRON_BATCH_TOP_N      = int(os.getenv("NEMOTRON_BATCH_TOP_N", "10"))
NEMOTRON_BATCH_SKIP_PHI3  = os.getenv("NEMOTRON_BATCH_SKIP_PHI3", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _resolve_candles_path(symbol: str, base_path: str, template: str) -> str:
    if template:
        return template.format(symbol=_symbol_token(symbol))
    return base_path


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def effective_nemotron_batch_mode() -> bool:
    return bool(NEMOTRON_BATCH_MODE)


def load_trading_symbols() -> list[str]:
    """Resolve the active symbol list from universe or env config."""
    if USE_ACTIVE_UNIVERSE:
        active = load_universe().get("active_pairs", [])
        resolved: list[str] = []
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


# ---------------------------------------------------------------------------
# Components container
# ---------------------------------------------------------------------------

@dataclass
class TraderComponents:
    """All initialised objects the trader loop needs, returned by boot_trader()."""
    trader_instance_id: str
    strategy: SimpleMomentumStrategy
    risk_engine: BasicRiskEngine
    portfolio_config: PortfolioConfig
    executor: CppExecutor
    nemotron: NemotronStrategist
    nemotron_needed: bool
    shadow_components: Any | None
    positions_state: PositionState
    portfolio: PortfolioState
    memory_store: TradeMemoryStore
    finbert_service: FinBertService
    xgb_model: XGBEntryModel
    sentiment_feed: NewsSentimentFeed
    live_feed: LiveMarketDataFeed | None
    collector_task: asyncio.Task | None  # type: ignore[type-arg]
    current_symbols: list[str]
    current_account_sync_status: dict[str, Any] = field(default_factory=dict)
    last_xgb_train_ts: float = 0.0


# ---------------------------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------------------------

async def boot_trader(instance_id: str | None = None) -> TraderComponents:
    """Initialise all trader components and return them as a single object.

    Caller is responsible for tearing down (unload Nemo, cancel collector).
    """
    trader_instance_id = instance_id or build_trader_instance_id()

    # Core strategy / risk / execution
    strategy        = SimpleMomentumStrategy()
    risk_engine     = BasicRiskEngine()
    portfolio_config = PortfolioConfig.from_runtime()
    executor        = CppExecutor()

    # Shadow engine (optional)
    shadow_components = (
        build_engine_components(SHADOW_DECISION_ENGINE)
        if SHADOW_VALIDATION_ENABLED
        else None
    )

    # Nemotron warm-up
    nemotron_needed = (
        DECISION_ENGINE == "llm"
        or (SHADOW_VALIDATION_ENABLED and SHADOW_DECISION_ENGINE == "llm")
    )
    if nemotron_needed:
        cleanup_nemotron_lock()   # Remove stale lock from any prior crash
        warm_nemotron_model()     # Pre-load into GPU before first inference

    nemotron = NemotronStrategist(
        strategy=strategy,
        risk_engine=risk_engine,
        portfolio_config=portfolio_config,
        executor=executor,
    )

    # State stores
    positions_state = PositionState()
    portfolio       = PortfolioState()
    memory_store    = TradeMemoryStore()
    finbert_service = FinBertService()
    xgb_model       = XGBEntryModel()
    xgb_model.load_or_init("models/xgb_entry.pkl")

    sentiment_feed = NewsSentimentFeed(
        fetch_interval_sec=float(os.getenv("NEWS_SENTIMENT_INTERVAL_SEC", "300"))
    )

    # Live feed startup
    live_feed: LiveMarketDataFeed | None = None
    collector_task: asyncio.Task | None = None  # type: ignore[type-arg]
    current_symbols: list[str] = load_trading_symbols()
    current_account_sync_status: dict[str, Any] = {}

    if DATA_SOURCE == "live":
        console_event({
            "ts": now_iso(),
            "event": "trader_symbols_loaded",
            "instance_id": trader_instance_id,
            "pid": os.getpid(),
            "use_active_universe": USE_ACTIVE_UNIVERSE,
            "symbols": current_symbols,
        })
        console_event({
            "ts": now_iso(),
            "event": "trader_model_mode",
            "instance_id": trader_instance_id,
            "pid": os.getpid(),
            "advisory_provider": advisory_provider_name(),
            "strategist_provider": nemotron_provider_name(),
            "nemotron_batch_mode_env": NEMOTRON_BATCH_MODE,
            "nemotron_batch_mode_effective": effective_nemotron_batch_mode(),
            "nemotron_max_per_cycle": NEMOTRON_MAX_PER_CYCLE,
            "nemotron_batch_top_n": NEMOTRON_BATCH_TOP_N,
            "nemotron_local_timeout_sec": float(os.getenv("NEMOTRON_LOCAL_TIMEOUT_SEC", "120")),
            "advisory_local_timeout_sec": float(os.getenv("ADVISORY_LOCAL_TIMEOUT_SEC", os.getenv("NEMOTRON_LOCAL_TIMEOUT_SEC", "120"))),
        })

        if os.getenv("SYNC_KRAKEN_ACCOUNT_ON_START", "true").lower() == "true":
            try:
                bootstrap = bootstrap_account_state(
                    client=KrakenRestClient(), symbols=current_symbols
                )
                portfolio = bootstrap.portfolio_state
                positions_state = merge_persisted_positions(
                    bootstrap.positions_state, load_position_state()
                )
                save_position_state(positions_state)
                current_account_sync_status = build_synced_account_sync_payload(bootstrap.to_dict())
            except Exception as exc:
                current_account_sync_status = build_degraded_account_sync_payload(
                    error=exc,
                    previous_payload=None,
                    fallback_payload={
                        "portfolio_state": portfolio.to_dict(),
                        "positions_state": [],
                        "cash_usd": portfolio.cash,
                        "initial_equity_usd": portfolio.initial_equity,
                        "ignored_dust": {},
                        "synced_positions_usd": {},
                        "diagnostics": {"account_sync_disabled": True},
                    },
                )
            console_event(current_account_sync_status)
            log_account_sync(current_account_sync_status)

        live_feed = LiveMarketDataFeed(current_symbols)
        seed_feed_from_snapshots(live_feed, current_symbols)
        seed_feed_from_rest(live_feed, current_symbols)
        collector_task = asyncio.create_task(collector_loop(live_feed))

    return TraderComponents(
        trader_instance_id=trader_instance_id,
        strategy=strategy,
        risk_engine=risk_engine,
        portfolio_config=portfolio_config,
        executor=executor,
        nemotron=nemotron,
        nemotron_needed=nemotron_needed,
        shadow_components=shadow_components,
        positions_state=positions_state,
        portfolio=portfolio,
        memory_store=memory_store,
        finbert_service=finbert_service,
        xgb_model=xgb_model,
        sentiment_feed=sentiment_feed,
        live_feed=live_feed,
        collector_task=collector_task,
        current_symbols=current_symbols,
        current_account_sync_status=current_account_sync_status,
    )
