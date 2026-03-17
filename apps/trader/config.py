from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TraderConfig:
    symbol: str = os.getenv("TRADER_SYMBOL", "ETH/USD")
    loop_interval_sec: int = int(os.getenv("LOOP_INTERVAL_SEC", "5"))
    candles_15m_dir: str = os.getenv("CANDLES_15M_DIR", "logs/candles_15m")
    breakout_lookback: int = int(os.getenv("BREAKOUT_LOOKBACK", "20"))
    volume_lookback: int = int(os.getenv("VOLUME_LOOKBACK", "20"))
    min_volume_ratio: float = float(os.getenv("MIN_VOLUME_RATIO", "1.2"))
    max_position_usd: float = float(os.getenv("MAX_POSITION_USD", "25"))
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
    account_equity_usd: float = float(os.getenv("ACCOUNT_EQUITY_USD", "67"))
    trace_path: str = os.getenv("DECISION_TRACE_PATH", "logs/decision_trace.jsonl")

