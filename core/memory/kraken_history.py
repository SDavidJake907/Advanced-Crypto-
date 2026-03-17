from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import zipfile

from core.symbols import normalize_symbol


@dataclass
class HistoricalTrade:
    ts: str
    symbol: str
    side: str
    order_type: str
    price: float
    cost: float
    fee: float
    volume: float


@dataclass
class HistoricalSymbolSummary:
    symbol: str
    realized_pnl_usd: float
    closed_trades: int
    win_rate: float
    avg_trade_pnl_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trades_from_zip(zip_path: str | Path) -> list[HistoricalTrade]:
    path = Path(zip_path)
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names:
            return []
        with archive.open(names[0], "r") as handle:
            text = handle.read().decode("utf-8", errors="replace").splitlines()
    reader = csv.DictReader(text)
    trades: list[HistoricalTrade] = []
    for row in reader:
        try:
            symbol = normalize_symbol(str(row.get("pair", "")).replace("XBT", "BTC").replace("XDG", "DOGE"))
            trades.append(
                HistoricalTrade(
                    ts=str(row.get("time", "")),
                    symbol=symbol,
                    side=str(row.get("type", "")).lower(),
                    order_type=str(row.get("ordertype", "")).lower(),
                    price=float(row.get("price", 0.0) or 0.0),
                    cost=float(row.get("cost", 0.0) or 0.0),
                    fee=float(row.get("fee", 0.0) or 0.0),
                    volume=float(row.get("vol", 0.0) or 0.0),
                )
            )
        except Exception:
            continue
    trades.sort(key=lambda item: item.ts)
    return trades


def summarize_trades(trades: list[HistoricalTrade]) -> list[HistoricalSymbolSummary]:
    lots: dict[str, list[tuple[float, float]]] = {}
    per_symbol_pnls: dict[str, list[float]] = {}
    for trade in trades:
        symbol_lots = lots.setdefault(trade.symbol, [])
        if trade.side == "buy":
            unit_cost = (trade.cost + trade.fee) / trade.volume if trade.volume > 0 else 0.0
            symbol_lots.append((trade.volume, unit_cost))
            continue
        remaining = trade.volume
        sell_unit = (trade.cost - trade.fee) / trade.volume if trade.volume > 0 else 0.0
        realized = 0.0
        while remaining > 0 and symbol_lots:
            lot_vol, lot_cost = symbol_lots[0]
            used = min(remaining, lot_vol)
            realized += used * (sell_unit - lot_cost)
            remaining -= used
            lot_vol -= used
            if lot_vol <= 1e-12:
                symbol_lots.pop(0)
            else:
                symbol_lots[0] = (lot_vol, lot_cost)
        per_symbol_pnls.setdefault(trade.symbol, []).append(realized)
    summaries: list[HistoricalSymbolSummary] = []
    for symbol, pnls in per_symbol_pnls.items():
        closed = len(pnls)
        wins = sum(1 for pnl in pnls if pnl > 0)
        avg = sum(pnls) / closed if closed else 0.0
        summaries.append(
            HistoricalSymbolSummary(
                symbol=symbol,
                realized_pnl_usd=sum(pnls),
                closed_trades=closed,
                win_rate=(wins / closed) if closed else 0.0,
                avg_trade_pnl_usd=avg,
            )
        )
    summaries.sort(key=lambda item: item.realized_pnl_usd, reverse=True)
    return summaries


def build_history_review_block(zip_path: str | Path, max_symbols: int = 10) -> str:
    trades = load_trades_from_zip(zip_path)
    if not trades:
        return ""
    summaries = summarize_trades(trades)
    if not summaries:
        return ""
    lines = ["=== KRAKEN HISTORY REVIEW ==="]
    for item in summaries[:max_symbols]:
        lines.append(
            f"- {item.symbol}: closed={item.closed_trades}, pnl=${item.realized_pnl_usd:+.2f}, "
            f"win_rate={item.win_rate:.0%}, avg=${item.avg_trade_pnl_usd:+.2f}"
        )
    return "\n".join(lines)
