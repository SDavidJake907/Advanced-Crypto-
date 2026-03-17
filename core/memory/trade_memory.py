from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from collections import deque


MEMORY_DIR = Path("data/trade_memory")


@dataclass
class OutcomeRecord:
    ts: str
    symbol: str
    side: str
    pnl_pct: float
    hold_minutes: float
    exit_reason: str
    entry_reasons: list[str]
    regime_label: str
    tag: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoinTrackRecord:
    symbol: str
    trade_count: int
    win_count: int
    loss_count: int
    avg_pnl_pct: float
    comment: str


class TradeMemoryStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or MEMORY_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes_path = self.base_dir / "outcomes.jsonl"

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        with self.outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(outcome.to_dict()) + "\n")

    def load_outcomes(self) -> list[OutcomeRecord]:
        if not self.outcomes_path.exists():
            return []
        outcomes: list[OutcomeRecord] = []
        with self.outcomes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    outcomes.append(OutcomeRecord(**payload))
                except Exception:
                    continue
        return outcomes

    def load_recent_outcomes(self, limit: int) -> list[OutcomeRecord]:
        if limit <= 0 or not self.outcomes_path.exists():
            return []
        recent_lines: deque[str] = deque(maxlen=limit)
        with self.outcomes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    recent_lines.append(line)
        outcomes: list[OutcomeRecord] = []
        for line in recent_lines:
            try:
                payload = json.loads(line)
                outcomes.append(OutcomeRecord(**payload))
            except Exception:
                continue
        return outcomes

    def compute_track_record(self) -> list[CoinTrackRecord]:
        grouped: dict[str, dict[str, float | int]] = {}
        if not self.outcomes_path.exists():
            return []
        with self.outcomes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    outcome = OutcomeRecord(**payload)
                except Exception:
                    continue
                stats = grouped.setdefault(
                    outcome.symbol,
                    {"trade_count": 0, "win_count": 0, "loss_count": 0, "sum_pnl_pct": 0.0},
                )
                stats["trade_count"] = int(stats["trade_count"]) + 1
                stats["sum_pnl_pct"] = float(stats["sum_pnl_pct"]) + outcome.pnl_pct
                if outcome.pnl_pct > 0.001:
                    stats["win_count"] = int(stats["win_count"]) + 1
                elif outcome.pnl_pct < -0.001:
                    stats["loss_count"] = int(stats["loss_count"]) + 1
        records: list[CoinTrackRecord] = []
        for symbol, stats in grouped.items():
            trade_count = int(stats["trade_count"])
            wins = int(stats["win_count"])
            losses = int(stats["loss_count"])
            avg = float(stats["sum_pnl_pct"]) / max(trade_count, 1)
            comment = ""
            if trade_count >= 4 and wins / max(trade_count, 1) < 0.2:
                comment = "stop_trading_this_coin"
            elif trade_count >= 3 and losses / max(trade_count, 1) > 0.6:
                comment = "entry_timing_issue"
            records.append(
                CoinTrackRecord(
                    symbol=symbol,
                    trade_count=trade_count,
                    win_count=wins,
                    loss_count=losses,
                    avg_pnl_pct=avg,
                    comment=comment,
                )
            )
        records.sort(key=lambda item: item.trade_count, reverse=True)
        return records

    def get_kelly_fraction(self, symbol: str, min_trades: int = 5) -> float:
        """Compute Kelly criterion fraction for position sizing based on recent symbol outcomes."""
        all_outcomes = self.load_outcomes()
        symbol_outcomes = [o for o in all_outcomes if o.symbol == symbol]
        recent = symbol_outcomes[-30:] if len(symbol_outcomes) > 30 else symbol_outcomes
        if len(recent) < min_trades:
            return 1.0
        wins = [o.pnl_pct for o in recent if o.pnl_pct > 0.0]
        losses = [o.pnl_pct for o in recent if o.pnl_pct < 0.0]
        if not wins or not losses:
            return 1.0
        win_rate = len(wins) / len(recent)
        avg_win_pct = float(sum(wins) / len(wins))
        avg_loss_pct = float(abs(sum(losses) / len(losses)))
        if avg_win_pct <= 0.0:
            return 1.0
        kelly = (win_rate * avg_win_pct - (1.0 - win_rate) * avg_loss_pct) / avg_win_pct
        return float(max(0.1, min(1.0, kelly)))

    def get_symbol_performance(self, symbol: str, lookback: int = 20) -> dict:
        """Return performance summary dict for the given symbol."""
        all_outcomes = self.load_outcomes()
        symbol_outcomes = [o for o in all_outcomes if o.symbol == symbol]
        recent = symbol_outcomes[-lookback:] if len(symbol_outcomes) > lookback else symbol_outcomes
        trade_count = len(recent)
        if trade_count == 0:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "recent_streak": 0,
                "kelly_fraction": 1.0,
                "verdict": "neutral",
            }
        wins = [o for o in recent if o.pnl_pct > 0.0]
        win_rate = len(wins) / trade_count
        avg_pnl_pct = float(sum(o.pnl_pct for o in recent) / trade_count)
        # compute streak from most recent trades
        streak = 0
        for o in reversed(recent):
            if streak == 0:
                streak = 1 if o.pnl_pct > 0.0 else -1
            elif streak > 0 and o.pnl_pct > 0.0:
                streak += 1
            elif streak < 0 and o.pnl_pct < 0.0:
                streak -= 1
            else:
                break
        kelly_fraction = self.get_kelly_fraction(symbol)
        if win_rate < 0.25 and trade_count >= 5:
            verdict = "avoid"
        elif win_rate < 0.4:
            verdict = "weak"
        elif win_rate > 0.6 and avg_pnl_pct > 0.005:
            verdict = "strong"
        else:
            verdict = "neutral"
        return {
            "trade_count": trade_count,
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl_pct,
            "recent_streak": streak,
            "kelly_fraction": kelly_fraction,
            "verdict": verdict,
        }

    def build_memory_block(self) -> str:
        records = self.compute_track_record()
        if not records:
            return ""
        lines = ["=== TRADE MEMORY ==="]
        for record in records[:10]:
            lines.append(
                f"- {record.symbol}: {record.trade_count} trades, {record.win_count}W/{record.loss_count}L, "
                f"avg {record.avg_pnl_pct * 100.0:+.2f}%"
                + (f" -- {record.comment}" if record.comment else "")
            )
        return "\n".join(lines)


def build_outcome_record(
    *,
    symbol: str,
    side: str,
    pnl_pct: float,
    hold_minutes: float,
    exit_reason: str,
    entry_reasons: list[str],
    regime_label: str,
) -> OutcomeRecord:
    tag = "normal"
    if pnl_pct > 0.02:
        tag = "big_win"
    elif pnl_pct < -0.015:
        tag = "big_loss"
    elif "stop_loss" in exit_reason:
        tag = "stop_loss"
    elif "take_profit" in exit_reason:
        tag = "take_profit"
    return OutcomeRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        side=side,
        pnl_pct=pnl_pct,
        hold_minutes=hold_minutes,
        exit_reason=exit_reason,
        entry_reasons=entry_reasons,
        regime_label=regime_label,
        tag=tag,
    )
