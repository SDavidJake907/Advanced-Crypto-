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
    pnl_usd: float
    hold_minutes: float
    exit_reason: str
    entry_reasons: list[str]
    regime_label: str
    tag: str
    lane: str = ""
    entry_score: float = 0.0
    entry_recommendation: str = ""
    pattern_name: str = ""
    setup_key: str = ""
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    capture_vs_mfe_pct: float = 0.0
    structure_state: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0
    etd_pct: float = 0.0
    etd_r: float = 0.0
    expected_edge_pct: float = 0.0
    realized_edge_pct: float = 0.0
    edge_capture_ratio: float = 0.0

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
        self.lessons_path = self.base_dir / "lessons.jsonl"

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        with self.outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(outcome.to_dict()) + "\n")

    @staticmethod
    def _parse_outcome(payload: dict) -> OutcomeRecord:
        """Deserialise one outcome record, tolerating missing or extra fields."""
        import dataclasses
        known = {f.name for f in dataclasses.fields(OutcomeRecord)}
        return OutcomeRecord(**{k: v for k, v in payload.items() if k in known})

    def load_outcomes(self) -> list[OutcomeRecord]:
        if not self.outcomes_path.exists():
            return []
        outcomes: list[OutcomeRecord] = []
        with self.outcomes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    outcomes.append(self._parse_outcome(json.loads(line)))
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
                outcomes.append(self._parse_outcome(json.loads(line)))
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
        if win_rate < 0.25 and trade_count >= 3:
            verdict = "avoid"
        elif win_rate < 0.4 and trade_count >= 2:
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

    def save_lesson(self, symbol: str, outcome_class: str, lesson: str, suggested_adjustment: str, confidence: float) -> None:
        """Persist a lesson from an outcome review so Nemo can learn from it."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "outcome_class": outcome_class,
            "lesson": lesson,
            "suggested_adjustment": suggested_adjustment,
            "confidence": confidence,
        }
        with self.lessons_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def build_lessons_block(self, max_lessons: int = 10) -> str:
        """Return recent lessons as a compact block for LLM injection."""
        if not self.lessons_path.exists():
            return ""
        lines: list[str] = []
        recent: deque[str] = deque(maxlen=max_lessons)
        with self.lessons_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    recent.append(line)
        for line in recent:
            try:
                r = json.loads(line)
                lines.append(
                    f"- [{r.get('outcome_class','?')}] {r.get('symbol','?')}: "
                    f"{r.get('lesson','?')} → adjust: {r.get('suggested_adjustment','?')}"
                )
            except Exception:
                continue
        if not lines:
            return ""
        return "=== NEMO LEARNED LESSONS ===\n" + "\n".join(lines)

    def is_in_cooldown(self, symbol: str, cooldown_minutes: float = 240.0) -> bool:
        """Return True if this symbol was closed within the cooldown window."""
        outcomes = self.load_recent_outcomes(50)
        symbol_outcomes = [o for o in outcomes if o.symbol == symbol]
        if not symbol_outcomes:
            return False
        last = symbol_outcomes[-1]
        try:
            last_ts = datetime.fromisoformat(last.ts)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0
            return elapsed < cooldown_minutes
        except Exception:
            return False

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

    def build_behavior_score_block(self, lookback: int = 50) -> str:
        """Return a formatted behavior score block for LLM injection.

        Returns an empty string when there are insufficient outcomes.
        """
        from core.llm.behavior_score import compute_behavior_score, format_behavior_score_block
        outcomes = self.load_recent_outcomes(lookback)
        bs = compute_behavior_score(outcomes, lookback=lookback)
        if bs is None:
            return ""
        return format_behavior_score_block(bs)

    def build_symbol_lesson_block(self, symbol: str, n: int = 3) -> str:
        """Return the last n outcomes for this symbol as a compact lesson block for LLM injection."""
        all_outcomes = self.load_recent_outcomes(100)
        symbol_outcomes = [o for o in all_outcomes if o.symbol == symbol][-n:]
        if not symbol_outcomes:
            return ""
        recent_losses = [o for o in symbol_outcomes if o.pnl_pct < 0]
        lines = [f"=== RECENT {symbol} LESSONS ==="]
        for o in symbol_outcomes:
            tag = "WIN" if o.pnl_pct >= 0 else "LOSS"
            entry_str = ",".join(o.entry_reasons[:2]) if o.entry_reasons else "unknown"
            lines.append(
                f"- [{tag}] exit={o.exit_reason} pnl={o.pnl_pct * 100.0:+.2f}% "
                f"held={o.hold_minutes:.0f}m entry={entry_str}"
            )
        if len(recent_losses) >= 3:
            lines.append(f"⚠ TEMPORARY AVOID: 3 consecutive losses on {symbol}")
        return "\n".join(lines)


def build_setup_key(
    *,
    lane: str = "",
    pattern_name: str = "",
    entry_recommendation: str = "",
    entry_reasons: list[str] | None = None,
) -> str:
    primary_reason = ""
    if entry_reasons:
        primary_reason = str(entry_reasons[0] or "").strip().lower()
    parts = [
        str(lane or "").strip().upper() or "NA",
        str(pattern_name or "").strip().lower() or "none",
        str(entry_recommendation or "").strip().upper() or "NA",
        primary_reason or "none",
    ]
    return "|".join(parts)


def build_outcome_record(
    *,
    symbol: str,
    side: str,
    lane: str = "",
    pnl_pct: float,
    pnl_usd: float = 0.0,
    hold_minutes: float,
    exit_reason: str,
    entry_reasons: list[str],
    regime_label: str,
    entry_score: float = 0.0,
    entry_recommendation: str = "",
    pattern_name: str = "",
    mfe_pct: float = 0.0,
    mae_pct: float = 0.0,
    capture_vs_mfe_pct: float = 0.0,
    structure_state: str = "",
    mfe_r: float = 0.0,
    mae_r: float = 0.0,
    etd_pct: float = 0.0,
    etd_r: float = 0.0,
    expected_edge_pct: float = 0.0,
    realized_edge_pct: float = 0.0,
    edge_capture_ratio: float = 0.0,
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
        lane=str(lane or ""),
        pnl_pct=pnl_pct,
        pnl_usd=round(pnl_usd, 4),
        hold_minutes=hold_minutes,
        exit_reason=exit_reason,
        entry_reasons=entry_reasons,
        regime_label=regime_label,
        tag=tag,
        entry_score=round(float(entry_score or 0.0), 3),
        entry_recommendation=str(entry_recommendation or ""),
        pattern_name=str(pattern_name or ""),
        setup_key=build_setup_key(
            lane=str(lane or ""),
            pattern_name=str(pattern_name or ""),
            entry_recommendation=str(entry_recommendation or ""),
            entry_reasons=entry_reasons,
        ),
        mfe_pct=round(mfe_pct, 6),
        mae_pct=round(mae_pct, 6),
        capture_vs_mfe_pct=round(capture_vs_mfe_pct, 6),
        structure_state=str(structure_state or ""),
        mfe_r=round(mfe_r, 6),
        mae_r=round(mae_r, 6),
        etd_pct=round(etd_pct, 6),
        etd_r=round(etd_r, 6),
        expected_edge_pct=round(expected_edge_pct, 4),
        realized_edge_pct=round(realized_edge_pct, 4),
        edge_capture_ratio=round(edge_capture_ratio, 4),
    )
