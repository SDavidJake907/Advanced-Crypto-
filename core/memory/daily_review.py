from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.memory.trade_memory import MEMORY_DIR, OutcomeRecord, TradeMemoryStore


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LATEST_DAILY_REVIEW_PATH = LOG_DIR / "daily_value_latest.json"
DAILY_REVIEW_HISTORY_PATH = LOG_DIR / "daily_value_history.jsonl"


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    recent: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                recent.append(line)
    items: list[dict[str, Any]] = []
    for line in recent:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
        except Exception:
            continue
    return items


def _load_recent_lessons(limit: int = 20, lessons_path: Path | None = None) -> list[dict[str, Any]]:
    path = lessons_path or (MEMORY_DIR / "lessons.jsonl")
    return _jsonl_tail(path, limit)


def _book_value_usd(account_sync: dict[str, Any]) -> float:
    cash = float(account_sync.get("cash_usd", 0.0) or 0.0)
    synced_positions_usd = account_sync.get("synced_positions_usd", {})
    book_value = cash
    if isinstance(synced_positions_usd, dict):
        book_value += sum(float(value or 0.0) for value in synced_positions_usd.values())
    return round(book_value, 4)


def _same_day(ts: datetime | None, day: str) -> bool:
    return bool(ts and ts.date().isoformat() == day)


def build_daily_review_report(
    *,
    account_sync: dict[str, Any],
    outcomes: list[OutcomeRecord],
    decision_debug: list[dict[str, Any]] | None = None,
    lessons: list[dict[str, Any]] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    as_of_dt = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
    day = as_of_dt.date().isoformat()

    today_outcomes = [o for o in outcomes if _same_day(_parse_ts(o.ts), day)]
    today_lessons = [item for item in (lessons or []) if _same_day(_parse_ts(item.get("ts")), day)]
    today_decisions = [item for item in (decision_debug or []) if _same_day(_parse_ts(item.get("ts")), day)]

    book_value_usd = _book_value_usd(account_sync)
    baseline_equity_usd = float(account_sync.get("initial_equity_usd", 0.0) or 0.0)
    open_pnl_usd = round(book_value_usd - baseline_equity_usd, 4)

    realized_pnl_usd = round(sum(float(o.pnl_usd or 0.0) for o in today_outcomes), 4)
    closed_trade_count = len(today_outcomes)
    win_count = sum(1 for o in today_outcomes if float(o.pnl_pct or 0.0) > 0.0)
    loss_count = sum(1 for o in today_outcomes if float(o.pnl_pct or 0.0) < 0.0)
    win_rate = (win_count / closed_trade_count) if closed_trade_count else 0.0
    avg_pnl_pct = (
        sum(float(o.pnl_pct or 0.0) for o in today_outcomes) / closed_trade_count
        if closed_trade_count else 0.0
    )

    by_symbol: dict[str, dict[str, float | int]] = defaultdict(lambda: {"pnl_usd": 0.0, "trades": 0})
    for outcome in today_outcomes:
        stats = by_symbol[outcome.symbol]
        stats["pnl_usd"] = float(stats["pnl_usd"]) + float(outcome.pnl_usd or 0.0)
        stats["trades"] = int(stats["trades"]) + 1

    ranked_symbols = sorted(
        (
            {"symbol": symbol, "pnl_usd": round(float(stats["pnl_usd"]), 4), "trades": int(stats["trades"])}
            for symbol, stats in by_symbol.items()
        ),
        key=lambda item: item["pnl_usd"],
        reverse=True,
    )
    top_winners = [item for item in ranked_symbols[:3] if item["pnl_usd"] > 0.0]
    top_losers = [item for item in sorted(ranked_symbols, key=lambda item: item["pnl_usd"])[:3] if item["pnl_usd"] < 0.0]

    blocked_reason_counter: Counter[str] = Counter()
    blocked_count = 0
    for item in today_decisions:
        if str(item.get("execution_status", "")).strip().lower() != "blocked":
            continue
        blocked_count += 1
        reasons = item.get("execution_reason")
        if isinstance(reasons, list):
            for reason in reasons:
                text = str(reason).strip()
                if text:
                    blocked_reason_counter[text] += 1
        else:
            text = str(reasons or "").strip()
            if text:
                blocked_reason_counter[text] += 1

    blocked_reasons = [
        {"reason": reason, "count": count}
        for reason, count in blocked_reason_counter.most_common(5)
    ]

    recent_lessons = []
    for item in today_lessons[-3:]:
        lesson = str(item.get("lesson", "")).strip()
        symbol = str(item.get("symbol", "")).strip()
        adjustment = str(item.get("suggested_adjustment", "")).strip()
        if lesson:
            recent_lessons.append(
                {
                    "symbol": symbol,
                    "lesson": lesson,
                    "suggested_adjustment": adjustment,
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                }
            )

    headline_parts = [
        f"value ${book_value_usd:.2f}",
        f"open_pnl ${open_pnl_usd:+.2f}",
        f"realized ${realized_pnl_usd:+.2f}",
    ]
    if closed_trade_count:
        headline_parts.append(f"closed {closed_trade_count}")
        headline_parts.append(f"win_rate {win_rate:.0%}")
    if blocked_count:
        headline_parts.append(f"blocked {blocked_count}")

    return {
        "ts": as_of_dt.isoformat(),
        "day": day,
        "account_value": {
            "book_value_usd": book_value_usd,
            "baseline_equity_usd": round(baseline_equity_usd, 4),
            "open_pnl_usd": open_pnl_usd,
            "cash_usd": round(float(account_sync.get("cash_usd", 0.0) or 0.0), 4),
            "position_count": len(account_sync.get("positions_state", []) or []),
        },
        "trade_day": {
            "closed_trade_count": closed_trade_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 4),
            "realized_pnl_usd": realized_pnl_usd,
            "avg_pnl_pct": round(avg_pnl_pct, 6),
            "top_winners": top_winners,
            "top_losers": top_losers,
        },
        "execution_review": {
            "blocked_trade_count": blocked_count,
            "blocked_reasons": blocked_reasons,
        },
        "learning_review": {
            "lesson_count": len(today_lessons),
            "recent_lessons": recent_lessons,
        },
        "summary": " | ".join(headline_parts),
    }


def build_daily_review_report_from_disk(
    *,
    root: Path | None = None,
    as_of: datetime | None = None,
    decision_limit: int = 800,
) -> dict[str, Any]:
    base = root or ROOT
    log_dir = base / "logs"
    account_sync_path = log_dir / "account_sync.json"
    account_sync: dict[str, Any] = {}
    if account_sync_path.exists():
        try:
            payload = json.loads(account_sync_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                account_sync = payload
        except Exception:
            account_sync = {}

    return build_daily_review_report(
        account_sync=account_sync,
        outcomes=TradeMemoryStore().load_outcomes(),
        decision_debug=_jsonl_tail(log_dir / "decision_debug.jsonl", decision_limit),
        lessons=_load_recent_lessons(),
        as_of=as_of,
    )


def write_daily_review_report(report: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DAILY_REVIEW_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with DAILY_REVIEW_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report) + "\n")
