from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTCOMES_PATH = ROOT / "data" / "trade_memory" / "outcomes.jsonl"
DEBUG_LOG_PATH = ROOT / "logs" / "decision_debug.jsonl"
SYSTEM_RECORD_DB_PATH = ROOT / "logs" / "system_record.sqlite3"
RUNTIME_OVERRIDES_PATH = ROOT / "configs" / "runtime_overrides.json"


def _load_runtime_cutoff() -> str | None:
    if not RUNTIME_OVERRIDES_PATH.exists():
        return None
    try:
        payload = json.loads(RUNTIME_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("updated_at")
    return str(value).strip() if value else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}%"


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _summarize_outcomes(rows: list[dict[str, Any]], *, label: str) -> None:
    _print_section(label)
    if not rows:
        print("No trades")
        return

    pnl_values = [float(row.get("pnl_pct", 0.0) or 0.0) * 100.0 for row in rows]
    wins = [value for value in pnl_values if value > 0.0]
    losses = [value for value in pnl_values if value <= 0.0]

    expectancy = sum(pnl_values) / len(pnl_values)
    profit_factor = sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 1e-9 else None

    print(f"Trades: {len(rows)}")
    print(f"Win rate: {len(wins) / len(rows):.4f}")
    print(f"Avg win: {_fmt_pct(sum(wins) / len(wins) if wins else None)}")
    print(f"Avg loss: {_fmt_pct(sum(losses) / len(losses) if losses else None)}")
    print(f"Expectancy: {_fmt_pct(expectancy)}")
    print(f"Profit factor: {profit_factor:.4f}" if profit_factor is not None else "Profit factor: n/a")

    exit_counts = Counter(str(row.get("exit_reason", "")) for row in rows)
    print("Exit reasons:")
    for reason, count in exit_counts.most_common(8):
        print(f"  {reason or 'unknown'}: {count}")


def _load_outcome_reviews_since(cutoff: str) -> list[dict[str, Any]]:
    if not SYSTEM_RECORD_DB_PATH.exists():
        return []
    with sqlite3.connect(SYSTEM_RECORD_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT ts, symbol, lane, side, exit_reason, pnl_pct, hold_minutes, payload_json
            FROM outcome_reviews
            WHERE ts >= ?
            ORDER BY ts
            """,
            (cutoff,),
        ).fetchall()
    parsed: list[dict[str, Any]] = []
    for ts, symbol, lane, side, exit_reason, pnl_pct, hold_minutes, payload_json in rows:
        payload: dict[str, Any]
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
        payload.setdefault("ts", ts)
        payload.setdefault("symbol", symbol)
        payload.setdefault("lane", lane)
        payload.setdefault("side", side)
        payload.setdefault("exit_reason", exit_reason)
        payload.setdefault("pnl_pct", pnl_pct)
        payload.setdefault("hold_minutes", hold_minutes)
        parsed.append(payload)
    return parsed


def _summarize_quality_buckets(rows: list[dict[str, Any]]) -> None:
    _print_section("Entry Quality Split")
    if not rows:
        print("No trades")
        return

    buckets: dict[str, list[float]] = {
        "watch_or_low_quality": [],
        "other": [],
    }
    for row in rows:
        reasons = " ".join(str(item) for item in row.get("entry_reasons", []))
        key = "watch_or_low_quality" if ("WATCH" in reasons or "low_risk" in reasons) else "other"
        buckets[key].append(float(row.get("pnl_pct", 0.0) or 0.0) * 100.0)

    for name, values in buckets.items():
        if not values:
            print(f"{name}: no trades")
            continue
        print(
            f"{name}: count={len(values)} avg={_fmt_pct(sum(values) / len(values))} "
            f"wins={sum(1 for value in values if value > 0.0) / len(values):.4f}"
        )


def _summarize_broken_structure() -> None:
    _print_section("Broken Structure Holds")
    rows = _load_jsonl(DEBUG_LOG_PATH)
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("phase") != "hold_manager":
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol:
            latest_by_symbol[symbol] = row

    broken = [
        row
        for row in latest_by_symbol.values()
        if str(row.get("structure_state", "")).lower() == "broken"
    ]
    if not broken:
        print("None")
        return

    broken.sort(key=lambda row: float(row.get("hold_minutes", 0.0) or 0.0), reverse=True)
    for row in broken[:10]:
        entry_price = float(row.get("entry_price", 0.0) or 0.0)
        price = float(row.get("price", 0.0) or 0.0)
        pnl_pct = ((price / entry_price) - 1.0) * 100.0 if entry_price > 0.0 and price > 0.0 else None
        print(
            f"{row.get('symbol')}: hold={float(row.get('hold_minutes', 0.0) or 0.0):.1f}m "
            f"pnl={_fmt_pct(pnl_pct)} exit={row.get('live_exit_posture') or row.get('exit_posture')} "
            f"stop={row.get('trail_stop') or row.get('stop_loss')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GODMODE trading health metrics.")
    parser.add_argument(
        "--cutoff",
        default=_load_runtime_cutoff(),
        help="ISO timestamp cutoff for GODMODE trades. Defaults to configs/runtime_overrides.json updated_at.",
    )
    args = parser.parse_args()

    all_outcomes = _load_jsonl(OUTCOMES_PATH)
    _summarize_outcomes(all_outcomes, label="All Recorded Outcomes")

    cutoff = str(args.cutoff or "").strip()
    if cutoff:
        godmode_rows = _load_outcome_reviews_since(cutoff)
        _summarize_outcomes(godmode_rows, label=f"GODMODE Outcomes Since {cutoff}")
        _summarize_quality_buckets(godmode_rows)
    else:
        _print_section("GODMODE Outcomes")
        print("No cutoff found. Pass --cutoff 2026-03-27T01:28:01+00:00")

    _summarize_broken_structure()


if __name__ == "__main__":
    main()
