from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POSITIONS_PATH = Path("logs/positions.json")


def load_positions() -> dict[str, Any]:
    if not POSITIONS_PATH.exists():
        return {}
    try:
        return json.loads(POSITIONS_PATH.read_text())
    except Exception:
        return {}


def save_positions(data: dict[str, Any]) -> None:
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_PATH.write_text(json.dumps(data, indent=2))


def apply_fill(symbol: str, side: str, size_base: float, price: float, fee_usd: float) -> None:
    data = load_positions()
    pos = data.get(
        symbol,
        {"size_base": 0.0, "avg_price": 0.0, "realized_pnl_usd": 0.0, "fees_usd": 0.0},
    )

    size = float(pos["size_base"])
    avg = float(pos["avg_price"])
    realized = float(pos["realized_pnl_usd"])
    fees = float(pos.get("fees_usd", 0.0))

    if side == "buy":
        new_size = size + size_base
        new_avg = ((size * avg) + (size_base * price)) / new_size if new_size > 0 else 0.0
        pos["size_base"] = new_size
        pos["avg_price"] = new_avg
        pos["realized_pnl_usd"] = realized - fee_usd
        pos["fees_usd"] = fees + fee_usd
    else:
        # sell reduces position
        new_size = size - size_base
        pnl = (price - avg) * size_base
        pos["size_base"] = new_size
        pos["avg_price"] = avg if new_size > 0 else 0.0
        pos["realized_pnl_usd"] = realized + pnl - fee_usd
        pos["fees_usd"] = fees + fee_usd

    data[symbol] = pos
    save_positions(data)


def snapshot_positions() -> None:
    data = load_positions()
    p = Path("logs/positions.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": __import__("datetime").datetime.utcnow().isoformat() + "Z", "positions": data}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
