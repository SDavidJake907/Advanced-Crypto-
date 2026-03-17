from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request

from core.state.open_orders import update_status
from core.state.positions import apply_fill, snapshot_positions


app = FastAPI()


@app.post("/callbacks/orders")
async def order_callback(req: Request) -> dict:
    payload = await req.json()
    client_order_id = payload.get("client_order_id")
    status = payload.get("status")
    if client_order_id and status:
        update_status(client_order_id, status)

    if status in {"filled", "partially_filled"}:
        symbol = payload.get("pair") or payload.get("symbol")
        side = payload.get("side")
        size_base = float(payload.get("filled_base", 0.0))
        avg_price = float(payload.get("avg_price", 0.0))
        fee_usd = float(payload.get("fee_usd", 0.0))
        if symbol and side and size_base > 0:
            apply_fill(symbol, side, size_base, avg_price, fee_usd)
            snapshot_positions()
    Path("logs").mkdir(parents=True, exist_ok=True)
    with Path("logs/order_callbacks.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return {"ok": True}
