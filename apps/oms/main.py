from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI()


OrderStatus = Literal[
    "accepted",
    "open",
    "partially_filled",
    "filled",
    "canceled",
    "rejected",
    "error",
]


class OrderRequest(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["limit"]
    price: float
    size_base: float
    time_in_force: Literal["FOK", "IOC", "GTC"] = "FOK"
    callback_url: str
    client_order_id: Optional[str] = None
    # optional metadata
    fee_usd: float = 0.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    sim_fill_fraction: float = 1.0
    # Paper-mode simulation hint: if true, order fills immediately
    sim_should_fill: bool = False


class OrderResponse(BaseModel):
    client_order_id: str
    order_id: str
    status: OrderStatus


@dataclass
class OrderUpdate:
    event: str
    client_order_id: str
    order_id: str
    status: OrderStatus
    filled_base: float
    avg_price: float
    fee_usd: float
    ts: str
    symbol: str
    side: str


async def post_callback(url: str, payload: dict) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(url, json=payload)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@app.post("/orders", response_model=OrderResponse)
async def create_order(req: OrderRequest) -> OrderResponse:
    if not req.callback_url:
        raise HTTPException(status_code=400, detail="callback_url required")

    client_order_id = req.client_order_id or f"u-{uuid.uuid4().hex[:8]}"
    order_id = f"oms-{uuid.uuid4().hex[:10]}"

    # immediately accept
    accepted = OrderUpdate(
        event="order_update",
        client_order_id=client_order_id,
        order_id=order_id,
        status="accepted",
        filled_base=0.0,
        avg_price=0.0,
        fee_usd=0.0,
        ts=now_iso(),
        symbol=req.symbol,
        side=req.side,
    )
    asyncio.create_task(post_callback(req.callback_url, asdict(accepted)))

    # Paper-mode FOK: fill immediately if sim_should_fill, else cancel
    async def finalize() -> None:
        await asyncio.sleep(0.1)
        if req.time_in_force == "FOK":
            if req.sim_should_fill:
                fill_base = req.size_base * min(max(req.sim_fill_fraction, 0.0), 1.0)
                if fill_base < req.size_base:
                    # FOK does not allow partials
                    canceled = OrderUpdate(
                        event="order_update",
                        client_order_id=client_order_id,
                        order_id=order_id,
                        status="canceled",
                        filled_base=0.0,
                        avg_price=0.0,
                        fee_usd=0.0,
                        ts=now_iso(),
                        symbol=req.symbol,
                        side=req.side,
                    )
                    await post_callback(req.callback_url, asdict(canceled))
                    return

                fill_price = req.price * (1 + (req.slippage_bps / 10000.0))
                fee_usd = (fill_price * fill_base) * (req.fee_bps / 10000.0)
                filled = OrderUpdate(
                    event="order_update",
                    client_order_id=client_order_id,
                    order_id=order_id,
                    status="filled",
                    filled_base=fill_base,
                    avg_price=fill_price,
                    fee_usd=fee_usd or req.fee_usd,
                    ts=now_iso(),
                    symbol=req.symbol,
                    side=req.side,
                )
                await post_callback(req.callback_url, asdict(filled))
            else:
                canceled = OrderUpdate(
                    event="order_update",
                    client_order_id=client_order_id,
                    order_id=order_id,
                    status="canceled",
                    filled_base=0.0,
                    avg_price=0.0,
                    fee_usd=req.fee_usd,
                    ts=now_iso(),
                    symbol=req.symbol,
                    side=req.side,
                )
                await post_callback(req.callback_url, asdict(canceled))
        elif req.time_in_force == "IOC":
            if req.sim_should_fill:
                fill_base = req.size_base * min(max(req.sim_fill_fraction, 0.0), 1.0)
                if fill_base <= 0:
                    canceled = OrderUpdate(
                        event="order_update",
                        client_order_id=client_order_id,
                        order_id=order_id,
                        status="canceled",
                        filled_base=0.0,
                        avg_price=0.0,
                        fee_usd=0.0,
                        ts=now_iso(),
                        symbol=req.symbol,
                        side=req.side,
                    )
                    await post_callback(req.callback_url, asdict(canceled))
                    return

                slip = req.slippage_bps / 10000.0
                fill_price = req.price * (1 + slip) if req.side == "buy" else req.price * (1 - slip)
                fee_usd = (fill_price * fill_base) * (req.fee_bps / 10000.0)

                status: OrderStatus = "filled"
                if fill_base < req.size_base:
                    status = "partially_filled"
                filled = OrderUpdate(
                    event="order_update",
                    client_order_id=client_order_id,
                    order_id=order_id,
                    status=status,
                    filled_base=fill_base,
                    avg_price=fill_price,
                    fee_usd=fee_usd or req.fee_usd,
                    ts=now_iso(),
                    symbol=req.symbol,
                    side=req.side,
                )
                await post_callback(req.callback_url, asdict(filled))

                if fill_base < req.size_base:
                    canceled = OrderUpdate(
                        event="order_update",
                        client_order_id=client_order_id,
                        order_id=order_id,
                        status="canceled",
                        filled_base=fill_base,
                        avg_price=fill_price,
                        fee_usd=fee_usd or req.fee_usd,
                        ts=now_iso(),
                        symbol=req.symbol,
                        side=req.side,
                    )
                    await post_callback(req.callback_url, asdict(canceled))
            else:
                canceled = OrderUpdate(
                    event="order_update",
                    client_order_id=client_order_id,
                    order_id=order_id,
                    status="canceled",
                    filled_base=0.0,
                    avg_price=0.0,
                    fee_usd=0.0,
                    ts=now_iso(),
                    symbol=req.symbol,
                    side=req.side,
                )
                await post_callback(req.callback_url, asdict(canceled))
        else:
            opened = OrderUpdate(
                event="order_update",
                client_order_id=client_order_id,
                order_id=order_id,
                status="open",
                filled_base=0.0,
                avg_price=0.0,
                fee_usd=0.0,
                ts=now_iso(),
                symbol=req.symbol,
                side=req.side,
            )
            await post_callback(req.callback_url, asdict(opened))

    asyncio.create_task(finalize())

    return OrderResponse(client_order_id=client_order_id, order_id=order_id, status="accepted")
