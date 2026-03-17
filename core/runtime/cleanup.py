from __future__ import annotations

import asyncio
from typing import Iterable


async def cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def cleanup_tasks(tasks: Iterable[asyncio.Task | None]) -> None:
    for task in tasks:
        await cancel_task(task)
