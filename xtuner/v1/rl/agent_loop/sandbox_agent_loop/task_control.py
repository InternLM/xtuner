"""Cooperative run-stop propagation; never terminate a shared actor process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from .run_control import is_stopping


@asynccontextmanager
async def watch_task(directory: str | None) -> AsyncIterator[asyncio.Event]:
    """Cancel this Runner on run stop, allowing its normal finalization to drain.

    Args:
        directory (str | None): Exact run directory, or disabled.

    Yields:
        asyncio.Event: Set when cancellation was requested by run shutdown.
    """
    stopped = asyncio.Event()
    if not directory:
        yield stopped
        return
    owner = asyncio.current_task()
    assert owner is not None

    async def observe() -> None:
        while True:
            try:
                stopping = await asyncio.to_thread(is_stopping, Path(directory))
            except OSError:
                stopping = False
            if stopping:
                stopped.set()
                owner.cancel("sandbox_run_stopping")
                return
            await asyncio.sleep(0.5)

    observer = asyncio.create_task(observe())
    try:
        yield stopped
    finally:
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
