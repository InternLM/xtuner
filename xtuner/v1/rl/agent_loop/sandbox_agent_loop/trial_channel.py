"""Keep sandbox creation permits in the actor that owns isolated trials.

Only allocation permits cross this channel. Model requests keep using the
existing HTTP proxy, and recipes continue using the ordinary Pool API.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator


if TYPE_CHECKING:
    from lagent.utils.rate_limiter import FairAsyncTokenBucket


_MAX_MESSAGE = 4096
_CHANNEL: ContextVar[_TrialChannel | None] = ContextVar("sandbox_trial_channel", default=None)


def _local_limiter(key: str, rate: float, capacity: float | None) -> FairAsyncTokenBucket:
    from lagent.utils.rate_limiter import get_shared_async_token_bucket

    return get_shared_async_token_bucket(key, rate, capacity)


def get_creation_limiter(
    key: str, rate: float, capacity: float | None = None
) -> FairAsyncTokenBucket | _CreationLimiter:
    """Use the actor's existing shared limiter, including from a trial child.

    Args:
        key (str): Process-wide registry key, identical for sibling trials.
        rate (float): Creation permits per second.
        capacity (float | None): Burst capacity; use the rate when omitted.

    Returns:
        FairAsyncTokenBucket | _CreationLimiter: An asynchronous permit source.
    """
    channel = _CHANNEL.get()
    return (
        _CreationLimiter(channel, key, rate, capacity) if channel is not None else _local_limiter(key, rate, capacity)
    )


async def _messages(sock: socket.socket) -> AsyncIterator[dict[str, Any]]:
    pending = bytearray()
    loop = asyncio.get_running_loop()
    while True:
        chunk = await loop.sock_recv(sock, _MAX_MESSAGE)
        if not chunk:
            if pending:
                raise ConnectionError("Incomplete trial channel message")
            return
        pending.extend(chunk)
        while b"\n" in pending:
            end = pending.index(b"\n")
            if end > _MAX_MESSAGE:
                raise ValueError("Trial channel message too large")
            message = json.loads(pending[:end])
            del pending[: end + 1]
            if not isinstance(message, dict):
                raise ValueError("Trial channel message must be an object")
            yield message
        if len(pending) > _MAX_MESSAGE:
            raise ValueError("Trial channel message too large")


async def _send(sock: socket.socket, lock: asyncio.Lock, message: dict[str, Any]) -> None:
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(payload) > _MAX_MESSAGE:
        raise ValueError("Trial channel message too large")
    async with lock:
        await asyncio.get_running_loop().sock_sendall(sock, payload)


def _queue_send(
    sock: socket.socket, lock: asyncio.Lock, tasks: set[asyncio.Task], message: dict[str, Any]
) -> asyncio.Task:
    task = asyncio.create_task(_send(sock, lock, message))
    tasks.add(task)

    def finished(done: asyncio.Task) -> None:
        tasks.discard(done)
        if not done.cancelled() and done.exception() is not None:
            # Wake readers and sibling waiters when a framed write fails.
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    task.add_done_callback(finished)
    return task


async def serve_channel(sock: socket.socket) -> None:
    """Serve one child's permits; cancel and settle its waiters on disconnect.

    The caller transfers ownership of the socket to this operation. Independent
    child channels use the same existing process-wide limiter registry.

    Args:
        sock (socket.socket): Connected parent end of a local socketpair.
    """
    sock.setblocking(False)
    sending = asyncio.Lock()
    pending: dict[int, asyncio.Task] = {}
    writers: set[asyncio.Task] = set()

    async def grant(request: dict[str, Any]) -> None:
        identity = request["id"]
        try:
            key, rate, capacity = request["key"], request["rate"], request["capacity"]
            if not isinstance(key, str) or not key or len(key) > 256:
                raise ValueError("Invalid creation limiter key")
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid creation rate")
            if capacity is not None and (
                not isinstance(capacity, (int, float)) or not math.isfinite(capacity) or capacity < 1
            ):
                raise ValueError("Invalid creation burst")
            await _local_limiter(key, rate, capacity).acquire()
            response = {"id": identity, "ok": True}
        except Exception as error:
            response = {"id": identity, "error": type(error).__name__}
        # A canceled waiter cannot interrupt a response halfway through a frame.
        await asyncio.shield(_queue_send(sock, sending, writers, response))

    try:
        async for request in _messages(sock):
            identity = request.get("id")
            if type(identity) is not int or identity < 1:
                raise ValueError("Invalid trial channel request ID")
            operation = request.get("op")
            if operation == "cancel":
                task = pending.get(identity)
                if task is not None:
                    task.cancel()
            elif operation == "acquire" and identity not in pending:
                task = asyncio.create_task(grant(request))
                pending[identity] = task
            else:
                raise ValueError("Invalid trial channel operation")
            # Reap finished requests while retaining active tasks until EOF.
            for old_id, old_task in list(pending.items()):
                if old_task.done():
                    await asyncio.gather(old_task, return_exceptions=True)
                    del pending[old_id]
    finally:
        sock.close()
        for task in pending.values():
            task.cancel()
        for task in writers:
            task.cancel()
        await asyncio.gather(*pending.values(), *writers, return_exceptions=True)


class _TrialChannel:
    """Multiplex permit waiters without tying socket framing to their lifetime."""

    def __init__(self, sock: socket.socket):
        self.socket = sock
        self.sending = asyncio.Lock()
        self.pending: dict[int, asyncio.Future] = {}
        self.writers: set[asyncio.Task] = set()
        self.next_id = 0
        self.reader = asyncio.create_task(self._read())

    def _queue(self, message: dict[str, Any]) -> asyncio.Task:
        return _queue_send(self.socket, self.sending, self.writers, message)

    async def acquire(self, key: str, rate: float, capacity: float | None) -> None:
        if self.reader.done():
            raise ConnectionError("Trial permit channel is closed")
        self.next_id += 1
        identity = self.next_id
        waiter = asyncio.get_running_loop().create_future()
        self.pending[identity] = waiter
        try:
            await asyncio.shield(self._queue(dict(op="acquire", id=identity, key=key, rate=rate, capacity=capacity)))
            await waiter
        except asyncio.CancelledError:
            # A partially sent frame must finish before its cancellation frame.
            # Socket writes have their own lifetime and stay serialized.
            self._queue(dict(op="cancel", id=identity))
            raise
        finally:
            self.pending.pop(identity, None)
            if not waiter.done():
                waiter.cancel()
            elif not waiter.cancelled():
                waiter.exception()

    async def _read(self) -> None:
        error: Exception = ConnectionError("Trial permit channel disconnected")
        try:
            async for response in _messages(self.socket):
                waiter = self.pending.get(response.get("id"))
                if waiter is not None and not waiter.done():
                    if response.get("ok") is True:
                        waiter.set_result(None)
                    else:
                        waiter.set_exception(
                            RuntimeError("Trial creation permit failed: " + str(response.get("error")))
                        )
        except Exception as caught:
            error = caught
        finally:
            for waiter in self.pending.values():
                if not waiter.done():
                    waiter.set_exception(error)

    async def close(self) -> None:
        self.socket.close()
        self.reader.cancel()
        for task in self.writers:
            task.cancel()
        await asyncio.gather(self.reader, *self.writers, return_exceptions=True)


@dataclass(frozen=True)
class _CreationLimiter:
    channel: _TrialChannel
    key: str
    rate: float
    capacity: float | None

    async def acquire(self) -> None:
        await self.channel.acquire(self.key, self.rate, self.capacity)


@asynccontextmanager
async def child_channel(fd: int) -> AsyncIterator[None]:
    """Own an inherited connected descriptor for this child's whole trial.

    Args:
        fd (int): Child end of the inherited local socketpair.

    Yields:
        None: A context in which Pools request permits from their parent.
    """
    sock = socket.socket(fileno=fd)
    sock.setblocking(False)
    channel = _TrialChannel(sock)
    token = _CHANNEL.set(channel)
    try:
        yield
    finally:
        _CHANNEL.reset(token)
        await channel.close()
