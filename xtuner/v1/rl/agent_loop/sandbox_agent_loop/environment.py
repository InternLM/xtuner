"""Episode-scoped environments, independent of runners and backend SDKs.

The caller owns ``provider.open(item)`` for the entire episode, including any
pauses between model requests. A trace segment never acquires an environment.
Providers must use a fresh namespace for each open, clean up on failure and
cancellation, and expose cleanup failures rather than turning them into reward.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from lagent.utils import create_object

from xtuner.v1.rl.utils.async_utils import CleanupError

from .sandbox import SandboxPool
from .schemas import AgentRolloutItem, StageRecord


@dataclass(frozen=True)
class SandboxInfo:
    image: str
    workspace: str
    environment_id: str | None = None
    url: str | None = None


class Environment(Protocol):
    """Named stage clients in one episode; no training or benchmark semantics.

    ``get`` waits for readiness and may allocate lazily. ``info`` validates the
    name without allocating; IDs may be absent until ``get`` succeeds. ``check``
    raises on known infrastructure failure, including background renewal errors.
    Clients implement the I/O operations used by the configured stages/hooks.
    """

    async def get(self, name: str, *, record: StageRecord | None = None) -> Any: ...

    def info(self, name: str) -> SandboxInfo: ...

    async def check(self) -> None: ...


class EnvironmentProvider(Protocol):
    """One provider per worker; one independently owned environment per open.

    ``item.environment`` is an opaque provider-specific request. Providers must
    not mutate it. ``close`` releases worker-level connections after all episode
    contexts exit; it must not silently abandon live environments.
    """

    def open(self, item: AgentRolloutItem) -> AbstractAsyncContextManager[Environment]: ...

    async def close(self) -> None: ...


class PoolEnvironment:
    """Adapt the existing lazy SandboxPool without changing its allocation rules."""

    def __init__(self, pool: SandboxPool):
        self.pool = pool
        self._clients: dict[str, Any] = {}

    async def get(self, name: str, *, record: StageRecord | None = None) -> Any:
        client = await self.pool.get(name, record=record)
        self._clients[name] = client
        return client

    def info(self, name: str) -> SandboxInfo:
        self.pool.validate_name(name)
        spec = self.pool.spec(name)
        return SandboxInfo(spec.image, spec.workspace_path, self.pool.env_id(name), self.pool.url(name))

    async def check(self) -> None:
        for name, client in self._clients.items():
            health = await asyncio.wait_for(client.health_check(), timeout=5)
            if not health.get("ok"):
                raise RuntimeError(f"sandbox {name!r} ({self.pool.env_id(name)}) is unhealthy")


class PoolEnvironmentProvider:
    """Existing SandboxPool backend; config is a template, never a shared pool."""

    def __init__(self, pool: dict[str, Any], cleanup_timeout_sec: float = 60.0):
        if not isinstance(pool, dict):
            raise TypeError("pool must be a config dict; sharing a live SandboxPool across episodes is unsafe")
        self._pool = deepcopy(pool)
        if cleanup_timeout_sec <= 0:
            raise ValueError("cleanup_timeout_sec must be positive")
        self._cleanup_timeout_sec = cleanup_timeout_sec
        self._pools: set[SandboxPool] = set()
        self._closed = False

    @asynccontextmanager
    async def open(self, item: AgentRolloutItem) -> AsyncIterator[Environment]:
        if self._closed:
            raise RuntimeError("environment provider is closed")
        config = deepcopy(self._pool)
        _deep_merge(config, item.environment)
        pool = create_object(config)
        self._pools.add(pool)
        try:
            yield PoolEnvironment(pool)
        finally:
            # Keep ownership until cleanup finishes even if a caller cancels
            # again while waiting for remote deletion.
            cleanup = asyncio.create_task(self._release(pool))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError

    async def close(self) -> None:
        self._closed = True
        results = await asyncio.gather(*(self._release(pool) for pool in list(self._pools)), return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise CleanupError("Environment cleanup failed", errors)

    async def _release(self, pool: SandboxPool) -> None:
        try:
            await asyncio.wait_for(pool.release_all(), self._cleanup_timeout_sec)
        except Exception as exc:
            raise CleanupError(
                f"Environment cleanup failed; pending sandbox IDs: {getattr(pool, 'pending_resource_ids', [])}", [exc]
            ) from exc
        self._pools.discard(pool)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = deepcopy(value)
