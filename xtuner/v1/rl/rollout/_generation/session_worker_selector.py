from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from .rollout_worker_generator import RolloutWorkerGenerator


RolloutWorkerUrlSource = Literal["backend", "session"]


@dataclass
class RolloutWorkerHandle:
    """Runtime handles for one active rollout worker.

    This object intentionally groups the control-plane worker actor with the
    data-plane generation entries selected from that worker.
    """

    # Stable active-worker rank. Used for session affinity, logging, and
    # registering worker entries into HTTP routers.
    rank: int

    # Original RolloutWorker Ray actor. Controller control-plane operations
    # such as offload/onload/pause/shutdown are sent to this actor.
    worker_actor: Any

    # Raw backend server URL exposed by lmdeploy/vLLM/SGLang.
    backend_url: str

    # Per-active-worker generation actor used by local Python/Ray generation.
    generator_actor: RolloutWorkerGenerator

    # Optional SessionServer proxy URL wrapping backend_url. It is only used
    # when HTTP generation is configured to require session/cache/trace logic.
    session_server_url: str | None = None

    def require_session_server_url(self) -> str:
        if self.session_server_url is None:
            raise RuntimeError(f"Rollout worker {self.rank} does not have a SessionServer URL.")
        return self.session_server_url

    def get_generate_url(self, source: RolloutWorkerUrlSource) -> str:
        if source == "backend":
            return self.backend_url
        if source == "session":
            return self.require_session_server_url()
        raise ValueError(f"Unsupported rollout worker URL source: {source!r}")


class SessionWorkerSelector:
    def __init__(
        self,
        workers: list[RolloutWorkerHandle],
        *,
        max_sessions: int = 10000,
        max_idle_seconds: float | None = 3600.0,
    ) -> None:
        self._workers = {worker.rank: worker for worker in workers}
        self._rank_cycle = cycle(self._workers)
        self._max_sessions = max_sessions
        self._max_idle_seconds = max_idle_seconds
        self._sessions: OrderedDict[int, tuple[int, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def select(self, session_id: int) -> RolloutWorkerHandle | None:
        async with self._lock:
            self._evict_expired()
            if session_id in self._sessions:
                rank, _ = self._sessions.pop(session_id)
                worker = self._workers.get(rank)
                if worker is not None:
                    self._sessions[session_id] = (rank, self._now())
                    return worker

            worker = self._next_worker()
            if worker is None:
                return None
            self._sessions[session_id] = (worker.rank, self._now())
            self._evict_to_capacity()
            return worker

    def _next_worker(self) -> RolloutWorkerHandle | None:
        if not self._workers:
            return None
        return self._workers[next(self._rank_cycle)]

    def _evict_expired(self) -> None:
        if self._max_idle_seconds is None:
            return
        now = self._now()
        expired = []
        for session_id, (_, last_used_at) in self._sessions.items():
            if now - last_used_at > self._max_idle_seconds:
                expired.append(session_id)
            else:
                break
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _evict_to_capacity(self) -> None:
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def _now(self) -> float:
        return time.time()
