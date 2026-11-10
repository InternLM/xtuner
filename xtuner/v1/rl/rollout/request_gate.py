"""Complete-turn admission barrier; forwarding alone certifies settlement.

Admission-accounting reference: matrix72c/sandbox_rl before fd76fba,
src/sandbox_rl/agents/request_gate.py. This native condition/lease implementation
adds explicit backend/trace settlement; an unknown outcome poisons the barrier.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class SettlementReceipt:
    backend_ended: bool
    terminal_recorded: bool
    outcome: str

    @property
    def safe_for_weight_update(self) -> bool:
        return self.backend_ended and self.terminal_recorded


@dataclass
class RequestLease:
    request_id: str
    session_id: str
    epoch: int
    model_step: int
    settlement: SettlementReceipt | None = None


class RequestGate:
    def __init__(self, *, wait_timeout: float, initially_open: bool = True):
        if wait_timeout <= 0:
            raise ValueError("wait_timeout must be positive")
        self.wait_timeout = wait_timeout
        self._condition = asyncio.Condition()
        self._open = initially_open
        self._epoch = 0
        self._installed_epoch = 0
        self._model_step = 0
        self._active: dict[str, RequestLease] = {}
        self._failures: dict[str, str] = {}

    def snapshot(self) -> dict:
        return dict(
            open=self._open,
            epoch=self._epoch,
            model_step=self._model_step,
            active=list(self._active),
            failures=dict(self._failures),
        )

    @asynccontextmanager
    async def request(self, session_id: str):
        async with self._condition:
            async with asyncio.timeout(self.wait_timeout):
                await self._condition.wait_for(lambda: self._open or bool(self._failures))
            if self._failures:
                raise RuntimeError(f"Request barrier failed: {self._failures}")
            if any(lease.session_id == session_id for lease in self._active.values()):
                raise RuntimeError("Concurrent generation in the same session is unsupported")
            lease = RequestLease("chatcmpl-" + uuid4().hex, session_id, self._epoch, self._model_step)
            self._active[lease.request_id] = lease
        try:
            yield lease
        finally:
            async with self._condition:
                if lease.settlement is None or not lease.settlement.safe_for_weight_update:
                    self._open = False
                    self._failures[lease.request_id] = "backend_or_trace_unsettled"
                else:
                    self._active.pop(lease.request_id)
                self._condition.notify_all()

    async def seal(self, epoch: int):
        async with self._condition:
            if epoch < self._epoch:
                raise ValueError("Stale barrier epoch")
            if epoch == self._epoch and self._open:
                raise ValueError("A reopened epoch cannot be sealed again")
            self._open = False
            self._epoch = epoch

    async def drain(self, timeout: float):
        async with self._condition:
            if self._open:
                raise RuntimeError("Seal admission before draining")
            try:
                async with asyncio.timeout(timeout):
                    await self._condition.wait_for(lambda: not self._active or bool(self._failures))
            except TimeoutError:
                self._failures["barrier"] = "drain_timeout"
                self._condition.notify_all()
                raise
            if self._failures:
                raise RuntimeError(f"Unsettled requests prevent weight update: {self._failures}")
            return self.snapshot()

    async def install_version(self, epoch: int, model_step: int):
        async with self._condition:
            if self._open or epoch != self._epoch or self._active or self._failures:
                raise RuntimeError("Version publication requires the drained current barrier")
            if model_step < self._model_step:
                raise ValueError("Model versions cannot move backwards")
            if self._installed_epoch == epoch and model_step != self._model_step:
                raise ValueError("Conflicting version for installed epoch")
            self._model_step = model_step
            self._installed_epoch = epoch

    async def reopen(self, epoch: int):
        async with self._condition:
            if epoch != self._epoch or self._installed_epoch != epoch or self._active or self._failures:
                raise RuntimeError("Cannot reopen an uninstalled or unsafe barrier")
            self._open = True
            self._condition.notify_all()
