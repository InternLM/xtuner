"""Complete-turn admission barrier for turn-resume rollout.

A request may finish only when the backend really ended and its trace committed. Forwarding an HTTP response
is not proof of settlement: an aborted or cancelled coroutine may leave the backend generating. The gate keeps
such a lease active and poisons every later drain, so a weight update can never race an unknown-exit request.

Admission-accounting reference: matrix72c/sandbox_rl before fd76fba,
src/sandbox_rl/agents/request_gate.py. This native condition/lease implementation
adds explicit backend/trace settlement; an unknown outcome poisons the barrier.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4


@dataclass(frozen=True)
class SettlementReceipt:
    """Evidence that one admitted request reached a terminal state.

    The receipt is assigned only after the success/failure terminal state has been handled, so its presence
    already means "terminal recorded"; only the backend outcome remains to be judged.

    Args:
        backend_ended (bool): The backend finished execution (or provably never started it).
        outcome (str): Human-readable terminal outcome used by logs and errors.
    """

    backend_ended: bool
    outcome: str


@dataclass
class RequestLease:
    """Identity and publication version of one admitted generation request.

    Args:
        request_id (str): Server-issued response id, also written to the trace and the HTTP response.
        session_id (str): Agent session that owns this request.
        epoch (int): Admission epoch the request was admitted under.
        model_step (int): Rollout model version serving this request.
        settlement (SettlementReceipt | None): Terminal evidence, set after the response is fully handled.
    """

    request_id: str
    session_id: str
    epoch: int
    model_step: int
    settlement: SettlementReceipt | None = None


class RequestGate:
    """One SessionServer's admission state machine.

    Args:
        wait_timeout (float): Maximum time an admitted request may wait for the gate to reopen.
        initially_open (bool): Whether the gate accepts requests before the first publication.
    """

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

    def snapshot(self) -> dict[str, Any]:
        return dict(
            open=self._open,
            epoch=self._epoch,
            model_step=self._model_step,
            active=list(self._active),
            failures=dict(self._failures),
        )

    @asynccontextmanager
    async def request(self, session_id: str) -> AsyncIterator[RequestLease]:
        async with self._condition:
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._open or bool(self._failures)),
                    timeout=self.wait_timeout,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Request admission timed out after {self.wait_timeout}s") from None
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
                receipt = lease.settlement
                if receipt is None or not receipt.backend_ended:
                    # An unknown exit keeps blocking later drains. Do not silently drop the lease.
                    self._open = False
                    self._failures[lease.request_id] = "backend_or_trace_unsettled"
                else:
                    self._active.pop(lease.request_id)
                self._condition.notify_all()

    async def seal(self, epoch: int) -> None:
        async with self._condition:
            if epoch < self._epoch:
                raise ValueError("Stale barrier epoch")
            if epoch == self._epoch and self._open:
                raise ValueError("A reopened epoch cannot be sealed again")
            self._open = False
            self._epoch = epoch

    async def drain(self, timeout: float) -> dict[str, Any]:
        async with self._condition:
            if self._open:
                raise RuntimeError("Seal admission before draining")
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: not self._active or bool(self._failures)),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                self._failures["barrier"] = "drain_timeout"
                self._condition.notify_all()
                raise TimeoutError(f"Request drain timed out after {timeout}s") from None
            if self._failures:
                raise RuntimeError(f"Unsettled requests prevent weight update: {self._failures}")
            return self.snapshot()

    async def install_version(self, epoch: int, model_step: int) -> None:
        async with self._condition:
            if self._open:
                raise RuntimeError("Version publication requires the gate to be sealed")
            if epoch != self._epoch:
                raise RuntimeError("Barrier epoch mismatch")
            if self._active or self._failures:
                raise RuntimeError("Barrier has unsettled or failed requests")
            if model_step < self._model_step:
                raise ValueError("Model versions cannot move backwards")
            if self._installed_epoch == epoch and model_step != self._model_step:
                raise ValueError("Conflicting version for installed epoch")
            self._model_step = model_step
            self._installed_epoch = epoch

    async def reopen(self, epoch: int) -> None:
        async with self._condition:
            if epoch != self._epoch:
                raise RuntimeError("Barrier epoch mismatch")
            if self._installed_epoch != epoch:
                raise RuntimeError("Cannot reopen an uninstalled barrier")
            if self._active or self._failures:
                raise RuntimeError("Cannot reopen a barrier with unsettled or failed requests")
            self._open = True
            self._condition.notify_all()
