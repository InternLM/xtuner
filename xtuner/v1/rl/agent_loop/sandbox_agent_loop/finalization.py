"""Bounded async finalization that preserves cancellation and records failures."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Coroutine


class CleanupError(RuntimeError):
    """Typed, sanitized cleanup failure; never confused with a trial deadline."""

    def __init__(self, kind: str, phase: str) -> None:
        super().__init__(f"Sandbox {kind} during {phase}")
        self.kind = kind
        self.phase = phase


def _consume_result(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def finalize_bounded(
    operation: Coroutine[Any, Any, Any],
    *,
    timeout_seconds: float,
    diagnostic: dict[str, Any],
) -> Any:
    """Give cleanup its own deadline, including when its caller is cancelled.

    Repeated cancellation cannot skip resource cleanup. At deadline, cancellation
    of the child gets a bounded grace period; a child ignoring cancellation is
    explicitly reported pending, with its eventual exception consumed.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        operation.close()
        raise ValueError("cleanup timeout must be finite and positive")
    started = time.monotonic()
    deadline = started + timeout_seconds
    task = asyncio.create_task(operation)
    cancellation = None
    diagnostic.update(
        timeout_seconds=timeout_seconds,
        started_at=time.time(),
        ok=False,
        error_kind=None,
        cancelled=False,
        tasks_settled=False,
    )
    try:
        while not task.done() and time.monotonic() < deadline:
            try:
                await asyncio.wait({task}, timeout=max(0.0, deadline - time.monotonic()))
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                diagnostic["cancelled"] = True

        if not task.done():
            diagnostic.update(error_kind="cleanup_timeout", phase_at_error=diagnostic.get("phase", "cleanup"))
            task.cancel()
            drain_deadline = time.monotonic() + min(1.0, timeout_seconds / 10)
            while not task.done() and time.monotonic() < drain_deadline:
                try:
                    await asyncio.wait({task}, timeout=max(0.0, drain_deadline - time.monotonic()))
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                    diagnostic["cancelled"] = True
            diagnostic["tasks_settled"] = task.done()
            if task.done():
                _consume_result(task)
            else:
                task.add_done_callback(_consume_result)
            if cancellation is not None:
                raise cancellation
            raise CleanupError("cleanup_timeout", diagnostic["phase_at_error"])

        diagnostic["tasks_settled"] = True
        try:
            result = task.result()
        except asyncio.CancelledError:
            diagnostic.update(error_kind="cleanup_cancelled", phase_at_error=diagnostic.get("phase", "cleanup"))
            if cancellation is not None:
                raise cancellation
            raise CleanupError("cleanup_cancelled", diagnostic["phase_at_error"]) from None
        except Exception as exc:
            diagnostic.update(
                error_kind=exc.kind if isinstance(exc, CleanupError) else "cleanup_failed",
                exception_type=type(exc).__name__,
                phase_at_error=diagnostic.get("failure_phase", diagnostic.get("phase", "cleanup")),
            )
            if cancellation is not None:
                raise cancellation
            raise CleanupError(diagnostic["error_kind"], diagnostic["phase_at_error"]) from None
        diagnostic["ok"] = True
        if cancellation is not None:
            raise cancellation
        return result
    finally:
        diagnostic.update(elapsed_seconds=time.monotonic() - started, finished_at=time.time())
