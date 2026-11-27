"""Command-ID based execution for providers with a public background protocol."""

from __future__ import annotations

import asyncio
import math
import shlex
import time
from typing import Any

from .sandbox import ShellEntry
from .schemas import AgentRolloutItem, EntryOutcome, StageRecord, StageResult, StageStatus


class BackgroundShellEntry(ShellEntry):
    """Run one command without shell PID or exit-code capture files.

    The client must provide async ``start_command(command, cwd, timeout_sec)``
    returning a command ID, ``get_command_status(id)`` returning a mapping with
    boolean ``running`` and optional integer ``return_code``,
    ``get_command_output(id)`` returning stdout/stderr, and
    ``interrupt_command(id)``. SDK-specific translation belongs to the provider.
    Existing synchronous and detached entries keep their current behavior.
    """

    def __init__(
        self,
        cmd: str,
        *,
        poll_interval_sec: float = 5,
        host_grace_sec: float = 60,
        cleanup_timeout_sec: float = 120,
        **kwargs: Any,
    ):
        super().__init__(cmd, **kwargs)
        for name, value, positive in (
            ("timeout", self.timeout, True),
            ("poll_interval_sec", poll_interval_sec, True),
            ("host_grace_sec", host_grace_sec, False),
            ("cleanup_timeout_sec", cleanup_timeout_sec, True),
        ):
            if not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        self.poll_interval_sec = poll_interval_sec
        self.host_grace_sec = host_grace_sec
        self.cleanup_timeout_sec = cleanup_timeout_sec

    async def run(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> EntryOutcome:
        entry = self._new_record()
        entry.mode = "background"
        entry.status = StageStatus.RUNNING
        entry.started_at = time.monotonic()
        record.entries.append(entry)
        details: dict[str, Any] = {"command_id": None}
        command = shlex.join(["env", *(f"{key}={value}" for key, value in self.env.items()), "bash", "-c", self.cmd])
        launch = None
        try:
            launch = asyncio.create_task(client.start_command(command, record.workspace or "/root", self.timeout))
            outcome = await asyncio.wait_for(
                self._observe(client, launch, details), self.timeout + self.host_grace_sec
            )
        except (Exception, asyncio.CancelledError) as exc:
            # Preserve an in-flight launch: cancelling its waiter must not lose
            # the ID needed to interrupt the original command. Never relaunch.
            cancelled = isinstance(exc, asyncio.CancelledError)
            cleanup = asyncio.create_task(
                asyncio.wait_for(self._interrupt(client, launch, details), self.cleanup_timeout_sec)
            )
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
                    continue
                except Exception:
                    break
            try:
                cleanup.result()
            except Exception as cleanup_error:
                details["cleanup_error"] = type(cleanup_error).__name__
            reason = (
                "cancelled" if cancelled else "timeout" if isinstance(exc, asyncio.TimeoutError) else "command_error"
            )
            outcome = EntryOutcome(
                source="background_command",
                reason=reason,
                retryable=reason == "timeout",
                details=dict(details),
                result=StageResult(return_code=None, error=f"{type(exc).__name__}: {exc}"),
            )
            self._finish_record(entry, outcome, exc=exc)
            if cancelled:
                raise asyncio.CancelledError from exc
        if not outcome.ok and self.failure is not None:
            outcome = await self.failure.handle(client, item, record, entry, outcome)
        self._finish_record(entry, outcome)
        return outcome

    async def _observe(self, client: Any, launch: asyncio.Task, details: dict[str, Any]) -> EntryOutcome:
        command_id = await asyncio.shield(launch)
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("Background command returned no command ID")
        details["command_id"] = command_id
        while True:
            status = await client.get_command_status(command_id)
            if type(status["running"]) is not bool:
                raise ValueError("Background command status must contain boolean running")
            if not status["running"]:
                rc = status.get("return_code")
                if rc is not None and type(rc) is not int:
                    raise ValueError("Background command return_code must be an integer or None")
                details["return_code"] = rc
                output = await client.get_command_output(command_id)
                return EntryOutcome(
                    source="background_command",
                    reason="exit_code_missing" if rc is None else "entry_completed" if rc == 0 else "entry_failed",
                    retryable=rc is None,
                    details=dict(details),
                    result=StageResult(
                        return_code=rc,
                        stdout=output.get("stdout") or "",
                        stderr=output.get("stderr") or "",
                        error="Background command returned no exit code" if rc is None else None,
                    ),
                )
            await asyncio.sleep(self.poll_interval_sec)

    async def _interrupt(self, client: Any, launch: asyncio.Task | None, details: dict[str, Any]) -> None:
        if launch is None:
            return
        command_id = details["command_id"] or await asyncio.shield(launch)
        if not isinstance(command_id, str) or not command_id:
            return
        details["command_id"] = command_id
        try:
            await client.interrupt_command(command_id)
            details["interrupt_sent"] = True
        except Exception as exc:
            # A command may already have reached its remote execution timeout.
            details["interrupt_error"] = type(exc).__name__
            status = await client.get_command_status(command_id)
            if status["running"] is not False:
                raise
        while True:
            status = await client.get_command_status(command_id)
            if status["running"] is False:
                details["command_terminal_after_cleanup"] = True
                return
            await asyncio.sleep(self.poll_interval_sec)
