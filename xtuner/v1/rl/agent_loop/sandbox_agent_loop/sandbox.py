"""Sandbox runtime primitives: hook base, stage execution, entries, and I/O.

A :class:`SandboxStage` is a sequence of phases::

    pre-hooks  →  entries  →  post-hooks

Each hook is a callable with the uniform signature::

    async def hook(client, item, record) -> None

where ``item`` is the :class:`AgentRolloutItem` for this sample. Hooks read
the item input fields and write the current :class:`StageRecord`.
Concrete hook implementations live in ``hooks.py``.

Reading a stage config top-to-bottom tells you the full execution order.  A
stage's sandbox binding is just ``SandboxStage(sandbox="main")``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import re
import shlex
import tarfile
import time
import traceback
import uuid
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx
from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.lifecycle import CreateSandbox, SandboxLease, SandboxLifecycle
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    EntryOutcome,
    EntryRecord,
    RolloutError,
    SandboxSpec,
    StageRecord,
    StageResult,
    StageStatus,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trace import span
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trial_channel import get_creation_limiter
from xtuner.v1.utils import get_logger


def _expand_env_refs(value: Any) -> Any:
    """Recursively resolve ``${VAR}`` string leaves from ``os.environ``.

    Only a whole-string ``${VAR}`` is treated as a reference; any other string
    is returned as-is. A missing env var raises ``KeyError`` so a misconfigured
    run aborts loudly instead of POSTing an unresolved placeholder.
    """
    if isinstance(value, str):
        match = re.match(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$", value)
        if match is None:
            return value
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"sandbox create field references undefined env var ${{{name}}}")
        return os.environ[name]
    if isinstance(value, dict):
        return {k: _expand_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(v) for v in value]
    return value


# ─────────────────────────────────────────────────────────────────
# Hook base
# ─────────────────────────────────────────────────────────────────


class Hook:
    """A named step in a stage's pre or post pipeline.

    Subclasses implement ``__call__(client, item, record)``.  The item is
    the single rollout envelope; ``record`` is the current stage row
    (``item.infer`` / ``item.judgers[name]``). Name it
    in ``name`` so logs/errors identify the hook.
    """

    name: str = "hook"

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────
# Entry execution
# ─────────────────────────────────────────────────────────────────


@dataclass
class DiagnosticFile:
    path: str | None = None
    entry_file: str | None = None
    key: str | None = None
    optional: bool = True
    encoding: str = "utf-8"
    errors: str = "replace"

    def __post_init__(self) -> None:
        if (self.path is None) == (self.entry_file is None):
            raise ValueError("DiagnosticFile requires exactly one of path= or entry_file=")
        if self.entry_file is not None and self.entry_file not in {"rc", "stdout", "stderr"}:
            raise ValueError("DiagnosticFile.entry_file must be one of: rc, stdout, stderr")

    def resolve_path(self, entry: EntryRecord) -> str:
        if self.path is not None:
            return self.path
        entry_paths = {
            "rc": entry.rc_file,
            "stdout": entry.stdout_file,
            "stderr": entry.stderr_file,
        }
        path = entry_paths.get(self.entry_file or "")
        if not path:
            raise ValueError(f"entry file {self.entry_file!r} is not available for entry {entry.name!r}")
        return path


class EntryDiagnostics:
    """Best-effort host-side diagnostic download for failed entries."""

    def __init__(self, files: list[dict[str, Any] | DiagnosticFile]):
        self.files = [f if isinstance(f, DiagnosticFile) else DiagnosticFile(**f) for f in files]

    async def collect(
        self,
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
        entry: EntryRecord,
    ) -> None:
        for spec in self.files:
            path = spec.resolve_path(entry)
            key = spec.key or path
            try:
                blob = await client.download_file(path)
                item.artifacts[key] = blob.decode(spec.encoding, errors=spec.errors)
                entry.diagnostics[key] = path
            except Exception as exc:
                entry.diagnostic_errors.append(
                    {
                        "path": path,
                        "key": key,
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
                    }
                )
                if not spec.optional:
                    raise
                get_logger().debug(
                    f"entry diagnostic download {path} failed:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )


class EntryFailurePolicy:
    """Actions to run after an entry has produced a failed outcome.

    The normal entry path does not know which files are useful for a given
    task.  This policy makes the failure-side behavior explicit in config,
    usually by binding :class:`EntryDiagnostics` to the entry.
    """

    def __init__(
        self,
        *,
        diagnostics: EntryDiagnostics | dict[str, Any] | None = None,
        diagnostic_error_policy: str = "preserve_entry_error",
    ):
        if diagnostic_error_policy not in {"preserve_entry_error", "fail_entry"}:
            raise ValueError(
                "EntryFailurePolicy.diagnostic_error_policy must be 'preserve_entry_error' or 'fail_entry'"
            )
        self.diagnostics = create_object(diagnostics) if diagnostics is not None else None
        self.diagnostic_error_policy = diagnostic_error_policy

    async def handle(
        self,
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
        entry: EntryRecord,
        outcome: EntryOutcome,
    ) -> EntryOutcome:
        if self.diagnostics is None:
            return outcome
        try:
            await self.diagnostics.collect(client, item, record, entry)
        except Exception as exc:
            if self.diagnostic_error_policy == "fail_entry":
                outcome.result.error = (
                    f"{outcome.result.error or 'entry failed'}; diagnostics failed:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )
            else:
                get_logger().debug(
                    f"entry {entry.name} diagnostics failed after "
                    f"{outcome.reason or outcome.source}:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )
        return outcome


class EntryCapture:
    """Sandbox files written by a detached entry wrapper.

    Client pid polling observes completion from ``execute(detach=True)``.
    These files carry the shell result after that point. ``stdout`` and
    ``stderr`` are copied into the same :class:`StageResult` shape used by
    synchronous entries.
    """

    def __init__(self, *, root: str = "/tmp", prefix: str = "xt_entry"):
        self.root = root.rstrip("/") or "/tmp"
        self.prefix = prefix

    def bind(self, entry: EntryRecord) -> None:
        base = f"{self.root}/{self.prefix}_{entry.id}"
        entry.rc_file = f"{base}.rc"
        entry.stdout_file = f"{base}.stdout"
        entry.stderr_file = f"{base}.stderr"


class ShellEntry:
    """One synchronous observable shell entry."""

    def __init__(
        self,
        cmd: str,
        *,
        name: str = "entry",
        timeout: int = 600,
        env: dict[str, str] | None = None,
        failure: EntryFailurePolicy | dict[str, Any] | None = None,
    ):
        self.cmd = cmd
        self.name = name
        self.timeout = timeout
        self.env = dict(env or {})
        self.failure = create_object(failure) if failure is not None else None

    async def run(
        self,
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
    ) -> EntryOutcome:
        entry = self._new_record()
        record.entries.append(entry)
        entry.status = StageStatus.RUNNING
        entry.started_at = time.monotonic()
        uid_obs = str(item.uid) if item.uid is not None else ""
        with span(uid_obs, f"entry:{self.name}", entry_kind="ShellEntry"):
            try:
                outcome = await self._execute(client, self.env)
                if not outcome.ok and self.failure is not None:
                    outcome = await self.failure.handle(client, item, record, entry, outcome)
                self._finish_record(entry, outcome)
                return outcome
            except Exception as exc:
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
                outcome = EntryOutcome(
                    source=type(self).__name__,
                    reason="exception",
                    result=StageResult(return_code=None, stderr=error, error=error),
                )
                if self.failure is not None:
                    outcome = await self.failure.handle(client, item, record, entry, outcome)
                self._finish_record(entry, outcome, exc=exc)
                raise

    async def _execute(self, client: Any, env: dict[str, str]) -> EntryOutcome:
        exec_res = await exec_in(
            client,
            self.cmd,
            env=env,
            timeout_sec=self.timeout,
            raise_on_error=False,
        )
        rc = _result_code(exec_res)
        stderr = exec_res.get("stderr") or ""
        return _entry_completed_outcome(
            source="sync_exec",
            return_code=rc,
            stdout=exec_res.get("stdout") or "",
            stderr=stderr,
        )

    def _new_record(self) -> EntryRecord:
        suffix = uuid.uuid4().hex[:12]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name).strip("_") or "entry"
        entry_id = f"{safe_name}_{suffix}"
        entry = EntryRecord(
            id=entry_id,
            name=self.name,
            cmd=self.cmd,
            mode="sync",
        )
        return entry

    def _finish_record(self, entry: EntryRecord, outcome: EntryOutcome, *, exc: Exception | None = None) -> None:
        entry.finished_at = time.monotonic()
        entry.result = outcome.result
        entry.outcome = outcome
        entry.status = StageStatus.COMPLETED if outcome.ok else StageStatus.FAILED
        if not outcome.ok:
            entry.error = RolloutError(
                stage=entry.name,
                category=_outcome_error_category(outcome),
                type=type(exc).__name__
                if exc is not None
                else ("ScriptReturnCode" if outcome.result.return_code is not None else "EntryRuntimeError"),
                message=outcome.result.error
                or outcome.result.stderr
                or (
                    f"return_code={outcome.result.return_code}"
                    if outcome.result.return_code is not None
                    else outcome.reason or "entry failed"
                ),
                retryable=outcome.retryable,
            )


class DetachedShellEntry:
    """One detached shell entry observed through client pid polling.

    The sandbox wrapper writes stdout/stderr/rc files.  Completion is observed
    by polling the detached pid returned by ``execute(detach=True)``.  Once the
    pid is no longer running, stdout/stderr/rc are read with short retry
    windows and normalized into the same :class:`StageResult` shape as
    synchronous entries.
    """

    def __init__(
        self,
        cmd: str,
        *,
        name: str = "entry",
        timeout: int = 600,
        env: dict[str, str] | None = None,
        capture: EntryCapture | dict[str, Any],
        failure: EntryFailurePolicy | dict[str, Any] | None = None,
        handshake_timeout_sec: float = 60.0,
        poll_interval_sec: float = 60.0,
        pid_check_max_failures: int = 3,
        pid_check_retry_interval_sec: float = 10.0,
        rc_wait_timeout_sec: float = 10.0,
        rc_poll_interval_sec: float = 1.0,
        output_wait_timeout_sec: float = 5.0,
        output_poll_interval_sec: float = 1.0,
    ):
        self.cmd = cmd
        self.name = name
        self.timeout = timeout
        self.env = dict(env or {})
        self.failure = create_object(failure) if failure is not None else None
        self.capture = create_object(capture)
        self.handshake_timeout_sec = handshake_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.pid_check_max_failures = pid_check_max_failures
        self.pid_check_retry_interval_sec = pid_check_retry_interval_sec
        self.rc_wait_timeout_sec = rc_wait_timeout_sec
        self.rc_poll_interval_sec = rc_poll_interval_sec
        self.output_wait_timeout_sec = output_wait_timeout_sec
        self.output_poll_interval_sec = output_poll_interval_sec

    async def run(
        self,
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
    ) -> EntryOutcome:
        entry = self._new_record()
        self.capture.bind(entry)
        record.entries.append(entry)
        entry.status = StageStatus.RUNNING
        entry.started_at = time.monotonic()
        uid_obs = str(item.uid) if item.uid is not None else ""
        with span(uid_obs, f"entry:{self.name}", entry_kind="DetachedShellEntry"):
            try:
                outcome = await self._run_detached(client, item, entry, self.env)
                await self._fill_output_files(client, entry, outcome.result)
                if not outcome.ok and self.failure is not None:
                    outcome = await self.failure.handle(client, item, record, entry, outcome)
                self._finish_record(entry, outcome)
                return outcome
            except Exception as exc:
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
                outcome = EntryOutcome(
                    source=type(self).__name__,
                    reason="exception",
                    result=StageResult(return_code=None, stderr=error, error=error),
                )
                if self.failure is not None:
                    outcome = await self.failure.handle(client, item, record, entry, outcome)
                self._finish_record(entry, outcome, exc=exc)
                raise

    def _new_record(self) -> EntryRecord:
        suffix = uuid.uuid4().hex[:12]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name).strip("_") or "entry"
        entry_id = f"{safe_name}_{suffix}"
        return EntryRecord(
            id=entry_id,
            name=self.name,
            cmd=self.cmd,
            mode="detach",
        )

    async def _run_detached(
        self,
        client: Any,
        item: AgentRolloutItem,
        entry: EntryRecord,
        env: dict[str, str],
    ) -> EntryOutcome:
        assert entry.rc_file and entry.stdout_file and entry.stderr_file
        rc_file = shlex.quote(entry.rc_file)
        stdout_file = shlex.quote(entry.stdout_file)
        stderr_file = shlex.quote(entry.stderr_file)
        wrapped = (
            f"rm -f {rc_file} {stdout_file} {stderr_file}; "
            f"({self.cmd}) > {stdout_file} 2> {stderr_file}; "
            f"rc=$?; "
            f"printf '%s\\n' \"$rc\" > {rc_file}; "
            f'exit "$rc"'
        )
        launch_result = await exec_in(
            client,
            wrapped,
            env=env,
            timeout_sec=self.handshake_timeout_sec,
            raise_on_error=True,
            detach=True,
        )
        pid = int(launch_result["pid"])
        entry.pid = pid
        return await self._wait_with_pid_poll(client, item, entry, pid)

    async def _wait_with_pid_poll(
        self,
        client: Any,
        item: AgentRolloutItem,
        entry: EntryRecord,
        pid: int,
    ) -> EntryOutcome:
        is_pid_running = client.is_pid_running
        deadline = time.monotonic() + self.timeout
        poll = max(0.2, self.poll_interval_sec)
        polls = 0
        last_error: str | None = None
        pid_check_failures = 0
        while time.monotonic() <= deadline:
            polls += 1
            try:
                running = await is_pid_running(pid)
                last_error = None
                pid_check_failures = 0
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                pid_check_failures += 1
                if pid_check_failures >= self.pid_check_max_failures:
                    return EntryOutcome(
                        source="detach_exec",
                        reason="pid_monitor_error",
                        retryable=True,
                        details={
                            "pid": pid,
                            "polls": polls,
                            "pid_check_failures": pid_check_failures,
                            "pid_check_retry_interval_sec": self.pid_check_retry_interval_sec,
                            "error": last_error,
                        },
                        result=StageResult(
                            return_code=None,
                            stderr=f"[{item.id}] entry {entry.name} failed to query pid {pid}: {last_error}",
                            error="detached entry pid monitor failed",
                        ),
                    )
                retry_sleep = min(self.pid_check_retry_interval_sec, max(0.0, deadline - time.monotonic()))
                if retry_sleep > 0:
                    await asyncio.sleep(retry_sleep)
                continue

            if not running:
                rc = await self._wait_int_capture_file(
                    client,
                    entry.rc_file,
                    timeout_sec=self.rc_wait_timeout_sec,
                    poll_interval_sec=self.rc_poll_interval_sec,
                )
                if rc is None:
                    sandbox_alive = await _sandbox_alive(client)
                    return EntryOutcome(
                        source="detach_exec",
                        reason="rc_missing",
                        retryable=True,
                        details={
                            "pid": pid,
                            "polls": polls,
                            "rc_wait_timeout_sec": self.rc_wait_timeout_sec,
                            "rc_poll_interval_sec": self.rc_poll_interval_sec,
                            "sandbox_alive": sandbox_alive,
                        },
                        result=StageResult(
                            return_code=None,
                            stderr=f"[{item.id}] entry {entry.name} pid {pid} exited without rc file",
                            error="detached entry rc missing",
                        ),
                    )
                return _entry_completed_outcome(
                    source="detach_exec",
                    return_code=rc,
                    details={"pid": pid, "polls": polls},
                )

            sleep_sec = min(poll, max(0.0, deadline - time.monotonic()))
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

        rc = await self._read_int_capture_file(client, entry.rc_file)
        if rc is not None:
            return _entry_completed_outcome(
                source="detach_exec",
                return_code=rc,
                details={
                    "pid": pid,
                    "polls": polls,
                    "timeout": self.timeout,
                    "timeout_with_rc": True,
                    "last_pid_check_error": last_error,
                    "pid_check_failures": pid_check_failures,
                },
            )

        return EntryOutcome(
            source="detach_exec",
            reason="timeout",
            retryable=True,
            details={
                "pid": pid,
                "polls": polls,
                "timeout": self.timeout,
                "last_pid_check_error": last_error,
                "pid_check_failures": pid_check_failures,
                "pid_check_retry_interval_sec": self.pid_check_retry_interval_sec,
            },
            result=StageResult(
                return_code=None,
                stderr=f"[{item.id}] entry {entry.name} pid {pid} exceeded max runtime {self.timeout}s",
                error=f"entry {entry.name} timed out",
            ),
        )

    async def _read_int_capture_file(self, client: Any, path: str | None) -> int | None:
        if not path:
            return None
        blob = await self._read_capture_file(client, path)
        if blob is None:
            return None
        try:
            return int(blob.strip())
        except ValueError:
            return None

    async def _wait_int_capture_file(
        self,
        client: Any,
        path: str | None,
        *,
        timeout_sec: float,
        poll_interval_sec: float,
    ) -> int | None:
        deadline = time.monotonic() + timeout_sec
        poll = max(0.1, poll_interval_sec)
        while time.monotonic() <= deadline:
            value = await self._read_int_capture_file(client, path)
            if value is not None:
                return value
            sleep_sec = min(poll, max(0.0, deadline - time.monotonic()))
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
        return None

    async def _fill_output_files(self, client: Any, entry: EntryRecord, result: StageResult) -> None:
        tasks = []
        if entry.stdout_file:
            tasks.append(
                (
                    "stdout",
                    asyncio.create_task(self._wait_capture_file(client, entry.stdout_file)),
                )
            )
        if entry.stderr_file:
            tasks.append(
                (
                    "stderr",
                    asyncio.create_task(self._wait_capture_file(client, entry.stderr_file)),
                )
            )
        for name, task in tasks:
            value = await task
            if value is None:
                continue
            if name == "stdout":
                result.stdout = value
            else:
                result.stderr = value
                if result.return_code is not None and result.return_code > 0:
                    result.error = f"return_code={result.return_code}: {value[:400]}"

    async def _wait_capture_file(self, client: Any, path: str) -> str | None:
        deadline = time.monotonic() + self.output_wait_timeout_sec
        poll = max(0.1, self.output_poll_interval_sec)
        while time.monotonic() <= deadline:
            value = await self._read_capture_file(client, path)
            if value is not None:
                return value
            sleep_sec = min(poll, max(0.0, deadline - time.monotonic()))
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
        return None

    async def _read_capture_file(self, client: Any, path: str, timeout_sec: float = 5.0) -> str | None:
        try:
            blob = await asyncio.wait_for(client.download_file(path), timeout=timeout_sec)
        except Exception:
            return None
        return blob.decode(errors="replace")

    def _finish_record(self, entry: EntryRecord, outcome: EntryOutcome, *, exc: Exception | None = None) -> None:
        entry.finished_at = time.monotonic()
        entry.result = outcome.result
        entry.outcome = outcome
        entry.status = StageStatus.COMPLETED if outcome.ok else StageStatus.FAILED
        if not outcome.ok:
            entry.error = RolloutError(
                stage=entry.name,
                category=_outcome_error_category(outcome),
                type=type(exc).__name__
                if exc is not None
                else ("ScriptReturnCode" if outcome.result.return_code is not None else "EntryRuntimeError"),
                message=outcome.result.error
                or outcome.result.stderr
                or (
                    f"return_code={outcome.result.return_code}"
                    if outcome.result.return_code is not None
                    else outcome.reason or "entry failed"
                ),
                retryable=outcome.retryable,
            )


class SandboxStage:
    """Pre-hooks → entries → post-hooks.  Each field is visible.

    Not every stage needs every phase:
      - ``entries`` can be empty (pure setup/teardown stage).
      - ``pre`` / ``post`` default to empty.
      - Artifact collection is just another configured post-hook; stage has
        no separate ``pull`` contract.

    Entry execution is owned by :class:`ShellEntry` or
    :class:`DetachedShellEntry`; completion, liveness, and failed-entry
    diagnostics stay with the entry object that needs them.
    Post-hooks run only after the entry succeeds.
    """

    def __init__(
        self,
        *,
        sandbox: str = "main",
        pre: list[Hook] = [],
        entries: list[ShellEntry | DetachedShellEntry | dict[str, Any]] | None = None,
        post: list[Hook] = [],
        hook_stuck_warn_sec: float = 30.0,
    ):
        self.sandbox = sandbox
        self.pre = [create_object(hook) for hook in pre]
        entry_specs = list(entries or [])
        self.entries = [create_object(spec) for spec in entry_specs]
        for stage_entry in self.entries:
            if not isinstance(stage_entry, (ShellEntry, DetachedShellEntry)):
                raise TypeError(
                    "SandboxStage.entries must contain ShellEntry or DetachedShellEntry configs, "
                    f"got {type(stage_entry).__name__}"
                )
        self.post = [create_object(hook) for hook in post]
        self.hook_stuck_warn_sec = hook_stuck_warn_sec

    async def run(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> StageResult:
        record.status = StageStatus.RUNNING
        record.started_at = record.started_at or time.monotonic()
        stage_label = record.judger_name or "infer"

        try:
            await self._run_phase("pre", self.pre, client, item, record, stage_label)

            result = StageResult()
            for entry in self.entries:
                outcome = await entry.run(client, item, record)
                result = outcome.result
                self._apply_result(record, result, stage_label)
                if not outcome.ok:
                    return result

            await self._run_phase("post", self.post, client, item, record, stage_label)
            return result
        except Exception as exc:
            if record.status != StageStatus.FAILED:
                record.status = StageStatus.FAILED
            if record.error is None:
                latest_entry = record.entries[-1] if record.entries else None
                if latest_entry is not None:
                    message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
                    if latest_entry.cmd:
                        cmd = latest_entry.cmd
                        if len(cmd) > 500:
                            cmd = f"{cmd[:500]}...<truncated {len(cmd) - 500} chars>"
                        message = f"{message}; entry_id={latest_entry.id}; entry_cmd={cmd}"
                    record.error = RolloutError(
                        stage=latest_entry.name,
                        category="entry_exception",
                        type=type(exc).__name__,
                        message=message,
                    )
                else:
                    record.error = RolloutError(
                        stage=stage_label,
                        category="stage_exception",
                        type=type(exc).__name__,
                        message="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
                    )
            raise
        finally:
            record.finished_at = time.monotonic()

    async def _run_phase(
        self,
        phase: str,
        hooks: list[Hook],
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
        stage_label: str,
    ) -> None:
        hook_name = ""
        try:
            for hook in hooks:
                hook_name = getattr(hook, "name", hook.__class__.__name__)
                await self._run_hook(
                    hook,
                    client,
                    item,
                    record,
                    phase=phase,
                    stuck_warn_sec=self.hook_stuck_warn_sec,
                )
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=f"{stage_label}.{phase}.{hook_name}" if hook_name else stage_label,
                category=f"{phase}hook",
                type=type(exc).__name__,
                message="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
            )
            raise

    def _apply_result(self, record: StageRecord, result: StageResult, stage_label: str) -> None:
        record.entry_result = result
        if result.ok:
            record.status = StageStatus.COMPLETED
            record.error = None
            return
        record.status = StageStatus.FAILED
        latest_entry = record.entries[-1] if record.entries else None
        if latest_entry is not None and latest_entry.error is not None:
            record.error = latest_entry.error
        else:
            record.error = RolloutError(
                stage=latest_entry.name if latest_entry is not None else stage_label,
                category="entry_failed" if result.return_code is not None else "entry",
                type="ScriptReturnCode" if result.return_code is not None else "EntryRuntimeError",
                message=result.error
                or result.stderr
                or (f"return_code={result.return_code}" if result.return_code is not None else "entry failed"),
            )

    async def _run_hook(
        self,
        hook: Hook,
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
        *,
        phase: str,
        stuck_warn_sec: float,
    ) -> None:
        started = time.monotonic()
        hook_name = getattr(hook, "name", hook.__class__.__name__)
        task_id = item.id
        done = False

        async def warn_if_stuck() -> None:
            await asyncio.sleep(stuck_warn_sec)
            if not done:
                get_logger().warning(
                    f"[{task_id}] stage hook still running after {stuck_warn_sec:.1f}s: phase={phase} hook={hook_name}"
                )

        warn_task = asyncio.create_task(warn_if_stuck()) if stuck_warn_sec > 0 else None
        try:
            await hook(client, item, record)
        finally:
            done = True
            if warn_task is not None:
                warn_task.cancel()
            elapsed = time.monotonic() - started
            if elapsed > 1:
                get_logger().debug(f"[{task_id}] hook done phase={phase} hook={hook_name} took={elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────
# Low-level sandbox I/O
# ─────────────────────────────────────────────────────────────────


async def _sandbox_alive(client: Any, timeout_sec: float = 5.0) -> bool:
    try:
        result = await asyncio.wait_for(client.health_check(), timeout=timeout_sec)
    except Exception:
        return False
    return bool(result.get("ok"))


# ─────────────────────────────────────────────────────────────────
# Sandbox pool
# ─────────────────────────────────────────────────────────────────


class SandboxPool:
    """Per-run sandbox pool with transactional dependency groups.

    One ``SandboxPool`` instance is created at the start of ``Runner.run`` and
    released in the ``finally`` block. The Pool validates names, plans named
    dependency groups, limits create rates, checks health and runs provisioners.
    One Lifecycle owns the allocation transaction, safe retry, reverse-order
    rollback and resource cleanup, and retains each primary client/ID/URL.

    ``release_all`` releases a phase's resources while retaining the provider;
    ``aclose`` ends the Trial and transfers unresolved IDs to a fresh provider
    when a reconstructible provider configuration is available. Cleanup is
    bounded at three levels, with ``T = cleanup_timeout_sec`` (default 120
    seconds): resource release uses T, public ``release_all`` uses 1.1*T, and
    final ``aclose`` uses 1.2*T + 6. Each finalizer may add at most one second
    of cancellation grace. Async deadlines cannot preempt blocking synchronous
    filesystem I/O. Deletes and retries precede connection closure; results are
    available in ``cleanup``. ``close_provider``
    defaults to true for dict configurations and false for shared instances.
    Set it explicitly only for a provider dedicated to this pool. By default,
    ``verify_deletion`` requires ``supports_deletion_confirmation = True``;
    explicit ``verify_deletion=True`` opts into the normalized ``get`` contract. Persistent
    cleanup requires verification and an explicit ``cleanup_provider_key``;
    credentials/configuration are never written to the journal.

    Providers create and delete exactly one physical sandbox per call. They
    must not manage dependency groups or close returned clients.  An optional
    spec provisioner runs only after every member is healthy and must restrict
    itself to in-sandbox setup and application-level readiness.

    ``get(name, record=...)`` writes the failure to ``record.error`` with
    ``category="acquire"`` so the caller does not need its own try/except.
    """

    def __init__(
        self,
        provider: Any,
        specs: Mapping[str, SandboxSpec],
        *,
        max_attempts: int = 3,
        health_max_wait_sec: float = 600.0,
        health_poll_interval_sec: float = 2.0,
        creates_per_sec: float | None = 3.0,
        creates_burst: int = 1,
        create_rate_limit_key: str = "xtuner.sandbox_agent_loop.sandbox_acquire",
        cleanup_timeout_sec: float = 120.0,
        close_provider: bool | None = None,
        verify_deletion: bool | None = None,
        cleanup_dir: str | None = None,
        cleanup_provider_key: str | None = None,
        cleanup_provider: dict[str, Any] | None = None,
        cleanup_retry_timeout_sec: float = 60.0,
    ) -> None:
        self._lifecycle = SandboxLifecycle(
            provider,
            cleanup_timeout_sec=cleanup_timeout_sec,
            close_provider=close_provider,
            verify_deletion=verify_deletion,
            cleanup_dir=cleanup_dir,
            cleanup_provider_key=cleanup_provider_key,
            cleanup_provider=cleanup_provider,
            cleanup_retry_timeout_sec=cleanup_retry_timeout_sec,
        )
        self._specs: dict[str, SandboxSpec] = {name: create_object(spec) for name, spec in specs.items()}
        self._provisioners = {
            name: create_object(spec.provisioner) if spec.provisioner is not None else None
            for name, spec in self._specs.items()
        }
        self._max_attempts = max_attempts
        self._health_max_wait_sec = health_max_wait_sec
        self._health_poll_interval_sec = health_poll_interval_sec
        self._create_limiter = (
            get_creation_limiter(create_rate_limit_key, creates_per_sec, creates_burst)
            if creates_per_sec is not None
            else None
        )

    async def get(self, name: str, *, record: StageRecord | None = None) -> Any:
        self.validate_name(name)
        spec = self._specs[name]
        started = time.monotonic()
        try:
            client = await self._lifecycle.acquire_group(
                name,
                lambda create_member, attempt: self._acquire_ready(
                    name, spec, create_member, attempt, started, provisioner=self._provisioners[name], record=record
                ),
                max_attempts=self._max_attempts,
            )
            return client
        except Exception as exc:
            if record is not None:
                record.status = StageStatus.FAILED
                record.error = record.error or RolloutError(
                    stage=f"sandbox:{name}.acquire",
                    category="acquire",
                    type=type(exc).__name__,
                    message="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
                )
            raise

    @property
    def cleanup(self) -> dict[str, Any]:
        """Independent cleanup snapshot, retained after release and recovery handoff."""
        return self._lifecycle.cleanup

    @property
    def pending_resource_ids(self) -> frozenset[str]:
        """IDs still requiring release, without exposing mutable ownership sets."""
        return self._lifecycle.pending_resource_ids

    def peek(self, name: str) -> Any | None:
        """Observe an existing client without allocating or extending its lifetime."""
        self.validate_name(name)
        return self._lifecycle.peek(name)

    def task_scope(self, trial_id: str) -> AbstractAsyncContextManager[None]:
        """Observe run stop and finalize the trial on success, error or cancellation."""
        return self._lifecycle.task_scope(trial_id, self.aclose)

    def set_cleanup_context(self, trial_id: str) -> None:
        self._lifecycle.set_cleanup_context(trial_id)

    def env_id(self, name: str) -> str | None:
        return self._lifecycle.env_id(name)

    def url(self, name: str) -> str | None:
        return self._lifecycle.url(name)

    async def release_all(self) -> None:
        """Release phase resources; the management transport stays available."""
        await self._lifecycle.release_all()

    async def aclose(self) -> None:
        """Finish resource release and recovery handoff before closing transport."""
        await self._lifecycle.finish(self.release_all)

    def validate_name(self, name: str) -> None:
        if name not in self._specs:
            raise KeyError(f"unknown sandbox {name!r}; known: {sorted(self._specs)}")

    def spec(self, name: str) -> SandboxSpec:
        return self._specs[name]

    async def _acquire_ready(
        self,
        name: str,
        spec: SandboxSpec,
        create_member: CreateSandbox,
        attempt: int,
        started: float,
        *,
        provisioner: Any | None,
        record: StageRecord | None = None,
    ) -> None:
        if record is not None:
            record.metadata["sandbox_create_attempts"] = attempt
        group: list[SandboxLease] = []
        members = [*spec.dependencies.items(), ("primary", spec)]
        for member_name, member_spec in members:
            lease = await create_member(member_name, lambda: self._prepare_create(member_spec, record=record))
            group.append(lease)
            if not await self._wait_healthy(lease.client):
                raise RuntimeError(f"sandbox group {name!r} member {member_name!r} ({lease.env_id}) unhealthy")
        if provisioner is not None:
            dependencies = {lease.name: lease.client for lease in group[:-1]}
            await provisioner(group[-1].client, dependencies)
        if record is not None:
            record.metadata["sandbox_create_to_ready_time_s"] = time.monotonic() - started

    async def _prepare_create(self, spec: SandboxSpec, *, record: StageRecord | None = None) -> dict[str, Any]:
        """Validate and wait for rate limiting before admitting an allocation."""
        user = spec.user and _expand_env_refs(spec.user) or os.getenv("SANDBOX_PROVIDER_USER")
        if not user:
            raise ValueError("sandbox spec must have user or SANDBOX_PROVIDER_USER must be set")
        create_kwargs: dict[str, Any] = {"user": user}
        if spec.cluster_name:
            create_kwargs["cluster_name"] = spec.cluster_name
        if spec.env:
            create_kwargs["env"] = spec.env
        if spec.extra_params:
            create_kwargs["extra_params"] = _expand_env_refs(spec.extra_params)
        if spec.key:
            create_kwargs["key"] = spec.key
        if spec.env_vars:
            create_kwargs["env_vars"] = spec.env_vars
        if spec.resources:
            create_kwargs["resources"] = spec.resources
        if self._create_limiter is not None:
            t_limit = time.monotonic()
            await self._create_limiter.acquire()
            if record is not None:
                record.metadata["sandbox_acquire_rate_limit_wait_s"] = (
                    record.metadata.get("sandbox_acquire_rate_limit_wait_s", 0.0) + time.monotonic() - t_limit
                )
        return {"image_tag": spec.image, "ttl_seconds": spec.ttl_seconds, **create_kwargs}

    async def _wait_healthy(self, client: Any) -> bool:
        deadline = time.monotonic() + self._health_max_wait_sec
        while time.monotonic() < deadline:
            try:
                h = await client.health_check()
                if h.get("ok"):
                    return True
            except Exception as exc:
                get_logger().debug(
                    "health poll error:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )
            await asyncio.sleep(self._health_poll_interval_sec)
        return False


# ─────────────────────────────────────────────────────────────────
# Low-level sandbox I/O
# ─────────────────────────────────────────────────────────────────


_VAR_RE = re.compile(r"\$(\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_vars(text: str, env: dict[str, str]) -> str:
    """Substitute ``$VAR`` / ``${VAR}`` in ``text`` using ``env`` then process
    env.

    Needed because inline ``KEY=val cmd`` doesn't expand ``$KEY``
    within the same shell line.
    """

    def _sub(m: re.Match) -> str:
        name = m.group(2) or m.group(3)
        return env.get(name, os.environ.get(name, m.group(0)))

    return _VAR_RE.sub(_sub, text)


async def exec_in(
    client: Any,
    command: str,
    cwd: str = "/root",
    timeout_sec: int | float = 600,
    env: dict[str, str] | None = None,
    raise_on_error: bool = True,
    detach: bool = False,
) -> dict[str, Any]:
    """Execute a shell command inside the sandbox.

    Raises on failure by default.
    """
    if env:
        command = _expand_vars(command, env)
        # Use `export` so vars carry across chained commands (`bash A && bash B`).
        # Inline `VAR=val cmd1 && cmd2` scopes VAR to cmd1 only, which bites
        # when an entry wrapper starts child processes that still need the
        # stage env (RL_LLM_MODEL, TASK_WORKSPACE, etc.).
        exports = "; ".join(f'export {k}="{v}"' for k, v in env.items())
        command = f"{exports}; {command}"
    result = await client.execute(command, cwd, timeout_sec, detach)
    rc = _result_code(result)
    if raise_on_error and rc != 0:
        raise RuntimeError(
            f"command failed (return_code={rc}): {command}\n"
            f"stdout: {result.get('stdout', '')}\n"
            f"stderr: {result.get('stderr', '')}"
        )
    return result


def _retryable_upload_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def http_upload(client: Any, target_path: str, content_b64: str, *, max_attempts: int = 4) -> None:
    content = base64.b64decode(content_b64)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            await client.upload_bytes(target_path, content)
            return
        except Exception as exc:
            if not _retryable_upload_error(exc):
                raise
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = min(0.5 * (2 ** (attempt - 1)), 4.0) + random.uniform(0.0, 0.25)
            get_logger().warning(
                f"sandbox upload retry {attempt}/{max_attempts} for {target_path} after {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def upload_tar_and_extract(
    client: Any,
    file_map: dict[str, Path],
    extract_root: str,
) -> None:
    if not file_map:
        return
    blob = await asyncio.to_thread(_tar_bytes, file_map, extract_root)
    tmp = "/tmp/_bundle.tar.gz"
    await http_upload(client, tmp, base64.b64encode(blob).decode())
    await exec_in(
        client,
        f"mkdir -p {extract_root} && cd {extract_root} && tar xzf {tmp}",
    )


# ─────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────


def _tar_bytes(file_map: dict[str, Path], strip_root: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sb_path, host_path in file_map.items():
            if not sb_path.startswith(strip_root):
                raise ValueError(f"{sb_path} not under {strip_root}")
            arcname = sb_path[len(strip_root) :].lstrip("/")
            tar.add(host_path, arcname=arcname, recursive=False)
    return buf.getvalue()


def _result_code(exec_res: dict[str, Any]) -> int:
    rc = exec_res.get("return_code")
    if rc is None:
        rc = exec_res.get("exit_code", 0 if exec_res.get("ok", True) else 1)
    return int(rc)


def _entry_completed_outcome(
    *,
    source: str,
    return_code: int,
    stdout: str = "",
    stderr: str = "",
    details: dict[str, Any] | None = None,
) -> EntryOutcome:
    return EntryOutcome(
        source=source,
        reason="entry_completed" if return_code == 0 else "entry_failed",
        details=details or {},
        result=StageResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            error=None if return_code == 0 else f"return_code={return_code}: {stderr[:400]}",
        ),
    )


def _outcome_error_category(outcome: EntryOutcome) -> str:
    if outcome.reason in {"timeout", "pid_monitor_error", "rc_missing"}:
        return outcome.reason
    return outcome.reason or "entry"
