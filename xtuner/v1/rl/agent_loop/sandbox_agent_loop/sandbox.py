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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx
from lagent.utils import create_object

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
from xtuner.v1.utils import get_logger


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
        if self.entry_file is not None and self.entry_file not in {"pid", "rc", "stdout", "stderr"}:
            raise ValueError("DiagnosticFile.entry_file must be one of: pid, rc, stdout, stderr")

    def resolve_path(self, entry: EntryRecord) -> str:
        if self.path is not None:
            return self.path
        entry_paths = {
            "pid": entry.pid_file,
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


class EntryMonitorProbe:
    """One configured monitor probe for a detached entry."""

    def __init__(self, *, interval_sec: float):
        self.interval_sec = interval_sec

    async def probe(
        self,
        client: Any,
        item: AgentRolloutItem,
        entry: EntryRecord,
        state: dict[str, Any],
    ) -> EntryOutcome | None:
        raise NotImplementedError

    async def _download_file(self, client: Any, path: str, timeout_sec: float = 5.0) -> bytes | None:
        if not path:
            return None
        try:
            return await asyncio.wait_for(client.download_file(path), timeout=timeout_sec)
        except Exception:
            return None

    async def _read_int_file(self, client: Any, path: str) -> int | None:
        blob = await self._download_file(client, path)
        if blob is None:
            return None
        try:
            return int(blob.decode(errors="replace").strip())
        except ValueError:
            return None

    async def _pid_alive(self, client: Any, pid: int, timeout_sec: float) -> bool | None:
        try:
            res = await exec_in(
                client,
                f"kill -0 {pid} 2>/dev/null && echo Y || echo N",
                raise_on_error=False,
                timeout_sec=timeout_sec,
            )
        except Exception:
            return None
        out = (res.get("stdout") or "").strip()
        if "Y" in out:
            return True
        if "N" in out:
            return False
        return None


class ReturnCodeFileCompletion(EntryMonitorProbe):
    """Finish condition: the detached wrapper has written its rc file."""

    def __init__(self, *, interval_sec: float = 2.0):
        super().__init__(interval_sec=interval_sec)

    async def probe(
        self,
        client: Any,
        item: AgentRolloutItem,
        entry: EntryRecord,
        state: dict[str, Any],
    ) -> EntryOutcome | None:
        rc = await self._read_int_file(client, entry.rc_file or "")
        if rc is None:
            return None
        get_logger().debug("[%s] entry %s finished rc=%s", item.id, entry.name, rc)
        return EntryOutcome(
            source=type(self).__name__,
            reason="entry_completed" if rc == 0 else "entry_failed",
            result=StageResult(return_code=rc, error=None if rc == 0 else f"return_code={rc}"),
        )


class SandboxHealthCheck(EntryMonitorProbe):
    """Failure condition: the sandbox client health endpoint is unreachable."""

    def __init__(self, *, interval_sec: float = 10.0, probe_timeout_sec: float = 10.0, fail_after: int = 3):
        super().__init__(interval_sec=interval_sec)
        self.probe_timeout_sec = probe_timeout_sec
        self.fail_after = fail_after

    async def probe(
        self,
        client: Any,
        item: AgentRolloutItem,
        entry: EntryRecord,
        state: dict[str, Any],
    ) -> EntryOutcome | None:
        if await _sandbox_alive(client, self.probe_timeout_sec):
            state["failures"] = 0
            return None
        failures = int(state.get("failures") or 0) + 1
        state["failures"] = failures
        if failures < self.fail_after:
            return None
        return EntryOutcome(
            source=type(self).__name__,
            reason="sandbox_unreachable",
            retryable=True,
            details={"failures": failures, "fail_after": self.fail_after},
            result=StageResult(
                return_code=None,
                stderr=f"[{item.id}] sandbox unreachable while waiting for entry {entry.name}",
                error="sandbox unreachable",
            ),
        )


class EntryProcessHealthCheck(EntryMonitorProbe):
    """Failure condition: the detached wrapper pid disappears before rc is written."""

    def __init__(self, *, interval_sec: float = 10.0, probe_timeout_sec: float = 10.0, fail_after: int = 2):
        super().__init__(interval_sec=interval_sec)
        self.probe_timeout_sec = probe_timeout_sec
        self.fail_after = fail_after

    async def probe(
        self,
        client: Any,
        item: AgentRolloutItem,
        entry: EntryRecord,
        state: dict[str, Any],
    ) -> EntryOutcome | None:
        pid = await self._read_int_file(client, entry.pid_file or "")
        if pid is None:
            missing_pid_file = int(state.get("missing_pid_file") or 0) + 1
            state["missing_pid_file"] = missing_pid_file
            if missing_pid_file < self.fail_after:
                return None
            rc = await self._read_int_file(client, entry.rc_file or "")
            if rc is not None:
                return EntryOutcome(
                    source=type(self).__name__,
                    reason="entry_completed" if rc == 0 else "entry_failed",
                    result=StageResult(return_code=rc, error=None if rc == 0 else f"return_code={rc}"),
                )
            return EntryOutcome(
                source=type(self).__name__,
                reason="pid_file_missing",
                retryable=True,
                details={"missing_pid_file": missing_pid_file, "fail_after": self.fail_after},
                result=StageResult(
                    return_code=None,
                    stderr=f"[{item.id}] entry {entry.name} pid file was not written",
                    error="entry pid file missing",
                ),
            )
        state["missing_pid_file"] = 0
        alive = await self._pid_alive(client, pid, self.probe_timeout_sec)
        if alive is True:
            state["missing"] = 0
            return None
        if alive is not False:
            return None
        missing = int(state.get("missing") or 0) + 1
        state["missing"] = missing
        if missing < self.fail_after:
            return None
        rc = await self._read_int_file(client, entry.rc_file or "")
        if rc is not None:
            return EntryOutcome(
                source=type(self).__name__,
                reason="entry_completed" if rc == 0 else "entry_failed",
                result=StageResult(return_code=rc, error=None if rc == 0 else f"return_code={rc}"),
            )
        return EntryOutcome(
            source=type(self).__name__,
            reason="pid_lost",
            retryable=True,
            details={"pid": pid, "missing": missing, "fail_after": self.fail_after},
            result=StageResult(
                return_code=None,
                stderr=f"[{item.id}] entry {entry.name} pid {pid} gone without rc file",
                error="entry pid lost",
            ),
        )


class EntryMonitor:
    """Run configured completion/health probes until one returns a result."""

    def __init__(
        self,
        *,
        timeout: int,
        probes: list[EntryMonitorProbe | dict[str, Any]],
    ):
        self.timeout = timeout
        self.probes = [create_object(probe) for probe in probes]
        if not self.probes:
            raise ValueError("EntryMonitor requires at least one probe")

    async def wait(self, client: Any, item: AgentRolloutItem, entry: EntryRecord) -> EntryOutcome:
        start = time.monotonic()
        next_probe = [start for _ in self.probes]
        states: list[dict] = [{} for _ in self.probes]

        while True:
            now = time.monotonic()
            if now - start > self.timeout:
                return EntryOutcome(
                    source=type(self).__name__,
                    reason="timeout",
                    retryable=True,
                    details={"timeout": self.timeout},
                    result=StageResult(
                        return_code=None,
                        stderr=f"[{item.id}] entry {entry.name} exceeded max runtime {self.timeout}s",
                        error=f"entry {entry.name} timed out",
                    ),
                )

            for idx, probe in enumerate(self.probes):
                if now < next_probe[idx]:
                    continue
                result = await probe.probe(client, item, entry, states[idx])
                if result is not None:
                    return result
                next_probe[idx] = now + probe.interval_sec

            sleep_until = min([*next_probe, start + self.timeout])
            await asyncio.sleep(max(0.2, min(1.0, sleep_until - time.monotonic())))


class EntryCapture:
    """Sandbox files written by a detached entry wrapper.

    These files are the contract consumed by monitor probes and diagnostics:
    ``pid`` and ``rc`` are used to observe completion, while ``stdout`` and
    ``stderr`` capture the command output.
    """

    def __init__(self, *, root: str = "/tmp", prefix: str = "xt_entry"):
        self.root = root.rstrip("/") or "/tmp"
        self.prefix = prefix

    def bind(self, entry: EntryRecord) -> None:
        base = f"{self.root}/{self.prefix}_{entry.id}"
        entry.pid_file = f"{base}.pid"
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
        return EntryOutcome(
            source="sync_exec",
            reason="entry_completed" if rc == 0 else "entry_failed",
            result=StageResult(
                stdout=exec_res.get("stdout") or "",
                stderr=stderr,
                return_code=rc,
                error=None if rc == 0 else f"return_code={rc}: {stderr[:400]}",
            ),
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
    """One detached shell entry using the capture/rc completion protocol."""

    def __init__(
        self,
        cmd: str,
        *,
        name: str = "entry",
        timeout: int = 600,
        env: dict[str, str] | None = None,
        capture: EntryCapture | dict[str, Any],
        monitor: EntryMonitor | dict[str, Any],
        failure: EntryFailurePolicy | dict[str, Any] | None = None,
        handshake_timeout_sec: float = 60.0,
    ):
        self.cmd = cmd
        self.name = name
        self.timeout = timeout
        self.env = dict(env or {})
        self.failure = create_object(failure) if failure is not None else None
        self.capture = create_object(capture)
        self.monitor = create_object(monitor)
        self.handshake_timeout_sec = handshake_timeout_sec

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
        assert entry.pid_file and entry.rc_file and entry.stdout_file and entry.stderr_file
        pid_file = shlex.quote(entry.pid_file)
        rc_file = shlex.quote(entry.rc_file)
        stdout_file = shlex.quote(entry.stdout_file)
        stderr_file = shlex.quote(entry.stderr_file)
        wrapped = (
            f"rm -f {pid_file} {rc_file} {stdout_file} {stderr_file}; "
            f"echo $$ > {pid_file}; "
            f"({self.cmd}) > {stdout_file} 2> {stderr_file}; "
            f"echo $? > {rc_file}"
        )
        await exec_in(
            client,
            wrapped,
            env=env,
            timeout_sec=self.handshake_timeout_sec,
            raise_on_error=True,
            detach=True,
        )
        return await self.monitor.wait(client, item, entry)

    async def _fill_output_files(self, client: Any, entry: EntryRecord, result: StageResult) -> None:
        if entry.stdout_file:
            stdout = await self._read_capture_file(client, entry.stdout_file)
            if stdout is not None:
                result.stdout = stdout
        if entry.stderr_file:
            stderr = await self._read_capture_file(client, entry.stderr_file)
            if stderr is not None:
                result.stderr = stderr
                if result.return_code is not None and result.return_code > 0:
                    result.error = f"return_code={result.return_code}: {stderr[:400]}"

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
            if record.status != StageStatus.FAILED:
                record.status = StageStatus.COMPLETED
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
        hook_record: dict[str, Any] = {
            "phase": phase,
            "hook": hook_name,
            "status": StageStatus.RUNNING.value,
            "started_at": started,
        }
        record.metadata.setdefault("hooks", []).append(hook_record)

        async def warn_if_stuck() -> None:
            await asyncio.sleep(stuck_warn_sec)
            if not done:
                get_logger().warning(
                    f"[{task_id}] stage hook still running after {stuck_warn_sec:.1f}s: phase={phase} hook={hook_name}"
                )

        warn_task = asyncio.create_task(warn_if_stuck()) if stuck_warn_sec > 0 else None
        try:
            await hook(client, item, record)
            hook_record["status"] = StageStatus.COMPLETED.value
        except Exception as exc:
            hook_record["status"] = StageStatus.FAILED.value
            hook_record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            done = True
            if warn_task is not None:
                warn_task.cancel()
            elapsed = time.monotonic() - started
            hook_record["finished_at"] = time.monotonic()
            hook_record["duration_sec"] = elapsed
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
    """Per-run sandbox client pool: lazily acquires + caches clients by name.

    One ``SandboxPool`` instance is created at the start of ``Runner.run`` and
    released in the ``finally`` block.  The pool owns:

      - ``provider.create`` / ``provider.delete`` lifecycle
      - retry loop with health-check polling on acquire
      - the ``client`` / ``env_id`` / ``url`` triplet per name
      - sandbox-name validation against the configured spec map

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
    ):
        self._provider = create_object(provider)
        self._specs: dict[str, SandboxSpec] = {name: create_object(spec) for name, spec in specs.items()}
        self._max_attempts = max_attempts
        self._health_max_wait_sec = health_max_wait_sec
        self._health_poll_interval_sec = health_poll_interval_sec
        self._clients: dict[str, Any] = {}
        self._env_ids: dict[str, str] = {}
        self._urls: dict[str, str | None] = {}

    async def get(self, name: str, *, record: StageRecord | None = None) -> Any:
        if name in self._clients:
            return self._clients[name]
        self.validate_name(name)
        spec = self._specs[name]
        try:
            client, env_id = await self._acquire_ready(spec)
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
        self._clients[name] = client
        self._env_ids[name] = env_id
        self._urls[name] = self._url_of(client)
        return client

    def validate_name(self, name: str) -> None:
        if name not in self._specs:
            raise KeyError(f"unknown sandbox {name!r}; known: {sorted(self._specs)}")

    def env_id(self, name: str) -> str | None:
        return self._env_ids.get(name)

    def url(self, name: str) -> str | None:
        return self._urls.get(name)

    def spec(self, name: str) -> SandboxSpec:
        return self._specs[name]

    async def release_all(self) -> None:
        for name in list(reversed(self._clients)):
            client = self._clients.pop(name)
            env_id = self._env_ids.pop(name, None)
            self._urls.pop(name, None)
            if env_id is not None:
                try:
                    await self._provider.delete(env_id)
                except Exception as exc:
                    get_logger().warning(
                        f"gateway delete failed for sandbox {name} env_id={env_id}:\n"
                        f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                    )
            try:
                await client.aclose()
            except Exception as exc:
                get_logger().warning(
                    f"client aclose failed for sandbox {name} env_id={env_id}:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )

    @staticmethod
    def _url_of(client: Any) -> str | None:
        for attr in ("base_url", "url", "endpoint", "_base_url"):
            val = getattr(client, attr, None)
            if val:
                return str(val)
        return None

    async def _acquire_ready(self, spec: SandboxSpec) -> tuple[Any, str]:
        last_err: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                create_kwargs: dict[str, Any] = {}
                if spec.key:
                    create_kwargs["key"] = spec.key
                if spec.env_vars:
                    create_kwargs["env_vars"] = spec.env_vars
                if spec.resources:
                    create_kwargs["resources"] = spec.resources
                client, env_id = await self._provider.create(
                    image_tag=spec.image,
                    ttl_seconds=spec.ttl_seconds,
                    **create_kwargs,
                )
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(min(2**attempt, 8))
                continue

            if await self._wait_healthy(client):
                return client, env_id

            try:
                await self._provider.delete(env_id)
            except Exception as exc:
                get_logger().warning(
                    f"delete of unhealthy sandbox env_id={env_id} failed:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )
            try:
                await client.aclose()
            except Exception as exc:
                get_logger().warning(
                    f"aclose of unhealthy sandbox env_id={env_id} failed:\n"
                    f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()}"
                )
            last_err = RuntimeError(f"sandbox {env_id} unhealthy")

        last_err_msg = (
            "".join(traceback.format_exception(type(last_err), last_err, last_err.__traceback__)).rstrip()
            if last_err is not None
            else "unknown"
        )
        raise RuntimeError(f"could not acquire a healthy sandbox after {self._max_attempts} attempts: {last_err_msg}")

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
        raise RuntimeError(f"command failed (return_code={rc}): {command}\nstderr: {result.get('stderr', '')}")
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


def _outcome_error_category(outcome: EntryOutcome) -> str:
    if outcome.reason in {"timeout", "sandbox_unreachable", "pid_lost", "pid_file_missing"}:
        return "pid_lost" if outcome.reason == "pid_file_missing" else outcome.reason
    return outcome.reason or "entry"
