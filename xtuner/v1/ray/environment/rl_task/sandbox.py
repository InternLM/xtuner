"""Sandbox primitives: hooks + stage execution + low-level HTTP.

A :class:`SandboxStage` is a sequence of phases::

    pre-hooks  →  entry command  →  pull declared paths  →  post-hooks

Each hook is a callable with the uniform signature::

    async def hook(client, ctx) -> None

where ``ctx`` is a mutable dict threaded through the whole stage — earlier
hooks read it (``ctx["task_root"]``, ``ctx["data"]``, …) and later hooks
write to it (``ctx["chosen_agent"]``, ``ctx["result"]``, …).  Three
primitive hook classes (:class:`UploadHook`, :class:`ExecHook`,
:class:`DownloadHook`) cover 80% of stage plumbing; specialized hooks
live in ``hooks.py``.

Reading a stage config top-to-bottom tells you the full execution order.
No hidden ``prepare()`` methods, no class-level mystery.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import io
import json
import os
import re
import shlex
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from xtuner.v1.utils import get_logger


class DetachConfig(TypedDict, total=False):
    """Options for ``SandboxStage(detach=...)``.

    All fields are optional (``total=False``) — pass ``{}`` for a
    minimal detach (background + poll rc file only, default timeouts).

    Keys:
        daemon_pattern (str | None): ``pgrep -f`` substring that the
            stage's companion daemon should match while the entry runs.
            ``None`` or unset → skip daemon liveness check.  Example
            for the lagent agent loop: ``"lagent.serving.sandbox.daemon"``.
        poll_sec (float): Seconds between rc/pid/pgrep polls.  Defaults
            to 30s; drop to 5–10s for short detached entries.
        probe_timeout_sec (float): Per-probe HTTP budget for the rc/pid
            reads and ``kill -0`` / ``pgrep`` calls.  Defaults to 10s.
            Raise it if cross-cluster latency makes individual probes
            time out even when the sandbox is alive.
        handshake_timeout_sec (float): HTTP budget for the initial
            detach ``exec_in`` call that just backgrounds the wrapper.
            Defaults to 60s.  The wrapper itself runs asynchronously; this
            cap only protects against a dead sandbox that never ACKs.
    """

    daemon_pattern: str | None
    poll_sec: float
    probe_timeout_sec: float
    handshake_timeout_sec: float


# ─────────────────────────────────────────────────────────────────
# Detached-entry polling
# ─────────────────────────────────────────────────────────────────
#
# ``SandboxStage.run`` with ``detach=True`` launches the entry command in
# the background and relies on external polling to detect completion:
#
#   1. ``{rc_file}``: written by the wrapping shell once the entry returns.
#      Presence = finished; content = return code.
#   2. ``{pid_file}``: written before the entry starts; ``kill -0 <pid>``
#      reports whether the wrapping shell is still alive.
#   3. ``pgrep -f <daemon_pattern>``: if the stage's companion daemon has
#      died while the entry is still running we fail immediately.  Each
#      stage declares its own pattern (``SandboxStage(daemon_pattern=...)``);
#      stages that don't spawn a daemon leave it ``None`` to skip.
#
# File paths are **per-call** — ``SandboxStage.run`` generates a unique
# suffix for each invocation.  Earlier builds used fixed paths
# ``/tmp/lagent_entry.{pid,rc}`` which caused a catastrophic race when two
# detached stages ran in the same sandbox: the judger's first poll would
# read the infer stage's stale rc file instantly, the judger's entry would
# be skipped, ``total`` scored as 0, and training data became silently
# corrupted.  Keeping the paths per-call isolates stages.
#
# This strategy replaces the old heartbeat-file + daemon-log watchdog —
# that design gave hundreds of false positives on cross-cluster
# ``/download`` spikes in rc18/rc19 (healthy samples killed mid-rollout).


_BUNDLE_SIZE_LOG = Path("/mnt/shared-storage-user/llmit/user/liukuikun/workspace/xtuner/work_dir/bundle_sizes.jsonl")


def _log_bundle_size(size: int, extract_root: str, file_count: int) -> None:
    """Append one JSON line describing this upload's tar size.

    Silent on failure — the log file is observational, not required.
    """
    try:
        _BUNDLE_SIZE_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "size_bytes": size,
            "file_count": file_count,
            "extract_root": extract_root,
            "pid": os.getpid(),
        }
        with _BUNDLE_SIZE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        get_logger().warning(f"bundle-size log failed: {exc}")


# ─────────────────────────────────────────────────────────────────
# StageResult
# ─────────────────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Outcome of a single :class:`SandboxStage` execution."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    pulled: dict[str, bytes] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.return_code == 0


# ─────────────────────────────────────────────────────────────────
# Hook: base + three primitives
# ─────────────────────────────────────────────────────────────────


class Hook:
    """A named step in a stage's pre or post pipeline.

    Subclasses implement ``__call__(client, ctx)``.  The ``ctx`` dict is
    the stage-wide scratchpad — hooks read inputs from it and write
    outputs to it.  Name it in ``name`` so logs/errors identify the hook.
    """

    name: str = "hook"

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        raise NotImplementedError


# Type alias: anything callable that resolves to a value given the ctx.
# Lets hooks accept either a literal or a function(ctx) -> literal.
Resolvable = Any  # literal value OR a zero-arg function of (ctx) -> value


def _resolve(value: Resolvable, ctx: dict[str, Any]) -> Any:
    return value(ctx) if callable(value) else value


class UploadHook(Hook):
    """Upload files via a list of explicit source→target mappings.

    Each mapping is a dict (or :class:`UploadMapping`) with:
      - ``source`` (str): glob or literal path relative to ``base``.  Prefix
        with ``"re:"`` to treat as a regex against POSIX-style relative paths.
      - ``target`` (str): sandbox destination.  Ending with ``/`` means
        "directory; preserve tree under source base"; otherwise the matched
        source must resolve to exactly one file placed at this exact path.
      - ``base`` (str | None): root that ``source`` is resolved against.
        Defaults to ``ctx["task_root"]`` at run time.
      - ``exclude`` (list[str]): glob or ``re:`` patterns matched against the
        relative path; matches are skipped.
      - ``flatten`` (bool): collapse the relative path — every match lands
        as ``target/<filename>``.

    Reading a list of these tells you exactly what gets uploaded without
    running anything.
    """

    name = "upload"

    def __init__(self, mappings: list[dict | UploadMapping]):
        self.mappings: list[UploadMapping] = [
            m if isinstance(m, UploadMapping) else UploadMapping(**m) for m in mappings
        ]

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        files: dict[str, Path] = {}
        for m in self.mappings:
            files.update(_resolve_mapping(m, ctx))
        if files:
            await upload_tar_and_extract(client, files, "/")


class ReadFileHook(Hook):
    """Read a sandbox file and store its text content in ``ctx["pulled"]``.

    Unlike :class:`DownloadHook` (which stores raw bytes and auto-detects
    files vs dirs), this hook is intentionally simple: it reads a single
    text file via ``exec_in`` and writes the decoded string into
    ``ctx["pulled"][key]``.

    Args:
        path: Sandbox path to read.  May be a literal string or a
            callable ``(ctx) -> str``.
        key:  Key under which the content is stored in ``ctx["pulled"]``.
            May be a literal string or a callable ``(ctx) -> str``.
        encoding: Text encoding used to decode the file bytes.
            Defaults to ``"utf-8"``.
        errors: Error handler passed to ``bytes.decode``.
            Defaults to ``"replace"``.
        optional: When ``True`` a missing / unreadable file logs a warning
            instead of raising.  Defaults to ``False``.
    """

    name = "read_file"

    def __init__(
        self,
        path: Resolvable,
        key: Resolvable,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        optional: bool = False,
    ):
        self.path = path
        self.key = key
        self.encoding = encoding
        self.errors = errors
        self.optional = optional

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        path = _resolve(self.path, ctx)
        key = _resolve(self.key, ctx)
        try:
            blob = await client.download_file(path)
            content = blob.decode(self.encoding, errors=self.errors)
            ctx.setdefault("pulled", {})[key] = content
        except Exception as exc:
            if self.optional:
                get_logger().warning("read_file %s (key=%r) failed: %s", path, key, exc)
            else:
                raise


@dataclass
class UploadMapping:
    source: str
    target: str
    base: str | None = None
    exclude: list[str] = field(default_factory=list)
    flatten: bool = False


class ExecHook(Hook):
    """Run a shell command in the sandbox with env vars.

    ``cmd`` / ``env`` may be literals or callable(ctx).  Set
    ``optional=True`` to downgrade failures to warnings (useful for
    install-deps etc. that may no-op).
    """

    name = "exec"

    def __init__(
        self,
        cmd: Resolvable,
        *,
        env: Resolvable = None,
        timeout: int = 60,
        optional: bool = False,
    ):
        self.cmd = cmd
        self.env = env
        self.timeout = timeout
        self.optional = optional

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        cmd = _resolve(self.cmd, ctx)
        env = _resolve(self.env, ctx) or {}
        await exec_in(
            client,
            cmd,
            env=env,
            timeout_sec=self.timeout,
            raise_on_error=not self.optional,
        )


class DownloadHook(Hook):
    """Pull sandbox paths into ``ctx["pulled"]``.

    Each entry is auto-detected as file vs directory:
      - **file** → bytes of the file (from ``/download`` endpoint)
      - **directory** → bytes of a gzipped tar produced in-sandbox
        (sandbox ``/download`` only serves files, so dirs are tarred first)

    ``ctx["pulled"]`` is ``{path: bytes}`` in both cases; ``ctx["pulled_kinds"]``
    records ``{path: "file" | "dir"}`` so consumers know how to interpret.
    """

    name = "download"

    def __init__(self, paths: Resolvable):
        self.paths = paths

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        paths = _resolve(self.paths, ctx)
        pulled = ctx.setdefault("pulled", {})
        kinds = ctx.setdefault("pulled_kinds", {})
        for p in paths:
            try:
                blob, kind = await download_path(client, p)
                pulled[p] = blob
                kinds[p] = kind
            except Exception as exc:
                get_logger().warning(f"download {p} failed: {exc}")


# ─────────────────────────────────────────────────────────────────
# SandboxStage
# ─────────────────────────────────────────────────────────────────


async def _sandbox_cat(client: Any, path: str, timeout_sec: float = 5.0) -> bytes | None:
    """Download a sandbox file; return ``None`` on any failure."""
    try:
        return await asyncio.wait_for(client.download_file(path), timeout=timeout_sec)
    except Exception:
        return None


async def _read_entry_rc(client: Any, rc_file: str) -> int | None:
    """Read ``rc_file`` content and parse as int.

    Returns ``None`` when
    the file doesn't exist yet (entry hasn't finished) or is malformed.
    """
    blob = await _sandbox_cat(client, rc_file)
    if blob is None:
        return None
    try:
        return int(blob.decode(errors="replace").strip())
    except ValueError:
        return None


async def _read_entry_pid(client: Any, pid_file: str) -> int | None:
    blob = await _sandbox_cat(client, pid_file)
    if blob is None:
        return None
    try:
        return int(blob.decode(errors="replace").strip())
    except ValueError:
        return None


async def _is_pid_alive(client: Any, pid: int, timeout_sec: float) -> bool | None:
    """``True`` = alive, ``False`` = gone, ``None`` = probe failed."""
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


async def _is_pattern_running(client: Any, pattern: str, timeout_sec: float) -> bool | None:
    """Pgrep-style liveness probe for a substring match on full command
    line."""
    try:
        res = await exec_in(
            client,
            f"pgrep -f {shlex.quote(pattern)} >/dev/null && echo Y || echo N",
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


async def wait_for_detached_entry(
    client: Any,
    tid: str,
    *,
    pid_file: str,
    rc_file: str,
    daemon_pattern: str | None,
    poll_sec: float,
    probe_timeout_sec: float,
    max_sec: int,
) -> dict[str, Any]:
    """Poll the sandbox until a detached entry command finishes or fails.

    Returns a dict with the same shape as ``client.execute`` results
    (``return_code`` / ``stdout`` / ``stderr``) so callers can treat it
    like a normal synchronous ``exec_in`` result.

    Non-zero ``return_code`` encodings we add on top of the entry's own rc:

      * ``-1`` — entry shell died without writing ``rc_file``
        (SIGKILL, node eviction, or fatal shell error).
      * ``-2`` — ``daemon_pattern`` no longer matches ``pgrep`` while
        the entry was still running.  Skipped if ``daemon_pattern`` is
        ``None`` (stage has no external daemon to monitor, e.g. judgers).
      * ``-3`` — exceeded ``max_sec`` (safety ceiling, default 2h).
      * ``-4`` — sandbox env itself has become unreachable for
        ``sandbox_dead_consecutive`` consecutive polls.  This catches the
        "gateway TTL killed the env under us" case that used to hide
        behind probes silently returning ``None`` — pre-fix those hangs
        quietly spun until ``max_sec`` and surfaced as either a bogus
        ``entry_timeout`` or a misleading post-hook 404.

    Args:
        client (Any): SandboxClient.
        tid (str): Task id for log tags.
        pid_file (str): Sandbox path of the wrapper shell's PID file;
            must be unique per call to avoid cross-stage contamination.
        rc_file (str): Sandbox path of the wrapper shell's exit-code file;
            must be unique per call.
        daemon_pattern (str | None): ``pgrep -f`` substring that the
            stage's companion daemon process should match.  When ``None``,
            the daemon-alive check is skipped entirely — appropriate for
            stages that don't spawn or depend on a background process
            (e.g. a judger's one-shot test runner).
        poll_sec (float): Seconds between polls.
        probe_timeout_sec (float): Per-probe HTTP budget for rc/pid reads
            and ``kill -0`` / ``pgrep`` calls.
        max_sec (int): Hard ceiling on total wait.
    """
    start = time.monotonic()
    consecutive_entry_missing = 0
    # Every probe returns ``None`` when the underlying HTTP call failed
    # (timeout, 404, connection reset).  A single ``None`` can be transient
    # cross-cluster noise; a run of them means the sandbox env is gone
    # (TTL expired, gateway evicted, container killed).  We need both
    # ``_read_*`` probes AND ``_is_pattern_running`` (when applicable)
    # to come back ``None`` on the same poll for it to count as unreachable
    # — that rules out a single flaky endpoint.
    consecutive_sandbox_unreachable = 0
    _SANDBOX_DEAD_THRESHOLD = 3
    logger = get_logger()
    while True:
        rc = await _read_entry_rc(client, rc_file)
        if rc is not None:
            logger.info(f"[{tid}] detached entry finished rc={rc}")
            return {"return_code": rc, "stdout": "", "stderr": ""}

        entry_pid = await _read_entry_pid(client, pid_file)
        entry_alive = await _is_pid_alive(client, entry_pid, probe_timeout_sec) if entry_pid else None
        daemon_alive = await _is_pattern_running(client, daemon_pattern, probe_timeout_sec) if daemon_pattern else None

        # "Sandbox unreachable" = every probe we could run this poll came
        # back ``None`` AND the rc file is also unreadable.  When daemon
        # check is disabled (daemon_pattern is None), we skip that leg.
        rc_probe_failed = rc is None  # already tried above
        pid_probe_failed = entry_pid is None
        daemon_probe_failed = daemon_pattern is not None and daemon_alive is None
        all_failed = rc_probe_failed and pid_probe_failed and (daemon_probe_failed or daemon_pattern is None)
        if all_failed:
            consecutive_sandbox_unreachable += 1
            logger.warning(
                f"[{tid}] sandbox probes all failing ({consecutive_sandbox_unreachable}/{_SANDBOX_DEAD_THRESHOLD})"
            )
            if consecutive_sandbox_unreachable >= _SANDBOX_DEAD_THRESHOLD:
                logger.warning(f"[{tid}] sandbox appears dead — declaring rc=-4")
                return {
                    "return_code": -4,
                    "stdout": "",
                    "stderr": f"[{tid}] sandbox unreachable for {consecutive_sandbox_unreachable} consecutive polls",
                }
        else:
            consecutive_sandbox_unreachable = 0

        if daemon_alive is False:
            logger.warning(f"[{tid}] daemon process (pattern={daemon_pattern!r}) gone while entry still running")
            return {
                "return_code": -2,
                "stdout": "",
                "stderr": f"[{tid}] daemon process gone (pattern={daemon_pattern!r})",
            }

        if entry_alive is False:
            consecutive_entry_missing += 1
            # One poll of "missing" can be a race with the rc-write step
            # — the shell may have exited moments before we re-read the
            # pid.  Require two consecutive misses + rc file still absent
            # before declaring a crash.
            if consecutive_entry_missing >= 2:
                rc = await _read_entry_rc(client, rc_file)
                if rc is not None:
                    return {"return_code": rc, "stdout": "", "stderr": ""}
                logger.warning(f"[{tid}] lagent_entry pid {entry_pid} gone without writing rc file")
                return {
                    "return_code": -1,
                    "stdout": "",
                    "stderr": f"[{tid}] lagent_entry pid {entry_pid} gone without writing rc file",
                }
        else:
            consecutive_entry_missing = 0

        elapsed = time.monotonic() - start
        if elapsed > max_sec:
            logger.warning(f"[{tid}] detached entry exceeded max runtime {max_sec}s")
            return {
                "return_code": -3,
                "stdout": "",
                "stderr": f"[{tid}] entry exceeded max runtime {max_sec}s",
            }

        await asyncio.sleep(poll_sec)


class SandboxStage:
    """Pre-hooks → entry → post-hooks.  Each field is visible.

    Not every stage needs every phase:
      - ``entry`` can be ``None`` (pure setup/teardown stage).
      - ``pre`` / ``post`` default to empty.
      - Downloads live in post-hooks via :class:`DownloadHook` — no separate
        ``pull`` contract.

    ``detach`` controls how the entry command is invoked:

    * ``None`` (default): synchronous ``exec_in`` — the sandbox HTTP call
      blocks until the command finishes; stdout/stderr are captured and
      returned.  Use this for short-lived stages (judgers, setup) that
      need the captured output for parsing.
    * ``{}`` or populated :class:`DetachConfig`: entry is launched in
      background via ``detach=True`` with a shell wrapper that writes its
      own PID and the exit code to per-call paths under ``/tmp/``.  The
      host polls those files + ``pgrep`` (when ``daemon_pattern`` is set)
      to detect completion or daemon death.  Use this for long-lived
      rollout stages whose runtime exceeds the HTTP read timeout on
      cross-cluster networks.  Stdout/stderr are NOT captured — stages
      that need them must consume output via files pulled by post-hooks.
    """

    def __init__(
        self,
        *,
        sandbox: Any = None,
        pre: list[Hook] = (),
        entry: Resolvable = None,
        env: Resolvable = None,
        timeout: int = 600,
        post: list[Hook] = (),
        detach: DetachConfig | None = None,
    ):
        self.sandbox = sandbox
        self.pre = list(pre)
        self._entry = entry
        self._env = env
        # ``timeout`` is the end-to-end wall-clock ceiling for ``entry``:
        # in sync mode it's the HTTP read timeout passed to ``exec_in``;
        # in detach mode it bounds the background poll loop.
        self.timeout = timeout
        self.post = list(post)
        # Unpack detach config once at init.  ``detach is None`` → sync
        # path; any dict (even ``{}``) opts into the detach path.
        self.detach = detach is not None
        if detach is None:
            detach = {}
        self.daemon_pattern: str | None = detach.get("daemon_pattern")
        self.poll_sec: float = detach.get("poll_sec", 30.0)
        self.probe_timeout_sec: float = detach.get("probe_timeout_sec", 10.0)
        self.handshake_timeout_sec: float = detach.get("handshake_timeout_sec", 60.0)

    async def run(self, client: Any, ctx: dict[str, Any]) -> StageResult:
        for hook in self.pre:
            await _run_hook(hook, client, ctx, phase="pre")

        stdout = stderr = ""
        rc = 0

        entry = _resolve(self._entry, ctx)
        if entry:
            env = _resolve(self._env, ctx) or {}
            tid = (ctx.get("data") and getattr(ctx["data"], "id", None)) or "?"

            if self.detach:
                # Long-running entry: wrap with PID/rc files so external polling
                # can detect completion without holding an HTTP connection for
                # the full rollout.  The file paths are generated per call so
                # stages that share a sandbox (infer → judger) cannot race on
                # a stale rc left by the previous stage.
                stage_id = uuid.uuid4().hex[:12]
                pid_file = f"/tmp/xt_stage_{stage_id}.pid"
                rc_file = f"/tmp/xt_stage_{stage_id}.rc"
                wrapped = f"rm -f {pid_file} {rc_file}; echo $$ > {pid_file}; {entry}; echo $? > {rc_file}"
                # The detach HTTP handshake itself should return promptly —
                # it just tells the sandbox to background the command.  Use
                # a short cap rather than ``self.timeout`` so a dead sandbox
                # doesn't block for hours before failing over.
                await exec_in(
                    client,
                    wrapped,
                    env=env,
                    timeout_sec=self.handshake_timeout_sec,
                    raise_on_error=True,
                    detach=True,
                )
                exec_res = await wait_for_detached_entry(
                    client,
                    tid,
                    pid_file=pid_file,
                    rc_file=rc_file,
                    daemon_pattern=self.daemon_pattern,
                    poll_sec=self.poll_sec,
                    probe_timeout_sec=self.probe_timeout_sec,
                    max_sec=self.timeout,
                )
            else:
                # Synchronous entry: ``exec_in`` blocks until the command
                # returns and captures stdout/stderr for post-hook parsing
                # (e.g. ParseJudgerStdout).
                exec_res = await exec_in(
                    client,
                    entry,
                    env=env,
                    timeout_sec=self.timeout,
                    raise_on_error=False,
                )
            rc = _result_code(exec_res)
            stdout = exec_res.get("stdout") or ""
            stderr = exec_res.get("stderr") or ""

        result = StageResult(
            stdout=stdout,
            stderr=stderr,
            return_code=rc,
            pulled=ctx.get("pulled", {}),
            error=None if rc == 0 else f"return_code={rc}: {stderr[:400]}",
        )
        ctx["result"] = result

        # Post-hooks run best-effort when the entry failed.  A hung entry
        # (rc=-3 timeout, rc=-1 pid lost, rc=-2 daemon gone, non-zero rc)
        # typically leaves the sandbox in a broken state where downloads
        # 404 or the file the hook wants isn't on disk yet.  If we let those
        # hook exceptions propagate, they replace the real entry error in
        # ``run_single``'s ``except Exception`` handler — and the fate is
        # misclassified (rc22: 3468 ``posthook_download_404`` fates that
        # were actually ``entry_timeout``).  Swallow hook failures in the
        # entry-failed path but keep them raising on the happy path so real
        # post-hook bugs still surface.
        entry_failed = rc != 0
        for hook in self.post:
            try:
                await _run_hook(hook, client, ctx, phase="post")
            except Exception as hook_exc:
                if entry_failed:
                    get_logger().warning(
                        f"post-hook {type(hook).__name__!r} failed after entry "
                        f"rc={rc}; preserving entry error: "
                        f"{type(hook_exc).__name__}: {hook_exc}"
                    )
                else:
                    raise

        # Post-hooks may have added downloads; reflect into the returned result.
        result.pulled = ctx.get("pulled", {})
        return result


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
    timeout_sec: int = 600,
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
        # when entry runs pre_entry.sh && lagent_entry.sh — daemon subprocess
        # wouldn't see RL_LLM_MODEL etc.
        exports = "; ".join(f'export {k}="{v}"' for k, v in env.items())
        command = f"{exports}; {command}"
    result = await client.execute(command, cwd, timeout_sec, detach)
    rc = _result_code(result)
    if raise_on_error and rc != 0:
        raise RuntimeError(f"command failed (return_code={rc}): {command}\nstderr: {result.get('stderr', '')[:1000]}")
    return result


async def http_upload(client: Any, target_path: str, content_b64: str) -> None:
    await client.upload_bytes(target_path, base64.b64decode(content_b64))


async def upload_tar_and_extract(
    client: Any,
    file_map: dict[str, Path],
    extract_root: str,
) -> None:
    if not file_map:
        return
    blob = await asyncio.to_thread(_tar_bytes, file_map, extract_root)
    # _log_bundle_size(len(blob), extract_root, len(file_map))
    tmp = "/tmp/_bundle.tar.gz"
    await http_upload(client, tmp, base64.b64encode(blob).decode())
    await exec_in(
        client,
        f"mkdir -p {extract_root} && cd {extract_root} && tar xzf {tmp}",
    )


async def download_path(client: Any, remote_path: str) -> tuple[bytes, str]:
    """Download a sandbox file or directory.

    Files come back via the ``/download`` endpoint.  Directories are tarred
    in-sandbox first (``/download`` only serves files), then the tar is
    downloaded and the tmp tar is cleaned up.

    Returns:
        tuple[bytes, str]: ``(content, kind)`` where ``kind`` is ``"file"``
        or ``"dir"``.  For ``"dir"`` the bytes are a gzipped tarball —
        caller unpacks with ``tarfile.open(fileobj=io.BytesIO(content))``.
    """
    check = await exec_in(
        client,
        f'test -d "{remote_path}" && echo DIR || echo FILE',
        raise_on_error=False,
    )
    is_dir = "DIR" in (check.get("stdout") or "")

    if not is_dir:
        blob = await client.download_file(remote_path)
        return blob, "file"

    # Tar the dir into /tmp, download, remove the tmp.
    rp = remote_path.rstrip("/") or "/"
    parent = rp.rsplit("/", 1)[0] or "/"
    name = rp.rsplit("/", 1)[-1] or "root"
    tar_remote = f"/tmp/_dl_{name}.tar.gz"
    await exec_in(
        client,
        f'cd "{parent}" && tar czf {tar_remote} "{name}"',
    )
    try:
        blob = await client.download_file(tar_remote)
    finally:
        await exec_in(client, f"rm -f {tar_remote}", raise_on_error=False)
    return blob, "dir"


def tar_dir(src: Path, arcname_prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in walk_files(src):
            rel = str(f.relative_to(src))
            arc = f"{arcname_prefix}/{rel}" if arcname_prefix else rel
            tar.add(f, arcname=arc, recursive=False)
    return buf.getvalue()


def walk_files(root: Path) -> list[Path]:
    """Recursive file list, skipping pycache / .pyc."""
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]


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


_HOOK_STUCK_WARN_SEC = 30.0


async def _run_hook(hook: Hook, client: Any, ctx: dict[str, Any], *, phase: str) -> None:
    """Run one hook; on failure, log + stash + re-raise with a label that names
    the hook class so the traceback says which one blew up."""
    name = getattr(hook, "name", None) or type(hook).__name__
    tid = (ctx.get("data") and getattr(ctx["data"], "id", None)) or "?"
    image = ctx.get("sandbox_image") or "?"
    label = f"{phase}-hook {type(hook).__name__}({name!r}) image={image}"
    get_logger().info(f"[{tid}] {label} start")
    t0 = time.monotonic()
    hook_task = asyncio.create_task(hook(client, ctx))
    try:
        while True:
            done, _ = await asyncio.wait([hook_task], timeout=_HOOK_STUCK_WARN_SEC)
            if done:
                break
            get_logger().warning(f"[{tid}] {label} still running ({time.monotonic() - t0:.0f}s)")
        await hook_task  # re-raise if hook failed
        get_logger().info(f"[{tid}] {label} done ({time.monotonic() - t0:.2f}s)")
    except Exception as exc:
        import traceback as _tb

        get_logger().error(f"[{tid}] {label} failed: {exc}\n{_tb.format_exc()}")
        ctx.setdefault("hook_errors", []).append(
            {
                "phase": phase,
                "hook": type(hook).__name__,
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise RuntimeError(f"{label} failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────
# UploadHook: source-matching (glob / regex / literal)
# ─────────────────────────────────────────────────────────────────


def _resolve_mapping(m: UploadMapping, ctx: dict[str, Any]) -> dict[str, Path]:
    # ``base`` is absolute if set and absolute; if relative, resolve against
    # ctx["task_root"]; if None, defaults to ctx["task_root"].
    if m.base is None:
        base = Path(ctx["task_root"])
    else:
        base = Path(m.base)
        if not base.is_absolute():
            base = Path(ctx["task_root"]) / base

    matches = _match_source(base, m.source)
    matches = [p for p in matches if not _is_excluded(p, base, m.exclude)]

    dir_target = m.target.endswith("/")
    files: dict[str, Path] = {}
    if not dir_target:
        if len(matches) != 1:
            raise ValueError(
                f"UploadMapping target {m.target!r} is a file path but source "
                f"{m.source!r} under {base} matched {len(matches)} files"
            )
        files[m.target] = matches[0]
        return files

    target = m.target.rstrip("/")
    for host in matches:
        if m.flatten:
            sb = f"{target}/{host.name}"
        else:
            rel = host.relative_to(base).as_posix()
            sb = f"{target}/{rel}"
        files[sb] = host
    return files


def _match_source(base: Path, pattern: str) -> list[Path]:
    """Return host files matching ``pattern`` under ``base``.

    Supports:
      - literal path (file → [that file]; dir → rglob)
      - glob (contains any of ``* ? [``), via :meth:`pathlib.Path.glob`
      - regex (``re:<regex>``) against POSIX rel path
    """
    if pattern.startswith("re:"):
        rx = re.compile(pattern[3:])
        return sorted(
            p for p in base.rglob("*") if p.is_file() and not _ignored(p) and rx.search(p.relative_to(base).as_posix())
        )

    if any(ch in pattern for ch in "*?["):
        return sorted(p for p in base.glob(pattern) if p.is_file() and not _ignored(p))

    p = base / pattern
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and not _ignored(f))
    return []


def _is_excluded(host: Path, base: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    rel = host.relative_to(base).as_posix()
    for pat in patterns:
        if pat.startswith("re:"):
            if re.search(pat[3:], rel):
                return True
        elif fnmatch.fnmatch(rel, pat):
            return True
    return False


def _ignored(p: Path) -> bool:
    return "__pycache__" in p.parts or p.suffix == ".pyc"
