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
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xtuner.v1.utils import get_logger


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


class SandboxStage:
    """Pre-hooks → entry → post-hooks.  Each field is visible.

    Not every stage needs every phase:
      - ``entry`` can be ``None`` (pure setup/teardown stage).
      - ``pre`` / ``post`` default to empty.
      - Downloads live in post-hooks via :class:`DownloadHook` — no separate
        ``pull`` contract.
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
    ):
        self.sandbox = sandbox
        self.pre = list(pre)
        self._entry = entry
        self._env = env
        self.timeout = timeout
        self.post = list(post)

    async def run(self, client: Any, ctx: dict[str, Any]) -> StageResult:
        for hook in self.pre:
            await _run_hook(hook, client, ctx, phase="pre")

        stdout = stderr = ""
        rc = 0

        entry = _resolve(self._entry, ctx)
        if entry:
            env = _resolve(self._env, ctx) or {}
            exec_res = await exec_in(
                client,
                entry,
                env=env,
                timeout_sec=self.timeout,
                raise_on_error=True,
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

        for hook in self.post:
            await _run_hook(hook, client, ctx, phase="post")

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
) -> dict[str, Any]:
    """Execute a shell command inside the sandbox.

    Raises on failure by default.
    """
    if env:
        command = _expand_vars(command, env)
        prefix = " ".join(f'{k}="{v}"' for k, v in env.items())
        command = f"{prefix} {command}"
    result = await client.execute(command, cwd, timeout_sec)
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
    _log_bundle_size(len(blob), extract_root, len(file_map))
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


async def _run_hook(hook: Hook, client: Any, ctx: dict[str, Any], *, phase: str) -> None:
    """Run one hook; on failure, log + stash + re-raise with a label that names
    the hook class so the traceback says which one blew up."""
    name = getattr(hook, "name", None) or type(hook).__name__
    label = f"{phase}-hook {type(hook).__name__}({name!r})"
    tid = (ctx.get("data") and getattr(ctx["data"], "id", None)) or "?"
    get_logger().debug(f"[{tid}] {label} start")
    t0 = time.monotonic()
    try:
        await hook(client, ctx)
        get_logger().debug(f"[{tid}] {label} done ({time.monotonic() - t0:.2f}s)")
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
