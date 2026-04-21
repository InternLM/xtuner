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
import logging
import os
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


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
            m if isinstance(m, UploadMapping) else UploadMapping(**m)
            for m in mappings
        ]

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        files: dict[str, Path] = {}
        for m in self.mappings:
            files.update(_resolve_mapping(m, ctx))
        if files:
            await upload_tar_and_extract(client, files, "/")


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
            client, cmd, env=env, timeout_sec=self.timeout,
            raise_on_error=not self.optional,
        )


class DownloadHook(Hook):
    """Pull sandbox paths into ``ctx["pulled"]`` (same shape as stage pull)."""

    name = "download"

    def __init__(self, paths: Resolvable):
        self.paths = paths

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        paths = _resolve(self.paths, ctx)
        pulled = ctx.setdefault("pulled", {})
        for p in paths:
            try:
                pulled[p] = await asyncio.to_thread(client.download_file, p)
            except Exception as exc:
                logger.warning("download %s failed: %s", p, exc)


# ─────────────────────────────────────────────────────────────────
# SandboxStage
# ─────────────────────────────────────────────────────────────────


class SandboxStage:
    """pre-hooks → entry → pull → post-hooks.  Each field is visible.

    Not every stage needs every phase:
      - ``entry`` can be ``None`` (pure setup/teardown stage).
      - ``pre`` / ``post`` default to empty.
      - ``pull`` fetches sandbox paths straight into ``ctx["pulled"]``.
    """

    def __init__(
        self,
        *,
        sandbox: Any = None,
        pre: list[Hook] = (),
        entry: Resolvable = None,
        env: Resolvable = None,
        timeout: int = 600,
        pull: Resolvable = (),
        post: list[Hook] = (),
    ):
        self.sandbox = sandbox
        self.pre = list(pre)
        self._entry = entry
        self._env = env
        self.timeout = timeout
        self._pull = pull
        self.post = list(post)

    async def run(self, client: Any, ctx: dict[str, Any]) -> StageResult:
        for hook in self.pre:
            await hook(client, ctx)

        stdout = stderr = ""
        rc = 0
        pulled: dict[str, bytes] = {}

        entry = _resolve(self._entry, ctx)
        if entry:
            env = _resolve(self._env, ctx) or {}
            exec_res = await exec_in(
                client, entry, env=env,
                timeout_sec=self.timeout, raise_on_error=False,
            )
            rc = _result_code(exec_res)
            stdout = exec_res.get("stdout") or ""
            stderr = exec_res.get("stderr") or ""

        for path in _resolve(self._pull, ctx) or []:
            try:
                pulled[path] = await asyncio.to_thread(client.download_file, path)
            except Exception as exc:
                logger.warning("pull %s failed: %s", path, exc)

        result = StageResult(
            stdout=stdout, stderr=stderr, return_code=rc, pulled=pulled,
            error=None if rc == 0 else f"return_code={rc}: {stderr[:400]}",
        )
        ctx["result"] = result

        for hook in self.post:
            await hook(client, ctx)

        return result


# ─────────────────────────────────────────────────────────────────
# Low-level sandbox I/O
# ─────────────────────────────────────────────────────────────────


_VAR_RE = re.compile(r"\$(\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_vars(text: str, env: dict[str, str]) -> str:
    """Substitute ``$VAR`` / ``${VAR}`` in ``text`` using ``env`` then process
    env.  Needed because inline ``KEY=val cmd`` doesn't expand ``$KEY``
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
    timeout_sec: int = 60,
    env: dict[str, str] | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Execute a shell command inside the sandbox.  Raises on failure by default."""
    if env:
        command = _expand_vars(command, env)
        prefix = " ".join(f'{k}="{v}"' for k, v in env.items())
        command = f"{prefix} {command}"
    result = await asyncio.to_thread(client.execute, command, cwd, timeout_sec)
    rc = _result_code(result)
    if raise_on_error and rc != 0:
        raise RuntimeError(
            f"command failed (return_code={rc}): {command}\n"
            f"stderr: {result.get('stderr', '')[:1000]}"
        )
    return result


async def http_upload(client: Any, target_path: str, content_b64: str) -> None:
    def _post() -> None:
        resp = client.session.post(
            f"{client.url}/upload",
            json={"target_path": target_path, "content_b64": content_b64},
        )
        resp.raise_for_status()

    await asyncio.to_thread(_post)


async def upload_tar_and_extract(
    client: Any, file_map: dict[str, Path], extract_root: str,
) -> None:
    if not file_map:
        return
    blob = await asyncio.to_thread(_tar_bytes, file_map, extract_root)
    tmp = "/tmp/_bundle.tar.gz"
    await http_upload(client, tmp, base64.b64encode(blob).decode())
    await exec_in(
        client,
        f"mkdir -p {extract_root} && cd {extract_root} && tar xzf {tmp} && rm {tmp}",
    )


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
    return [
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]


# ─────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────


def _tar_bytes(file_map: dict[str, Path], strip_root: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sb_path, host_path in file_map.items():
            if not sb_path.startswith(strip_root):
                raise ValueError(f"{sb_path} not under {strip_root}")
            arcname = sb_path[len(strip_root):].lstrip("/")
            tar.add(host_path, arcname=arcname, recursive=False)
    return buf.getvalue()


def _result_code(exec_res: dict[str, Any]) -> int:
    rc = exec_res.get("return_code")
    if rc is None:
        rc = exec_res.get("exit_code", 0 if exec_res.get("ok", True) else 1)
    return int(rc)


# ─────────────────────────────────────────────────────────────────
# UploadHook: source-matching (glob / regex / literal)
# ─────────────────────────────────────────────────────────────────


def _resolve_mapping(m: "UploadMapping", ctx: dict[str, Any]) -> dict[str, Path]:
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
            p for p in base.rglob("*")
            if p.is_file() and not _ignored(p) and rx.search(p.relative_to(base).as_posix())
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
