"""Hook implementations — primitive sandbox plumbing plus agent setup.

Each class here is a :class:`sandbox.Hook` subclass with a clear purpose
you can infer from its name.  Primitive hooks (Upload/Exec/Download) cover
the common case; specialized hooks encapsulate logic that's awkward to
express as a plain lambda (e.g. running agent config exec on host).

Hooks receive the current :class:`AgentRolloutItem` and the current
:class:`StageRecord`. They read task fields from the item and write stage
execution fields on the record.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import io
import json
import re
import shlex
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import (
    Hook,
    exec_in,
    http_upload,
    upload_tar_and_extract,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    AgentSpec,
    RolloutError,
    SelectedAgentRecord,
    StageRecord,
    StageResult,
    StageStatus,
)
from xtuner.v1.utils import get_logger


# ─────────────────────────────────────────────────────────────────
# Primitive hooks
# ─────────────────────────────────────────────────────────────────


@dataclass
class UploadMapping:
    source: str
    target: str
    base: str | None = None
    exclude: list[str] = field(default_factory=list)
    flatten: bool = False


class UploadHook(Hook):
    """Upload files via a list of explicit source→target mappings."""

    name = "upload"

    def __init__(self, mappings: list[dict | UploadMapping]):
        self.mappings: list[UploadMapping] = [
            m if isinstance(m, UploadMapping) else UploadMapping(**m) for m in mappings
        ]

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        files: dict[str, Path] = {}
        for m in self.mappings:
            files.update(self._resolve_mapping(m, item))
        if files:
            await upload_tar_and_extract(client, files, "/")

    def _resolve_mapping(self, m: UploadMapping, item: AgentRolloutItem) -> dict[str, Path]:
        task_root = item.task_root
        if task_root is None:
            raise ValueError("AgentRolloutItem.task_root is required by UploadHook")
        base = (task_root / m.base).resolve() if m.base else task_root.resolve()
        if not base.exists():
            return {}

        matches = self._match_source(base, m.source)
        matches = [p for p in matches if not self._is_excluded(p, base, m.exclude)]
        out: dict[str, Path] = {}
        for src in matches:
            rel = src.relative_to(base)
            if m.target.endswith("/"):
                dst = Path(m.target) / (src.name if m.flatten else rel)
                out[dst.as_posix()] = src
            else:
                if len(matches) != 1:
                    raise ValueError(f"target {m.target!r} is a file but {m.source!r} matched {len(matches)} files")
                out[m.target] = src
        return out

    def _match_source(self, base: Path, pattern: str) -> list[Path]:
        if pattern.startswith("re:"):
            rx = re.compile(pattern[3:])
            return sorted(
                p
                for p in base.rglob("*")
                if p.is_file() and not self._ignored(p) and rx.search(p.relative_to(base).as_posix())
            )
        hits = sorted(p for p in base.glob(pattern) if p.is_file() and not self._ignored(p))
        if hits:
            return hits
        p = base / pattern
        if p.is_dir():
            return sorted(f for f in p.rglob("*") if f.is_file() and not self._ignored(f))
        return []

    def _is_excluded(self, host: Path, base: Path, patterns: list[str]) -> bool:
        rel = host.relative_to(base).as_posix()
        for pat in patterns:
            if pat.startswith("re:"):
                if re.search(pat[3:], rel):
                    return True
            elif fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _ignored(self, p: Path) -> bool:
        parts = set(p.parts)
        return "__pycache__" in parts or ".git" in parts or p.suffix == ".pyc"


class ExecHook(Hook):
    """Run a shell command in the sandbox with env vars."""

    name = "exec"

    def __init__(
        self,
        cmd: str,
        *,
        env: dict[str, str] | None = None,
        timeout: int = 60,
        optional: bool = False,
    ):
        self.cmd = cmd
        self.env = env
        self.timeout = timeout
        self.optional = optional

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        await exec_in(
            client,
            self.cmd,
            env=self.env or {},
            timeout_sec=self.timeout,
            raise_on_error=not self.optional,
        )


@dataclass
class SharedPathMapping:
    source: str
    target: str
    contents: bool = False
    optional: bool = False


class LinkSharedPathHook(Hook):
    """Create sandbox symlinks to files already staged on a shared mount.

    ``source`` and ``target`` accept ``str.format`` placeholders for common
    item fields: ``{id}``, ``{data_source}``, ``{task_root}``, ``{uid}``, and
    ``{group_id}``.
    """

    name = "link_shared_path"

    def __init__(self, mappings: list[dict | SharedPathMapping], *, timeout: int = 60):
        self.mappings = [
            mapping if isinstance(mapping, SharedPathMapping) else SharedPathMapping(**mapping)
            for mapping in mappings
        ]
        self.timeout = timeout

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        commands = ["set -e"]
        for mapping in self.mappings:
            source = self._format(mapping.source, item)
            target = self._format(mapping.target, item)
            commands.append(self._link_command(source, target, contents=mapping.contents, optional=mapping.optional))
        await exec_in(client, "\n".join(commands), timeout_sec=self.timeout, raise_on_error=True)

    def _format(self, value: str, item: AgentRolloutItem) -> str:
        fields = {
            "id": item.id,
            "data_source": item.data_source,
            "task_root": item.task_root.as_posix() if item.task_root is not None else "",
            "sandbox_task_dir": str(item.metadata.get("sandbox_task_dir") or ""),
            "uid": "" if item.uid is None else str(item.uid),
            "group_id": "" if item.group_id is None else str(item.group_id),
        }
        return value.format(**fields)

    def _link_command(self, source: str, target: str, *, contents: bool, optional: bool) -> str:
        src = shlex.quote(source)
        dst = shlex.quote(target)
        missing = "exit 0" if optional else f"echo 'missing shared path: {source}' >&2; exit 2"
        if contents:
            return "\n".join(
                [
                    f"if [ ! -d {src} ]; then {missing}; fi",
                    f"mkdir -p {dst}",
                    f"cd {src}",
                    f"find . -type d -exec mkdir -p {dst}/{{}} \\;",
                    (
                        "find . -type f -exec sh -c "
                        + shlex.quote(
                            'src_root="$1"; dst_root="$2"; shift 2; '
                            'for rel in "$@"; do '
                            'mkdir -p "$dst_root/$(dirname "$rel")"; '
                            'ln -sfn "$src_root/$rel" "$dst_root/$rel"; '
                            'done'
                        )
                        + f" sh {src} {dst} {{}} +"
                    ),
                ]
            )
        return "\n".join(
            [
                f"if [ ! -e {src} ]; then {missing}; fi",
                f"mkdir -p $(dirname {dst})",
                f"rm -rf {dst}",
                f"ln -s {src} {dst}",
            ]
        )


class DownloadHook(Hook):
    """Pull sandbox paths into ``item.artifacts``."""

    name = "download"

    def __init__(self, paths: list[str]):
        self.paths = list(paths)

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        for path in self.paths:
            try:
                blob, _ = await self._download_path(client, path)
                item.artifacts[path] = blob
            except Exception as exc:
                get_logger().warning("download %s failed: %s", path, exc)

    async def _download_path(self, client: Any, remote_path: str) -> tuple[bytes, str]:
        check = await exec_in(
            client,
            f'test -d "{remote_path}" && echo DIR || echo FILE',
            raise_on_error=False,
        )
        is_dir = "DIR" in (check.get("stdout") or "")

        if not is_dir:
            blob = await client.download_file(remote_path)
            return blob, "file"

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


class ReadFileHook(Hook):
    """Read a sandbox text file and store its content in ``item.artifacts``."""

    name = "read_file"

    def __init__(
        self,
        path: str,
        key: str,
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

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        try:
            blob = await client.download_file(self.path)
            item.artifacts[self.key] = blob.decode(self.encoding, errors=self.errors)
        except Exception as exc:
            if self.optional:
                get_logger().warning("read_file %s (key=%r) failed: %s", self.path, self.key, exc)
            else:
                raise


# ─────────────────────────────────────────────────────────────────
# Agent selection
# ─────────────────────────────────────────────────────────────────


class PickAgent(Hook):
    """Weighted-pick one agent from ``agents``; record it on the stage.

    Selection is deterministic on ``item.group_id`` so the same group's
    rollouts always pick the same agent.  The selected agent record includes
    enough template/config fields for downstream hooks to run without a
    separate in-memory context object.
    """

    name = "pick_agent"

    def __init__(self, agents: list[AgentSpec | dict[str, Any]], *, template_root: str):
        if not agents:
            raise ValueError("PickAgent.agents is empty")
        self.agents = [create_object(agent) for agent in agents]
        self.template_root = Path(template_root)

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        group_id = item.group_id or 0
        weights = [max(a.weight, 0.0) for a in self.agents]
        total = sum(weights)
        if total <= 0:
            chosen = self.agents[group_id % len(self.agents)]
        else:
            target = (group_id * 2654435761 % 2**32) / 2**32 * total
            running = 0.0
            chosen = self.agents[-1]
            for agent, w in zip(self.agents, weights):
                running += w
                if target < running:
                    chosen = agent
                    break
        record.agent = SelectedAgentRecord(
            name=chosen.name,
            config=chosen.config,
            install=chosen.install,
            tools=chosen.tools,
            weight=chosen.weight,
            template_root=self.template_root.as_posix(),
        )


# ─────────────────────────────────────────────────────────────────
# Lagent runtime source upload
# ─────────────────────────────────────────────────────────────────


# Minimal __init__ replacements so vendored lagent doesn't eager-import
# optional deps (pandas, etc.) that sandbox images may lack.
_MINIMAL_ACTIONS_INIT = (
    "from .base_action import AsyncActionMixin, BaseAction, tool_api\n"
    "from .parser import BaseParser, JsonParser, TupleParser\n"
)
_MINIMAL_HOOKS_INIT = "from .hook import Hook, RemovableHandle\n"


class InstallLagent(Hook):
    """Ship the lagent library into the sandbox.

    If ``lagent_src_dir`` is configured, uploads the local ``lagent/`` tree
    to ``/tmp/lagent/`` and replaces the eager-import ``__init__.py`` files
    with minimal ones.  Pass ``None`` to skip (sandbox is expected to have
    lagent installed already).
    """

    name = "install_lagent"

    def __init__(self, lagent_src_dir: str | None = None):
        self.lagent_src_dir = lagent_src_dir

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        if self.lagent_src_dir is None:
            return
        blob = await asyncio.to_thread(
            self._tar_lagent_source,
            Path(self.lagent_src_dir) / "lagent",
        )
        await http_upload(
            client,
            "/tmp/_lagent.tar.gz",
            base64.b64encode(blob).decode(),
        )
        await exec_in(
            client,
            "cd /tmp && tar xzf /tmp/_lagent.tar.gz && rm /tmp/_lagent.tar.gz",
        )
        await http_upload(
            client,
            "/tmp/lagent/actions/__init__.py",
            base64.b64encode(_MINIMAL_ACTIONS_INIT.encode()).decode(),
        )
        await http_upload(
            client,
            "/tmp/lagent/hooks/__init__.py",
            base64.b64encode(_MINIMAL_HOOKS_INIT.encode()).decode(),
        )

    def _tar_lagent_source(self, src: Path) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in self._walk_files(src):
                rel = path.relative_to(src).as_posix()
                tar.add(path, arcname=f"lagent/{rel}", recursive=False)
        return buf.getvalue()

    def _walk_files(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        ]


# ─────────────────────────────────────────────────────────────────
# Agent config: upload config.py source (daemon execs it in-sandbox)
# ─────────────────────────────────────────────────────────────────


class UploadAgentConfigSource(Hook):
    """Upload the chosen agent's ``config.py`` source file to ``dst``.

    The lagent daemon exec's this file in the sandbox to build the agent
    dict — so ``os.environ`` lookups inside ``config.py`` resolve against
    the sandbox's own env (populated by the entry's ``env`` field), not the
    host's.

    Agent template lives at ``record.agent.template_root / record.agent.name /``
    (populated by :class:`PickAgent`).
    """

    name = "upload_agent_config_source"

    def __init__(self, dst: str = "/tmp/agent_config.py"):
        self.dst = dst

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        chosen = record.agent
        if chosen is None:
            raise RuntimeError("PickAgent must run before UploadAgentConfigSource")
        cfg_path = Path(chosen.template_root) / chosen.name / chosen.config
        if not cfg_path.is_file():
            raise FileNotFoundError(f"agent config {cfg_path!r} not found")
        await upload_tar_and_extract(client, {self.dst: cfg_path}, "/")


# ─────────────────────────────────────────────────────────────────
# Upload the chosen agent's template tree
# ─────────────────────────────────────────────────────────────────


class UploadChosenAgent(Hook):
    """Ship the chosen agent's template dir into the workspace.

    Reads ``record.agent`` (populated by :class:`PickAgent`);
    set by :class:`PickAgent`); uploads the chosen subtree under
    ``target_dir/<agent_name>/``.
    """

    name = "upload_chosen_agent"

    def __init__(self, *, target_dir: str):
        self.target_dir = target_dir.rstrip("/")

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        chosen = record.agent
        if chosen is None:
            raise RuntimeError("PickAgent must run before UploadChosenAgent")
        src = Path(chosen.template_root) / chosen.name
        if not src.is_dir():
            raise FileNotFoundError(f"agent template {src!r} not found for chosen agent {chosen.name!r}")
        files: dict[str, Path] = {}
        for f in self._walk_files(src):
            rel = f.relative_to(src).as_posix()
            files[f"{self.target_dir}/{chosen.name}/{rel}"] = f
        await upload_tar_and_extract(client, files, "/")

    def _walk_files(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        ]


class LinkChosenAgent(Hook):
    """Link the selected agent template from a shared mount.

    Reads ``record.agent`` populated by :class:`PickAgent`. The selected
    record's ``template_root`` is interpreted as a path visible inside the
    sandbox.
    """

    name = "link_chosen_agent"

    def __init__(
        self,
        *,
        target_dir: str,
        config_dst: str | None = None,
        timeout: int = 60,
    ):
        self.target_dir = target_dir.rstrip("/")
        self.config_dst = config_dst
        self.timeout = timeout

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        chosen = record.agent
        if chosen is None:
            raise RuntimeError("PickAgent must run before LinkChosenAgent")
        template_root = chosen.template_root.rstrip("/")
        agent_src = f"{template_root}/{chosen.name}"
        agent_dst = f"{self.target_dir}/{chosen.name}"
        commands = [
            "set -e",
            self._link_path(agent_src, agent_dst, expect_dir=True),
        ]
        if self.config_dst is not None:
            config_src = f"{agent_src}/{chosen.config}"
            commands.append(self._link_path(config_src, self.config_dst, expect_dir=False))
        await exec_in(client, "\n".join(commands), timeout_sec=self.timeout, raise_on_error=True)

    def _link_path(self, source: str, target: str, *, expect_dir: bool) -> str:
        src = shlex.quote(source)
        dst = shlex.quote(target)
        test = "-d" if expect_dir else "-f"
        kind = "directory" if expect_dir else "file"
        return "\n".join(
            [
                f"if [ ! {test} {src} ]; then echo 'missing shared {kind}: {source}' >&2; exit 2; fi",
                f"mkdir -p $(dirname {dst})",
                f"rm -rf {dst}",
                f"ln -s {src} {dst}",
            ]
        )


class RunAgentInstallDeps(Hook):
    """Run the chosen agent's ``install-deps.sh`` if it has one.

    Looks for ``<workspace>/agent/<name>/install-deps.sh`` after the agent
    template has been uploaded.  No-op if the file doesn't exist in the
    chosen agent's template.
    """

    name = "run_agent_install_deps"

    def __init__(self, *, workspace: str, timeout: int = 600):
        self.workspace = workspace
        self.timeout = timeout

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        chosen = record.agent
        if chosen is None:
            raise RuntimeError("PickAgent must run before RunAgentInstallDeps")
        script = f"{self.workspace}/agent/{chosen.name}/install-deps.sh"
        await exec_in(
            client,
            f'[ -f "{script}" ] && bash "{script}" || true',
            timeout_sec=self.timeout,
            raise_on_error=True,
        )


# ─────────────────────────────────────────────────────────────────
# Judger result parsing
# ─────────────────────────────────────────────────────────────────


class ParseJudgerStdout(Hook):
    """Parse the judger stage stdout into ``record.score`` (and optional
    criteria into ``record.metadata['criteria']``).

    Accepts the two payload shapes the wrappers can emit:
      - dict with a ``total`` key (and optional ``criteria``).
      - ``{"total_score": ..., <criterion>: <number>, …}`` (legacy shape).

    On parse failure: leaves ``record.score`` as ``None`` and marks the stage
    record as failed via ``record.status`` / ``record.error`` so the runner
    surfaces the failure exactly like an entry failure.
    """

    name = "parse_judger_stdout"

    def __init__(self, judger_name: str):
        self.judger_name = judger_name

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        if record.entry_result is None:
            raise RuntimeError("ParseJudgerStdout requires record.entry_result")
        try:
            score, criteria = _parse_stage_stdout(record.entry_result)
        except _ParseError as exc:
            record.score = None
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=self.judger_name,
                category="judger_parse",
                type=type(exc).__name__,
                message=str(exc),
            )
            return
        record.score = score
        if criteria:
            record.metadata["criteria"] = criteria


# ─────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────


class _ParseError(RuntimeError):
    pass


def _parse_stage_stdout(result: StageResult) -> tuple[float, dict[str, dict[str, float]]]:
    """Returns ``(score, criteria)`` or raises :class:`_ParseError`."""
    if result.return_code != 0:
        raise _ParseError(f"return_code={result.return_code}: {result.stderr}")
    stdout = result.stdout.strip()
    if not stdout:
        raise _ParseError("empty stdout")

    json_line = stdout
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break

    try:
        payload = json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise _ParseError(f"json decode: {exc}") from exc

    if isinstance(payload, dict) and "total" in payload:
        total = payload["total"]
        if not isinstance(total, (int, float)) or not 0 <= total <= 1:
            raise _ParseError(f"total out of range: {total!r}")
        criteria = payload.get("criteria") or {}
        return float(total), criteria

    if isinstance(payload, dict) and "total_score" in payload:
        total = float(payload.pop("total_score"))
        criteria = {k: {"score": float(v)} for k, v in payload.items() if isinstance(v, (int, float))}
        return total, criteria

    raise _ParseError(f"unrecognized payload: {json_line[:200]}")
