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
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lagent.utils import create_object
from pydantic import ValidationError

from xtuner.v1.ray.environment.rl_task.sandbox import (
    Hook,
    exec_in,
    http_upload,
    upload_tar_and_extract,
)
from xtuner.v1.ray.environment.rl_task.schemas import (
    AgentRolloutItem,
    AgentSpec,
    JudgerResult,
    SelectedAgentRecord,
    StageRecord,
    StageResult,
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

    Selection is deterministic on ``item.uid["root_id"]`` so the same
    rollout always picks the same agent.  The selected agent record includes
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
        group_id = (item.uid or {}).get("root_id", 0)
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
    "from .action_executor import ActionExecutor, AsyncActionExecutor\n"
    "from .base_action import AsyncActionMixin, BaseAction, tool_api\n"
    "from .builtin_actions import FinishAction, InvalidAction, NoAction\n"
    "from .parser import BaseParser, JsonParser, TupleParser\n"
)
_MINIMAL_HOOKS_INIT = "from .hook import Hook, RemovableHandle\n"


class InstallLagent(Hook):
    """Ship the lagent library into the sandbox.

    Reads ``record.runtime["lagent_src_dir"]`` — if set, uploads the local
    ``lagent/`` tree to ``/tmp/lagent/`` and replaces the eager-import
    ``__init__.py`` files with minimal ones.
    """

    name = "install_lagent"

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        lagent_src = record.runtime.get("lagent_src_dir")
        if lagent_src is not None:
            blob = await asyncio.to_thread(
                self._tar_lagent_source,
                Path(lagent_src) / "lagent",
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
# Instruction render: bench-specific string rewrites, in place
# ─────────────────────────────────────────────────────────────────


class RenderInstruction(Hook):
    """Rewrite the instruction file + upload the rendered version.

    Works via a /tmp scratch file and an UploadHook.  Two substitution
    passes:
      1. ``rewrites`` (bench-supplied literal map, with ``$TASK_WORKSPACE``
         in values pre-resolved to the absolute path).
      2. ``{{KEY}}`` → the corresponding env var (from
         ``record.env_vars``, if set).

    Writes the rendered path back into ``record.metadata["instruction_rendered_path"]``
    so a following :class:`UploadHook` can ship it.
    """

    name = "render_instruction"

    def __init__(self, rewrites: dict[str, str] | None = None):
        self.rewrites = dict(rewrites or {})

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        workspace = record.workspace
        if not workspace:
            raise ValueError("StageRecord.workspace is required by RenderInstruction")
        task_root = item.task_root
        if task_root is None:
            raise ValueError("AgentRolloutItem.task_root is required by RenderInstruction")

        src = task_root / item.instruction
        if not src.exists():
            return

        text = src.read_text(encoding="utf-8")
        resolved = {needle: self._rewrite_text(repl, {"$TASK_WORKSPACE": workspace}) for needle, repl in self.rewrites.items()}
        text = self._rewrite_text(text, resolved)
        env_for_placeholders = record.env_vars
        text = self._rewrite_text(
            text,
            {"{{" + k + "}}": v for k, v in env_for_placeholders.items()},
        )

        tmp = Path("/tmp") / f".rendered_{src.name}"
        tmp.write_text(text, encoding="utf-8")
        # Upload the rendered file over the mirror-version.
        sandbox_path = f"{workspace}/{item.instruction}"
        await upload_tar_and_extract(client, {sandbox_path: tmp}, "/")
        record.metadata["instruction_rendered_path"] = sandbox_path

    def _rewrite_text(self, text: str, substitutions: dict[str, str]) -> str:
        for needle, replacement in substitutions.items():
            text = text.replace(needle, replacement)
        return text


# ─────────────────────────────────────────────────────────────────
# Agent config: upload config.py source (daemon execs it in-sandbox)
# ─────────────────────────────────────────────────────────────────


class UploadAgentConfigSource(Hook):
    """Upload the chosen agent's ``config.py`` source file to ``dst``.

    The lagent daemon exec's this file in the sandbox to build the agent
    dict — so ``os.environ`` lookups inside ``config.py`` resolve against
    the sandbox's own env (populated by :class:`BenchEnv`), not the host's.

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
# Validate workspace setup
# ─────────────────────────────────────────────────────────────────


class CopyInferWorkspace(Hook):
    """Validator hook: copy the infer workspace into an isolated judger sandbox.

    ``JudgerValidator`` exposes the source client/path to isolated judger
    hooks through runtime-only fields on ``StageRecord``.  Putting the copy in
    a hook keeps the validate data dependency visible in config:

        on_isolated_pre=[dict(type=CopyInferWorkspace)]
    """

    name = "copy_infer_workspace"

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        infer_client = record.runtime.get("infer_client")
        infer_workspace = record.runtime.get("infer_workspace")
        target_workspace = record.runtime.get("target_workspace") or record.workspace
        if infer_client is None:
            raise RuntimeError("CopyInferWorkspace requires record.runtime['infer_client']")
        if not infer_workspace:
            raise RuntimeError("CopyInferWorkspace requires record.runtime['infer_workspace']")
        if not target_workspace:
            raise RuntimeError("CopyInferWorkspace requires record.workspace or record.runtime['target_workspace']")

        suffix = uuid.uuid4().hex[:12]
        infer_tmp = f"/tmp/_infer_ws_{suffix}.tar.gz"
        target_tmp = f"/tmp/_target_ws_{suffix}.tar.gz"
        await exec_in(client, f'mkdir -p "{target_workspace}"')
        try:
            await exec_in(infer_client, f'cd "{infer_workspace}" && tar czf {infer_tmp} .')
            blob = await infer_client.download_file(infer_tmp)
            await http_upload(client, target_tmp, base64.b64encode(blob).decode())
            await exec_in(
                client,
                f'cd "{target_workspace}" && tar xzf {target_tmp} && rm -f {target_tmp}',
                raise_on_error=False,
            )
        finally:
            try:
                await exec_in(infer_client, f"rm -f {infer_tmp}", raise_on_error=False)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────
# Env-var builder
# ─────────────────────────────────────────────────────────────────


class BenchEnv:
    """Build infer stage env vars from an item/record.

    Exports only what wrappers + agent config actually read — no
    speculative vars.  ``extras`` lets a bench-specific pipeline inject
    additional literal vars (e.g. upstream-convention aliases like
    ``WORKSPACE``, ``CLAW_WORKSPACE``) without subclassing.

    Also stores the map in ``record.env_vars`` so
    :class:`RenderInstruction` can substitute ``{{KEY}}`` placeholders.

    Pass an instance to ``SandboxStage(env=BenchEnv(...))``.  The explicit
    interface is ``build(item, record)``; arbitrary callable stage fields are
    intentionally not supported.
    """

    def __init__(self, *, workspace: str, extras: dict[str, str] | None = None):
        self.workspace = workspace
        self.extras = dict(extras or {})

    def build(self, item: AgentRolloutItem, record: StageRecord) -> dict[str, str]:
        runtime = record.runtime
        env = {
            "TASK_WORKSPACE": self.workspace,
            "TASK_INSTRUCTION": f"{self.workspace}/{item.instruction}",
        }
        for env_key, runtime_key in (
            ("RL_LLM_MODEL", "llm_model"),
            ("RL_LLM_BASE_URL", "llm_base_url"),
            ("RL_LLM_API_KEY", "llm_api_key"),
        ):
            val = runtime.get(runtime_key)
            if val:
                env[env_key] = val
        env.update(self.extras)
        record.env_vars = env
        return env


# ─────────────────────────────────────────────────────────────────
# Daemon log retrieval (post-hook)
# ─────────────────────────────────────────────────────────────────


class DumpDaemonLogOnFailure(Hook):
    """Post-hook: pull ``/tmp/agent_daemon.log`` and log its tail on failure.

    Two triggers:
      - Stage's entry returned non-zero (``rc != 0``) — usual sandbox error.
      - Silent-pass: ``rc == 0`` but the collected ``message_key`` contents show
        the last ``policy_agent.messages`` entry lacks ``raw_content_ids``
        (LLM call somehow produced no token ids — typically an exception
        swallowed by the agent layer).  Disable by passing ``message_key=None``.

    Always stores the full daemon log at ``item.artifacts[key]`` for
    downstream consumers regardless of whether we log.
    """

    name = "dump_daemon_log_on_failure"

    def __init__(
        self,
        path: str = "/tmp/agent_daemon.log",
        *,
        tail_lines: int = 500,
        key: str = "daemon_log",
        message_key: str | None = "message",
    ):
        self.path = path
        self.tail_lines = tail_lines
        self.key = key
        self.message_key = message_key

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        try:
            blob = await client.download_file(self.path)
        except Exception as exc:
            get_logger().warning(f"could not download daemon log at {self.path}: {exc}")
            return
        text = blob.decode(errors="replace")
        item.artifacts[self.key] = text

        result = record.entry_result
        rc = getattr(result, "return_code", 0) if result else 0

        should_dump = rc != 0
        reason = f"rc={rc}"
        if not should_dump and self.message_key:
            raw = item.artifacts.get(self.message_key) or ""
            try:
                msgs = json.loads(raw).get("policy_agent.messages", []) if raw else []
            except Exception:
                msgs = []
            last = msgs[-1] if msgs else {}
            required = ("raw_content", "raw_content_ids", "raw_content_logprobs")
            missing = [k for k in required if not last.get(k)]
            if missing:
                should_dump = True
                reason = f"silent-pass (last message missing {missing})"

        if should_dump:
            lines = text.splitlines()
            tail = "\n".join(lines[-self.tail_lines :]) if len(lines) > self.tail_lines else text
            get_logger().error(f"daemon log tail [{reason}] ({self.path}):\n{tail}")


# ─────────────────────────────────────────────────────────────────
# Judger result parsing
# ─────────────────────────────────────────────────────────────────


class ParseJudgerStdout(Hook):
    """Turn the stage's stdout into a :class:`JudgerResult` at
    ``record.result``.

    Accepts the two payload shapes the wrappers can emit:
      - JudgerResult-shaped dict (has ``total`` key).
      - ``{"total_score": ..., <criterion>: <number>, …}`` (legacy shape).
    """

    name = "parse_judger_stdout"

    def __init__(self, judger_name: str):
        self.judger_name = judger_name

    async def __call__(self, client: Any, item: AgentRolloutItem, record: StageRecord) -> None:
        if record.entry_result is None:
            raise RuntimeError("ParseJudgerStdout requires record.entry_result")
        record.result = _parse_stage_stdout(self.judger_name, record.entry_result)


# ─────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────


def _parse_stage_stdout(name: str, result: StageResult) -> JudgerResult:
    if result.return_code != 0:
        return JudgerResult(
            judger_name=name,
            total=0.0,
            error=f"return_code={result.return_code}: {result.stderr[:500]}",
        )
    stdout = result.stdout.strip()
    if not stdout:
        return JudgerResult(judger_name=name, total=0.0, error="empty stdout")

    json_line = stdout
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break

    try:
        payload = json.loads(json_line)
    except json.JSONDecodeError as exc:
        return JudgerResult(judger_name=name, total=0.0, error=f"json decode: {exc}")

    if isinstance(payload, dict) and "total" in payload:
        payload.setdefault("judger_name", name)
        try:
            return JudgerResult(**payload)
        except ValidationError as exc:
            return JudgerResult(judger_name=name, total=0.0, error=f"schema: {exc}")

    if isinstance(payload, dict) and "total_score" in payload:
        total = float(payload.pop("total_score"))
        criteria = {k: {"score": float(v)} for k, v in payload.items() if isinstance(v, (int, float))}
        return JudgerResult(judger_name=name, total=total, criteria=criteria)

    return JudgerResult(
        judger_name=name,
        total=0.0,
        error=f"unrecognized payload: {json_line[:200]}",
    )
