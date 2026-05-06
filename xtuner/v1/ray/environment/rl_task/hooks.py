"""Specialized hooks — agent setup, instruction rendering, output/judger
plumbing.

Each class here is a :class:`sandbox.Hook` subclass with a clear purpose
you can infer from its name.  Primitive hooks (Upload/Exec/Download) cover
the common case; specialized hooks encapsulate logic that's awkward to
express as a plain lambda (e.g. running agent config exec on host).

Context dict keys this module reads/writes::

    ctx["task_root"]        : Path  (set by Runner)
    ctx["data"]             : TaskData  (set by Runner)
    ctx["workspace"]        : str  (set by Runner from stage.sandbox.workspace_path)
    ctx["uid"]              : dict  (set by Runner)
    ctx["runtime"]          : dict  (lagent_src_dir, llm_base_url, llm_api_key)
    ctx["chosen_agent"]     : AgentSpec  (set by PickAgent)
    ctx["judger_result"]    : JudgerResult  (set by ParseJudgerStdout)
    ctx["result"]           : StageResult  (set by SandboxStage after entry)
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from xtuner.v1.ray.environment.rl_task import bundle
from xtuner.v1.ray.environment.rl_task.sandbox import (
    Hook,
    StageResult,
    exec_in,
    http_upload,
    tar_dir,
    upload_tar_and_extract,
    walk_files,
)
from xtuner.v1.ray.environment.rl_task.schemas import AgentSpec, CriterionScore, JudgerResult
from xtuner.v1.utils import get_logger


# ─────────────────────────────────────────────────────────────────
# Agent selection
# ─────────────────────────────────────────────────────────────────


class PickAgent(Hook):
    """Weighted-pick one agent from ``agents``; record it in ``ctx``.

    Selection is deterministic on ``ctx["uid"]["root_id"]`` so the same
    rollout always picks the same agent.  Also stores ``template_root`` in
    ``ctx["agent_template_root"]`` so downstream hooks
    (:class:`UploadChosenAgent`, :class:`UploadAgentConfigSource`,
    :class:`RunAgentInstallDeps`) know where the agent's files live on
    the host.
    """

    name = "pick_agent"

    def __init__(self, agents: list[AgentSpec], *, template_root: str):
        if not agents:
            raise ValueError("PickAgent.agents is empty")
        self.agents = agents
        self.template_root = Path(template_root)

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        group_id = (ctx.get("uid") or {}).get("root_id", 0)
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
        ctx["chosen_agent"] = chosen
        ctx["agent_template_root"] = self.template_root


# ─────────────────────────────────────────────────────────────────
# Lagent runtime (library + python wrapper + minimal inits)
# ─────────────────────────────────────────────────────────────────


_LAGENT_PY_PATH = "/tmp/lagent-py"
_LAGENT_PY_WRAPPER = """#!/bin/bash
PYTHONPATH="${TASK_WORKSPACE:-/workspace}:/tmp:${PYTHONPATH:-}" \
exec /mnt/llm-ai-infra/miniconda3/envs/train/bin/python "$@"
"""

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
    """Ship the lagent library + a python wrapper into the sandbox.

    Reads ``ctx["runtime"]["lagent_src_dir"]`` — if set, uploads the local
    ``lagent/`` tree to ``/tmp/lagent/`` and replaces the eager-import
    ``__init__.py`` files with minimal ones.  Always writes the
    ``/tmp/lagent-py`` bash wrapper.
    """

    name = "install_lagent"

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        lagent_src = (ctx.get("runtime") or {}).get("lagent_src_dir")
        if lagent_src is not None:
            blob = await asyncio.to_thread(
                tar_dir,
                Path(lagent_src) / "lagent",
                "lagent",
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
        await http_upload(
            client,
            _LAGENT_PY_PATH,
            base64.b64encode(_LAGENT_PY_WRAPPER.encode()).decode(),
        )
        await exec_in(client, f"chmod +x {_LAGENT_PY_PATH}")


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
         ``ctx["env_vars_for_instruction"]``, if set).

    Writes the rendered path back into ``ctx["instruction_rendered_path"]``
    so a following :class:`UploadHook` can ship it.
    """

    name = "render_instruction"

    def __init__(self, rewrites: dict[str, str] | None = None):
        self.rewrites = dict(rewrites or {})

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        data = ctx["data"]
        workspace = ctx["workspace"]
        task_root = ctx["task_root"]

        src = task_root / data.instruction
        if not src.exists():
            return

        text = src.read_text(encoding="utf-8")
        resolved = {
            needle: bundle.rewrite_text(repl, {"$TASK_WORKSPACE": workspace}) for needle, repl in self.rewrites.items()
        }
        text = bundle.rewrite_text(text, resolved)
        env_for_placeholders = ctx.get("env_vars_for_instruction", {})
        text = bundle.rewrite_text(
            text,
            {"{{" + k + "}}": v for k, v in env_for_placeholders.items()},
        )

        tmp = Path("/tmp") / f".rendered_{src.name}"
        tmp.write_text(text, encoding="utf-8")
        # Upload the rendered file over the mirror-version.
        sandbox_path = f"{workspace}/{data.instruction}"
        await upload_tar_and_extract(client, {sandbox_path: tmp}, "/")


# ─────────────────────────────────────────────────────────────────
# Agent config: upload config.py source (daemon execs it in-sandbox)
# ─────────────────────────────────────────────────────────────────


class UploadAgentConfigSource(Hook):
    """Upload the chosen agent's ``config.py`` source file to ``dst``.

    The lagent daemon exec's this file in the sandbox to build the agent
    dict — so ``os.environ`` lookups inside ``config.py`` resolve against
    the sandbox's own env (populated by :class:`BenchEnv`), not the host's.

    Agent template lives at ``ctx["agent_template_root"] / chosen.name /``
    (populated by :class:`PickAgent`).
    """

    name = "upload_agent_config_source"

    def __init__(self, dst: str = "/tmp/agent_config.py"):
        self.dst = dst

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        chosen: AgentSpec = ctx["chosen_agent"]
        template_root: Path = ctx["agent_template_root"]
        cfg_path = template_root / chosen.name / chosen.config
        if not cfg_path.is_file():
            raise FileNotFoundError(f"agent config {cfg_path!r} not found")
        await upload_tar_and_extract(client, {self.dst: cfg_path}, "/")


# ─────────────────────────────────────────────────────────────────
# Upload the chosen agent's template tree
# ─────────────────────────────────────────────────────────────────


class UploadChosenAgent(Hook):
    """Ship the chosen agent's template dir into the workspace.

    Reads ``ctx["agent_template_root"]`` and ``ctx["chosen_agent"]`` (both
    set by :class:`PickAgent`); uploads the chosen subtree under
    ``target_dir/<agent_name>/``.
    """

    name = "upload_chosen_agent"

    def __init__(self, *, target_dir: str):
        self.target_dir = target_dir.rstrip("/")

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        chosen: AgentSpec = ctx["chosen_agent"]
        template_root: Path = ctx["agent_template_root"]
        src = template_root / chosen.name
        if not src.is_dir():
            raise FileNotFoundError(f"agent template {src!r} not found for chosen agent {chosen.name!r}")
        files: dict[str, Path] = {}
        for f in walk_files(src):
            rel = f.relative_to(src).as_posix()
            files[f"{self.target_dir}/{chosen.name}/{rel}"] = f
        await upload_tar_and_extract(client, files, "/")


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

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        chosen: AgentSpec = ctx["chosen_agent"]
        script = f"{self.workspace}/agent/{chosen.name}/install-deps.sh"
        await exec_in(
            client,
            f'[ -f "{script}" ] && bash "{script}" || true',
            timeout_sec=self.timeout,
            raise_on_error=True,
        )


# ─────────────────────────────────────────────────────────────────
# Env-var builder (replaces lambda env= on SandboxStage)
# ─────────────────────────────────────────────────────────────────


class BenchEnv:
    """Callable that produces the infer stage's env vars from ctx.

    Exports only what wrappers + agent config actually read — no
    speculative vars.  ``extras`` lets a bench-specific pipeline inject
    additional literal vars (e.g. upstream-convention aliases like
    ``WORKSPACE``, ``CLAW_WORKSPACE``) without subclassing.

    Also stores the map in ``ctx["env_vars_for_instruction"]`` so
    :class:`RenderInstruction` can substitute ``{{KEY}}`` placeholders.

    Pass an instance to ``SandboxStage(env=BenchEnv(...))``.
    """

    def __init__(self, *, workspace: str, extras: dict[str, str] | None = None):
        self.workspace = workspace
        self.extras = dict(extras or {})

    def __call__(self, ctx: dict[str, Any]) -> dict[str, str]:
        data = ctx["data"]
        runtime = ctx.get("runtime") or {}
        env = {
            "TASK_WORKSPACE": self.workspace,
            "TASK_INSTRUCTION": f"{self.workspace}/{data.instruction}",
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
        ctx["env_vars_for_instruction"] = env
        return env


# ─────────────────────────────────────────────────────────────────
# Daemon log retrieval (post-hook)
# ─────────────────────────────────────────────────────────────────


class DumpDaemonLogOnFailure(Hook):
    """Post-hook: pull ``/tmp/agent_daemon.log`` and log its tail on failure.

    Two triggers:
      - Stage's entry returned non-zero (``rc != 0``) — usual sandbox error.
      - Silent-pass: ``rc == 0`` but the pulled ``message_key`` contents show
        the last ``policy_agent.messages`` entry lacks ``raw_content_ids``
        (LLM call somehow produced no token ids — typically an exception
        swallowed by the agent layer).  Disable by passing ``message_key=None``.

    Always stores the full daemon log at ``ctx["pulled"][key]`` for
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

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        try:
            blob = await client.download_file(self.path)
        except Exception as exc:
            get_logger().warning(f"could not download daemon log at {self.path}: {exc}")
            return
        text = blob.decode(errors="replace")
        ctx.setdefault("pulled", {})[self.key] = text

        result = ctx.get("result")
        rc = getattr(result, "return_code", 0) if result else 0

        should_dump = rc != 0
        reason = f"rc={rc}"
        if not should_dump and self.message_key:
            raw = (ctx.get("pulled") or {}).get(self.message_key) or ""
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
    ``ctx['judger_result']``.

    Accepts the two payload shapes the wrappers can emit:
      - JudgerResult-shaped dict (has ``total`` key).
      - ``{"total_score": ..., <criterion>: <number>, …}`` (legacy shape).
    """

    name = "parse_judger_stdout"

    def __init__(self, judger_name: str):
        self.judger_name = judger_name

    async def __call__(self, client: Any, ctx: dict[str, Any]) -> None:
        ctx["judger_result"] = _parse_stage_stdout(self.judger_name, ctx["result"])


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
        criteria = {k: CriterionScore(score=float(v)) for k, v in payload.items() if isinstance(v, (int, float))}
        return JudgerResult(judger_name=name, total=total, criteria=criteria)

    return JudgerResult(
        judger_name=name,
        total=0.0,
        error=f"unrecognized payload: {json_line[:200]}",
    )
