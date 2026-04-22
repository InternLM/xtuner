"""claw-bench pipeline factory — all-explicit, hook-composed.

Read :func:`claw_pipeline` top-to-bottom: infer-stage pre-hooks and the
one rule_grader judger are both spelled out inline.  Same config level,
no factory hiding.  The only module that knows what a claw-bench task
looks like; core stays bench-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from xtuner.v1.ray.environment.rl_task.hooks import (
    BenchEnv,
    InstallLagent,
    ParseJudgerStdout,
    PickAgent,
    RenderInstruction,
    RunAgentInstallDeps,
    UploadChosenAgent,
    WriteAgentConfig,
)
from xtuner.v1.ray.environment.rl_task.judgers import Judger
from xtuner.v1.ray.environment.rl_task.runner import Runner
from xtuner.v1.ray.environment.rl_task.sandbox import (
    DownloadHook,
    ExecHook,
    SandboxStage,
    UploadHook,
)
from xtuner.v1.ray.environment.rl_task.schemas import AgentSpec, SandboxSpec
from xtuner.v1.ray.environment.rl_task.validator import JudgerValidator


HERE = Path(__file__).resolve().parent
WRAPPERS = HERE / "wrappers"
AGENT_TEMPLATES = HERE / "agents"


# ─────────────────────────────────────────────────────────────────
# Task-dir conventions  (how an upstream claw-bench task looks on disk)
# ─────────────────────────────────────────────────────────────────
#
# Each task_dir holds:
#   task.toml             — metadata (id, domain, level, tags, timeout, …)
#   instruction.md        — natural-language task
#   environment/setup.sh  — optional, runs before agent (gets workspace as $1)
#   environment/data/     — optional, flattened into workspace by pre_entry.sh
#   solution/solve.sh     — oracle (for SolutionScriptInferencer variant)
#   verifier/test_output.py  — pytest grader

CLAW_INSTRUCTION_REWRITES: dict[str, str] = {
    "workspace/": "$TASK_WORKSPACE/",
    "`workspace/": "`$TASK_WORKSPACE/",
}


# ─────────────────────────────────────────────────────────────────
# Sandbox runtime paths  (where things live once running)
# ─────────────────────────────────────────────────────────────────

PATHS = SimpleNamespace(
    wrappers_bench="/tmp/wrappers/claw_bench",
    wrappers_lagent="/tmp/wrappers/lagent",
    agent_config="/tmp/agent_config.json",
    trajectory="/tmp/trajectory.json",
    verifier="/tmp/verifier",
)


# Entry commands.  ``cd /`` before the script is upstream convention — they
# run with ``cwd=task_dir`` so that relative ``workspace/<file>`` paths land
# under ``/workspace/``.  Equivalent here: the parent of $TASK_WORKSPACE is /,
# so cd / makes ``workspace/foo`` resolve to ``/workspace/foo``.
AGENT_ENTRY = (
    f"bash {PATHS.wrappers_bench}/pre_entry.sh && "
    f"bash {PATHS.wrappers_lagent}/lagent_entry.sh "
    f"--config {PATHS.agent_config} "
    f"--instruction-file $TASK_INSTRUCTION "
    f"--response-out /tmp/agent_response.txt "
    f"--trajectory-out {PATHS.trajectory}"
)


SOLUTION_ENTRY = (
    f"bash {PATHS.wrappers_bench}/pre_entry.sh && "
    f"cd / && bash $TASK_WORKSPACE/solution/solve.sh $TASK_WORKSPACE"
)


# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────

DEFAULT_AGENTS: list[AgentSpec] = [
    AgentSpec(
        name="internclaw",
        config="config.py",
        install="install-deps.sh",
        tools="tools",
        weight=1.0,
    ),
]

DEFAULT_SANDBOX = SandboxSpec(
    image="ubuntu2404-v1", ttl_seconds=1800, workspace_path="/workspace",
)


# ─────────────────────────────────────────────────────────────────
# Judger (claw-bench only ever runs one pytest grader per task)
# ─────────────────────────────────────────────────────────────────


def _rule_grader(ws: str) -> Judger:
    """Claw-bench's single pytest-based judger.  Reads ``verifier/test_output.py``
    from the task dir, ships it to the infer sandbox, runs it via the shared
    pytest wrapper, parses CTRF JSON on stdout.

    Every piece of the SandboxStage is spelled out inline — the env vars
    are literal, the pre-hook upload mapping is literal, no factory helper.
    """
    root = f"{PATHS.verifier}/rule_grader"
    return Judger(
        name="rule_grader",
        weight=1.0,
        sandbox="shared",     # reuse the infer sandbox (no extra provisioning)
        stage=SandboxStage(
            pre=[
                # Ship verifier/ tree to /tmp/verifier/rule_grader/.
                UploadHook([
                    {"base": "verifier", "source": "**/*", "target": f"{root}/"},
                ]),
            ],
            entry=f"bash {PATHS.wrappers_bench}/pytest_ctrf.sh",
            env={
                "JUDGER_NAME": "rule_grader",
                "TASK_WORKSPACE": ws,
                "TASK_JUDGER_DIR": root,
                "PYTEST_TARGET": f"{root}/test_output.py",
            },
            timeout=300,
            post=[ParseJudgerStdout("rule_grader")],
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Pipeline factories
# ─────────────────────────────────────────────────────────────────


def claw_pipeline(
    *,
    sandbox: SandboxSpec = DEFAULT_SANDBOX,
    agents: list[AgentSpec] = DEFAULT_AGENTS,
) -> Runner:
    """Build a Runner for a claw-bench task.  Infer stage + rule_grader
    judger both read top-to-bottom.
    """
    ws = sandbox.workspace_path

    infer = SandboxStage(
        sandbox=sandbox,
        pre=[
            # 1. Ship wrapper scripts (bench + agent-framework) into /tmp/wrappers/.
            UploadHook([
                {"base": str(WRAPPERS / "claw_bench"),
                 "source": "*", "target": PATHS.wrappers_bench + "/", "flatten": True},
                {"base": str(WRAPPERS / "lagent"),
                 "source": "*", "target": PATHS.wrappers_lagent + "/", "flatten": True},
            ]),
            # 2. Install lagent library + /tmp/lagent-py python wrapper.
            InstallLagent(),
            # 3. Weighted-pick one agent; record choice + template_root in ctx.
            PickAgent(agents=agents, template_root=str(AGENT_TEMPLATES)),
            # 4. Mirror the task tree into $TASK_WORKSPACE.  Exclude oracle /
            #    judger dirs so the agent can't see them; exclude task.toml
            #    since it's metadata the agent has no use for.
            UploadHook([
                {"source": "**/*", "target": f"{ws}/",
                 "exclude": [
                     "task.toml", "task.py",
                     "reference/**", "verifier/**", "solution/**", "agent/**",
                 ]},
            ]),
            # 5. Overlay chosen agent's template at $TASK_WORKSPACE/agent/<name>/.
            UploadChosenAgent(target_dir=f"{ws}/agent/"),
            # 6. Ensure workspace dir exists (mirror doesn't create empty dirs).
            ExecHook(f"mkdir -p {ws}"),
            # 7. Render instruction.md: `workspace/` → abs path + {{KEY}} → env.
            # RenderInstruction(rewrites=CLAW_INSTRUCTION_REWRITES),
            # 8. Exec chosen agent's config.py on host → upload resulting JSON.
            WriteAgentConfig(dst=PATHS.agent_config),
            # 9. Run install-deps.sh if the chosen agent template has one.
            RunAgentInstallDeps(workspace=ws),
        ],
        entry=AGENT_ENTRY,
        env=BenchEnv(
            workspace=ws,
            # Upstream setup.sh/solve.sh conventions.
            extras={"WORKSPACE": ws, "CLAW_WORKSPACE": ws},
        ),
        timeout=1800,
        post=[DownloadHook(["/workspace", "/tmp/agent_response.txt"])],
    )

    return Runner(
        infer=infer,
        validate=JudgerValidator([_rule_grader(ws)]),
    )


def claw_solution_pipeline() -> Runner:
    """Variant: run ``solution/solve.sh`` instead of an LLM rollout."""
    sandbox = SandboxSpec(
        image="ubuntu2404-v2", ttl_seconds=600, workspace_path="/workspace",
    )
    ws = sandbox.workspace_path

    infer = SandboxStage(
        sandbox=sandbox,
        pre=[
            ExecHook(f"mkdir -p {ws}"),
            UploadHook([
                {"base": str(WRAPPERS / "claw_bench"),
                 "source": "*", "target": PATHS.wrappers_bench + "/", "flatten": True},
                {"base": str(WRAPPERS / "lagent"),
                 "source": "*", "target": PATHS.wrappers_lagent + "/", "flatten": True},
            ]),
            # Oracle may consult reference/ and solution/ — those stay in the mirror.
            UploadHook([
                {"source": "**/*", "target": f"{ws}/",
                 "exclude": ["task.toml", "task.py", "verifier/**", "agent/**"]},
            ]),
        ],
        entry=SOLUTION_ENTRY,
        env=BenchEnv(
            workspace=ws,
            extras={"WORKSPACE": ws, "CLAW_WORKSPACE": ws},
        ),
        timeout=600,
        post=[DownloadHook(["/workspace"])],
    )

    return Runner(
        infer=infer,
        validate=JudgerValidator([_rule_grader(ws)]),
    )
