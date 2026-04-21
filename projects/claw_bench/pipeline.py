"""claw-bench pipeline factory — all-explicit, hook-composed.

Read :func:`claw_pipeline` top-to-bottom: every pre-hook is a named class
with declared inputs, no lambdas, no hidden callables.  The only module
that knows what a claw-bench task looks like; core stays bench-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hooks import (
    BenchEnv,
    CheckOutputs,
    InstallLagent,
    PickAgent,
    RenderInstruction,
    RunAgentInstallDeps,
    UploadChosenAgent,
    WriteAgentConfig,
)
from judgers import Judger
from runner import Runner
from sandbox import ExecHook, SandboxStage, UploadHook
from schemas import AgentSpec, SandboxSpec
from validator import JudgerValidator


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
    reference="/tmp/reference",
    verifier="/tmp/verifier",
)


# ─────────────────────────────────────────────────────────────────
# Entry commands
# ─────────────────────────────────────────────────────────────────

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
    f"bash $TASK_WORKSPACE/solution/solve.sh $TASK_WORKSPACE"
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
# Pipeline factories
# ─────────────────────────────────────────────────────────────────


def claw_pipeline(
    judgers: list[Judger],
    *,
    sandbox: SandboxSpec = DEFAULT_SANDBOX,
    agents: list[AgentSpec] = DEFAULT_AGENTS,
) -> Runner:
    """Build a Runner for a claw-bench task.  Infer stage reads top-to-bottom."""
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
            RenderInstruction(rewrites=CLAW_INSTRUCTION_REWRITES),
            # 8. Exec chosen agent's config.py on host → upload resulting JSON.
            WriteAgentConfig(dst=PATHS.agent_config),
            # 9. Run install-deps.sh if the chosen agent template has one.
            RunAgentInstallDeps(workspace=ws),
        ],
        entry=AGENT_ENTRY,
        env=BenchEnv(workspace=ws),
        timeout=1800,
        pull=["/workspace", "/tmp/agent_response.txt"],
        post=[CheckOutputs()],
    )

    return Runner(
        infer=infer,
        validate=JudgerValidator(judgers, reference_path=PATHS.reference),
    )


def claw_solution_pipeline(judgers: list[Judger]) -> Runner:
    """Variant: run ``solution/solve.sh`` instead of an LLM rollout."""
    sandbox = SandboxSpec(
        image="ubuntu2404-v1", ttl_seconds=600, workspace_path="/workspace",
    )
    ws = sandbox.workspace_path

    infer = SandboxStage(
        sandbox=sandbox,
        pre=[
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
            ExecHook(f"mkdir -p {ws}"),
        ],
        entry=SOLUTION_ENTRY,
        env=BenchEnv(workspace=ws),
        timeout=600,
        pull=["/workspace"],
        post=[CheckOutputs()],
    )

    return Runner(
        infer=infer,
        validate=JudgerValidator(judgers, reference_path=PATHS.reference),
    )
