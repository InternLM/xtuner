"""tb2-eval pipeline factory.

Layout of a tb2-eval task (flat, no category subdirs)::

    <task-name>/
        task.toml              — metadata + per-task docker_image
        instruction.md         — natural-language task
        environment/Dockerfile — baked into the per-task image
        environment/files/     — data files the Dockerfile COPYs to /app/
        tests/test.sh          — bench-provided verifier entrypoint
        tests/test_outputs.py  — pytest module
        tests/test_requirements.txt

Unlike tb2-rl (which uses a fixed ``t-data-processing-v1`` image for all
tasks), each eval task uses its own pre-built docker image specified in
``task.toml [environment] docker_image``.  The image is passed at runtime
via the sandbox_spec in ``extra_info`` (see ``tb2_eval_tokenize_fn.py``).

The ``DEFAULT_SANDBOX`` below uses a placeholder image; in practice the
per-task image always overrides it via ``sandbox_spec`` in ``extra_info``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from xtuner.v1.ray.environment.rl_task.hooks import (
    BenchEnv,
    InstallLagent,
    ParseJudgerStdout,
    PickAgent,
    RunAgentInstallDeps,
    UploadChosenAgent,
    WriteAgentConfig,
)
from xtuner.v1.ray.environment.rl_task.judgers import Judger
from xtuner.v1.ray.environment.rl_task.runner import Runner
from xtuner.v1.ray.environment.rl_task.sandbox import (
    DownloadHook,
    ExecHook,
    ReadFileHook,
    SandboxStage,
    UploadHook,
)
from xtuner.v1.ray.environment.rl_task.schemas import AgentSpec, SandboxSpec
from xtuner.v1.ray.environment.rl_task.validator import JudgerValidator

HERE = Path(__file__).resolve().parent
WRAPPERS = HERE / "wrappers"
AGENT_TEMPLATES = HERE / "agents"


# ─────────────────────────────────────────────────────────────────
# Sandbox runtime paths
# ─────────────────────────────────────────────────────────────────

PATHS = SimpleNamespace(
    wrappers_bench="/tmp/wrappers/tb2_eval",
    wrappers_lagent="/tmp/wrappers/lagent",
    agent_config="/tmp/agent_config.json",
    trajectory="/tmp/trajectory.json",
    message="/tmp/message.json",
    tests="/tests",
)


AGENT_ENTRY = (
    f"bash {PATHS.wrappers_bench}/pre_entry.sh && "
    f"bash {PATHS.wrappers_lagent}/lagent_entry.sh "
    f"--config {PATHS.agent_config} "
    f"--instruction-file $TASK_INSTRUCTION "
    f"--response-out /tmp/agent_response.txt "
    f"--trajectory-out {PATHS.trajectory} "
    f"--message-out {PATHS.message}"
)


# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────

DEFAULT_AGENTS: list[AgentSpec] = [
    AgentSpec(
        name="interndp",
        config="config.py",
        install="install-deps.sh",
        tools="tools",
        weight=1.0,
    ),
]

# Placeholder image — each task overrides this via sandbox_spec in extra_info.
DEFAULT_SANDBOX = SandboxSpec(image="tb2-eval-placeholder", ttl_seconds=1800, workspace_path="/app")


# ─────────────────────────────────────────────────────────────────
# Judger — runs the bench's own tests/test.sh and parses CTRF JSON
# ─────────────────────────────────────────────────────────────────


def _rule_grader(ws: str) -> Judger:
    """Bench-provided ``tests/test.sh`` grader.

    Args:
        ws (str): Agent workspace absolute path (exported as ``$TASK_WORKSPACE``).

    Returns:
        Judger: A shared-sandbox judger that uploads ``tests/*`` to ``/tests/``,
        runs ``bash /tests/test.sh`` (bench convention — writes CTRF JSON to
        ``/logs/verifier/ctrf.json``), and emits a ``JudgerResult`` line on
        stdout via the wrapper parser.
    """
    return Judger(
        name="rule_grader",
        weight=1.0,
        sandbox="shared",
        stage=SandboxStage(
            pre=[
                UploadHook(
                    [
                        {"base": "tests", "source": "**/*", "target": f"{PATHS.tests}/"},
                    ]
                ),
                ExecHook(f"chmod +x {PATHS.tests}/test.sh || true", optional=True),
            ],
            entry=f"bash {PATHS.wrappers_bench}/run_tests.sh",
            env={
                "JUDGER_NAME": "rule_grader",
                "TASK_WORKSPACE": ws,
                "TESTS_DIR": PATHS.tests,
                "WRAPPER_DIR": PATHS.wrappers_bench,
            },
            timeout=900,
            post=[ParseJudgerStdout("rule_grader")],
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────


def tb2_eval_pipeline(
    *,
    sandbox: SandboxSpec = DEFAULT_SANDBOX,
    agents: list[AgentSpec] = DEFAULT_AGENTS,
) -> Runner:
    """Build a Runner for a tb2-eval task.

    Args:
        sandbox (SandboxSpec): Sandbox image + workspace config.  In practice
            the per-task image from ``task.toml`` is injected at runtime via
            ``sandbox_spec`` in ``extra_info``.
        agents (list[AgentSpec]): Agent candidates.  Defaults to the single
            ``interndp`` template shipped under ``agents/interndp/``.

    Returns:
        Runner: Infer stage + rule_grader judger wired together.
    """
    ws = sandbox.workspace_path

    infer = SandboxStage(
        sandbox=sandbox,
        pre=[
            # 1. Ship wrapper scripts (bench + lagent) into /tmp/wrappers/.
            UploadHook(
                [
                    {
                        "base": str(WRAPPERS / "tb2_eval"),
                        "source": "*",
                        "target": PATHS.wrappers_bench + "/",
                        "flatten": True,
                    },
                    {
                        "base": str(WRAPPERS / "lagent"),
                        "source": "*",
                        "target": PATHS.wrappers_lagent + "/",
                        "flatten": True,
                    },
                ]
            ),
            # 2. Install lagent library + /tmp/lagent-py python wrapper.
            InstallLagent(),
            # 3. Weighted-pick one agent; record choice + template_root in ctx.
            PickAgent(agents=agents, template_root=str(AGENT_TEMPLATES)),
            # 4. Upload instruction.md to the workspace.
            UploadHook(
                [
                    {"source": "instruction.md", "target": f"{ws}/instruction.md"},
                ]
            ),
            # 5. Upload environment/files/* flat into the workspace.
            UploadHook(
                [
                    {
                        "base": "environment/files",
                        "source": "**/*",
                        "target": f"{ws}/",
                    },
                ]
            ),
            # 6. Overlay chosen agent's template at $TASK_WORKSPACE/agent/<name>/.
            UploadChosenAgent(target_dir=f"{ws}/agent/"),
            # 7. Ensure workspace dir exists.
            ExecHook(f"mkdir -p {ws}"),
            # 8. Exec chosen agent's config.py on host → upload resulting JSON.
            WriteAgentConfig(dst=PATHS.agent_config),
            # 9. Run install-deps.sh if the chosen agent template has one.
            RunAgentInstallDeps(workspace=ws),
        ],
        entry=AGENT_ENTRY,
        env=BenchEnv(
            workspace=ws,
            extras={"WORKSPACE": ws},
        ),
        timeout=900,
        post=[
            DownloadHook([ws, "/tmp/agent_response.txt"]),
            ReadFileHook("/tmp/message.json", "message"),
        ],
    )

    return Runner(
        infer=infer,
        validate=JudgerValidator([_rule_grader(ws)]),
    )
