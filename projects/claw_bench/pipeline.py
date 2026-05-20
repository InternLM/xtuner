"""claw-bench rollout runner configs."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xtuner.v1.rl_task import (
    AgentSpec,
    BenchEnv,
    DetachedShellEntry,
    DownloadHook,
    EntryCapture,
    EntryDiagnostics,
    EntryFailurePolicy,
    EntryMonitor,
    EntryProcessHealthCheck,
    DumpDaemonLogOnFailure,
    ExecHook,
    GatewayProvider,
    InstallLagent,
    Judger,
    JudgerValidator,
    ParseJudgerStdout,
    PickAgent,
    ReadFileHook,
    ReturnCodeFileCompletion,
    Runner,
    RunAgentInstallDeps,
    SandboxHealthCheck,
    SandboxSpec,
    ShellEntry,
    Stage,
    UploadAgentConfigSource,
    UploadChosenAgent,
    UploadHook,
)

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

# ─────────────────────────────────────────────────────────────────
# Sandbox runtime paths  (where things live once running)
# ─────────────────────────────────────────────────────────────────

PATHS = SimpleNamespace(
    wrappers_bench="/tmp/wrappers/claw_bench",
    agent_config="/tmp/agent_config.py",
    agent_sock="/tmp/lagent_agent.sock",
    agent_daemon_log="/tmp/agent_daemon.log",
    agent_daemon_pid="/tmp/agent_daemon.pid",
    agent_response="/tmp/agent_response.txt",
    trajectory="/tmp/trajectory.json",
    message="/tmp/message.json",
    verifier="/tmp/verifier",
)

SHARED_LAGENT_PYTHON = os.getenv(
    "LAGENT_PYTHON",
    "/mnt/llm-ai-infra/miniconda3/envs/train/bin/python",
)
LAGENT_PYTHONPATH = "${TASK_WORKSPACE:-/workspace}:/tmp:${PYTHONPATH:-}"

# Entry commands.  ``cd /`` before the script is upstream convention — they
# run with ``cwd=task_dir`` so that relative ``workspace/<file>`` paths land
# under ``/workspace/``.  Equivalent here: the parent of $TASK_WORKSPACE is /,
# so cd / makes ``workspace/foo`` resolve to ``/workspace/foo``.
START_AGENT_DAEMON = (
    f'PYTHONPATH="{LAGENT_PYTHONPATH}" "{SHARED_LAGENT_PYTHON}" '
    f"-m lagent.serving.sandbox.client_cli start-agent-daemon "
    f"--mode agent --config {PATHS.agent_config} --sock {PATHS.agent_sock} "
    f"--pid-file {PATHS.agent_daemon_pid} --log {PATHS.agent_daemon_log} --truncate-log"
)
WAIT_AGENT_DAEMON = (
    f'PYTHONPATH="{LAGENT_PYTHONPATH}" "{SHARED_LAGENT_PYTHON}" '
    f"-m lagent.serving.sandbox.client_cli wait-ready "
    f"--sock {PATHS.agent_sock} --pid-file {PATHS.agent_daemon_pid} "
    f"--log {PATHS.agent_daemon_log} --timeout 60"
)
AGENT_CHAT = (
    f'PYTHONPATH="{LAGENT_PYTHONPATH}" "{SHARED_LAGENT_PYTHON}" '
    f"-m lagent.serving.sandbox.client_cli chat "
    f"--sock {PATHS.agent_sock} --instruction-file \"$TASK_INSTRUCTION\" "
    f"--response-out {PATHS.agent_response} --log {PATHS.agent_daemon_log}"
)
AGENT_STATE_DICT = (
    f'PYTHONPATH="{LAGENT_PYTHONPATH}" "{SHARED_LAGENT_PYTHON}" '
    f"-m lagent.serving.sandbox.client_cli state-dict "
    f"--sock {PATHS.agent_sock} --trajectory-out {PATHS.trajectory} "
    f"--log {PATHS.agent_daemon_log}"
)
AGENT_GET_MESSAGES = (
    f'PYTHONPATH="{LAGENT_PYTHONPATH}" "{SHARED_LAGENT_PYTHON}" '
    f"-m lagent.serving.sandbox.client_cli get-messages "
    f"--sock {PATHS.agent_sock} --message-out {PATHS.message} "
    f"--log {PATHS.agent_daemon_log}"
)
STOP_AGENT_DAEMON = (
    f'PYTHONPATH="{LAGENT_PYTHONPATH}" "{SHARED_LAGENT_PYTHON}" '
    f"-m lagent.serving.sandbox.client_cli shutdown "
    f"--sock {PATHS.agent_sock} --pid-file {PATHS.agent_daemon_pid} "
    f"--log {PATHS.agent_daemon_log}"
)


SOLUTION_ENTRY = "cd / && bash $TASK_WORKSPACE/solution/solve.sh $TASK_WORKSPACE"


def entry_failure(*, include_entry_output: bool = False) -> dict[str, Any]:
    files = [dict(path=PATHS.agent_daemon_log, key="daemon_log", optional=True)]
    if include_entry_output:
        files.extend(
            [
                dict(entry_file="stdout", key="entry_stdout", optional=True),
                dict(entry_file="stderr", key="entry_stderr", optional=True),
            ]
        )
    return dict(
        type=EntryFailurePolicy,
        diagnostics=dict(type=EntryDiagnostics, files=files),
        diagnostic_error_policy="preserve_entry_error",
    )


# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────

DEFAULT_WORKSPACE = "/workspace"
DEFAULT_AGENTS: list[dict[str, Any]] = [
    dict(
        type=AgentSpec,
        name="internclaw",
        config="config.py",
        install="install-deps.sh",
        tools="tools",
        weight=1.0,
    )
]

DEFAULT_SANDBOX = dict(
    type=SandboxSpec,
    image="ubuntu2404-v2",
    ttl_seconds=1800,
    key=os.getenv("SANDBOX_PROVIDER_KEY"),
    workspace_path=DEFAULT_WORKSPACE,
)
DEFAULT_PROVIDER = {
    "type": GatewayProvider,
    "gateway_url": os.getenv("SANDBOX_GATEWAY_URL", "http://env-gateway.ailab.ailab.ai"),
}

runner = dict(
    type=Runner,
    provider=DEFAULT_PROVIDER,
    sandboxes={"main": DEFAULT_SANDBOX},
    infer=dict(
        type=Stage,
        sandbox="main",
        runtime={},
        pre=[
            dict(
                type=UploadHook,
                mappings=[
                    dict(base=str(WRAPPERS / "claw_bench"), source="*", target=PATHS.wrappers_bench + "/", flatten=True),
                ],
            ),
            dict(type=InstallLagent),
            dict(type=PickAgent, agents=DEFAULT_AGENTS, template_root=str(AGENT_TEMPLATES)),
            dict(
                type=UploadHook,
                mappings=[
                    dict(
                        source="**/*",
                        target=f"{DEFAULT_WORKSPACE}/",
                        exclude=["task.toml", "task.py", "reference/**", "verifier/**", "solution/**", "agent/**"],
                    )
                ],
            ),
            dict(
                type=ExecHook,
                cmd=f"bash {PATHS.wrappers_bench}/pre_entry.sh",
                env={"TASK_WORKSPACE": DEFAULT_WORKSPACE},
                timeout=300,
            ),
            dict(type=UploadChosenAgent, target_dir=f"{DEFAULT_WORKSPACE}/agent/"),
            dict(type=ExecHook, cmd=f"mkdir -p {DEFAULT_WORKSPACE}"),
            dict(type=UploadAgentConfigSource, dst=PATHS.agent_config),
            dict(type=RunAgentInstallDeps, workspace=DEFAULT_WORKSPACE),
        ],
        entries=[
            dict(
                type=ShellEntry,
                name="start_agent_daemon",
                cmd=START_AGENT_DAEMON,
                timeout=60,
                failure=entry_failure(),
            ),
            dict(
                type=ShellEntry,
                name="wait_agent_daemon",
                cmd=WAIT_AGENT_DAEMON,
                timeout=90,
                failure=entry_failure(),
            ),
            dict(
                type=DetachedShellEntry,
                name="agent_chat",
                cmd=AGENT_CHAT,
                capture=dict(type=EntryCapture, root="/tmp", prefix="xt_entry"),
                monitor=dict(
                    type=EntryMonitor,
                    timeout=1800,
                    probes=[
                        dict(type=ReturnCodeFileCompletion, interval_sec=2.0),
                        dict(type=SandboxHealthCheck, interval_sec=10.0, probe_timeout_sec=10.0, fail_after=3),
                        dict(type=EntryProcessHealthCheck, interval_sec=10.0, probe_timeout_sec=10.0, fail_after=2),
                    ],
                ),
                failure=entry_failure(include_entry_output=True),
            ),
            dict(
                type=ShellEntry,
                name="agent_state_dict",
                cmd=AGENT_STATE_DICT,
                timeout=300,
                failure=entry_failure(),
            ),
            dict(
                type=ShellEntry,
                name="agent_get_messages",
                cmd=AGENT_GET_MESSAGES,
                timeout=300,
                failure=entry_failure(),
            ),
        ],
        env=dict(
            type=BenchEnv,
            workspace=DEFAULT_WORKSPACE,
            extras={"WORKSPACE": DEFAULT_WORKSPACE, "CLAW_WORKSPACE": DEFAULT_WORKSPACE},
        ),
        post=[
            dict(type=DownloadHook, paths=[DEFAULT_WORKSPACE, PATHS.agent_response]),
            dict(type=ReadFileHook, path="/tmp/message.json", key="message"),
            dict(type=DumpDaemonLogOnFailure),
            dict(type=ExecHook, cmd=STOP_AGENT_DAEMON, optional=True, timeout=30),
        ],
    ),
    validate=dict(
        type=JudgerValidator,
        judgers=[
            dict(
                type=Judger,
                name="rule_grader",
                weight=1.0,
                stage=dict(
                    type=Stage,
                    sandbox="main",
                    pre=[
                        dict(
                            type=UploadHook,
                            mappings=[
                                dict(base="verifier", source="**/*", target=f"{PATHS.verifier}/rule_grader/")
                            ],
                        )
                    ],
                    entries=[
                        dict(
                            type=ShellEntry,
                            name="run_tests",
                            cmd=f"bash {PATHS.wrappers_bench}/pytest_ctrf.sh",
                                        timeout=300,
                        )
                    ],
                    env={
                        "JUDGER_NAME": "rule_grader",
                        "TASK_WORKSPACE": DEFAULT_WORKSPACE,
                        "TASK_JUDGER_DIR": f"{PATHS.verifier}/rule_grader",
                        "PYTEST_TARGET": f"{PATHS.verifier}/rule_grader/test_output.py",
                    },
                    post=[dict(type=ParseJudgerStdout, judger_name="rule_grader")],
                ),
            )
        ],
    ),
)

solution_runner = dict(
    type=Runner,
    provider=DEFAULT_PROVIDER,
    sandboxes={
        "main": dict(
            type=SandboxSpec,
            image="ubuntu2404-v2",
            ttl_seconds=600,
            workspace_path=DEFAULT_WORKSPACE,
        )
    },
    infer=dict(
        type=Stage,
        sandbox="main",
        runtime={},
        pre=[
            dict(type=ExecHook, cmd=f"mkdir -p {DEFAULT_WORKSPACE}"),
            dict(
                type=UploadHook,
                mappings=[
                    dict(base=str(WRAPPERS / "claw_bench"), source="*", target=PATHS.wrappers_bench + "/", flatten=True),
                ],
            ),
            dict(
                type=UploadHook,
                mappings=[
                    dict(
                        source="**/*",
                        target=f"{DEFAULT_WORKSPACE}/",
                        exclude=["task.toml", "task.py", "verifier/**", "agent/**"],
                    )
                ],
            ),
            dict(
                type=ExecHook,
                cmd=f"bash {PATHS.wrappers_bench}/pre_entry.sh",
                env={"TASK_WORKSPACE": DEFAULT_WORKSPACE},
                timeout=300,
            ),
        ],
        entries=[
            dict(
                type=ShellEntry,
                name="solution",
                cmd=SOLUTION_ENTRY,
                timeout=600,
            )
        ],
        env=dict(
            type=BenchEnv,
            workspace=DEFAULT_WORKSPACE,
            extras={"WORKSPACE": DEFAULT_WORKSPACE, "CLAW_WORKSPACE": DEFAULT_WORKSPACE},
        ),
        post=[dict(type=DownloadHook, paths=[DEFAULT_WORKSPACE])],
    ),
    validate=runner["validate"],
)
