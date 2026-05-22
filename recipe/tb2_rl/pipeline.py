"""tb2-rl rollout runner config.

Layout of an upstream tb2-rl task::

    task.toml              — metadata
    instruction.md         — natural-language task (paths are absolute, e.g. /app/foo.csv)
    environment/Dockerfile — baked into the pre-built ``t-data-processing-v1`` image
    environment/files/     — data files that the Dockerfile COPYs to /app/
    tests/test.sh          — bench-provided verifier entrypoint
    tests/test_outputs.py  — pytest module
    tests/test_requirements.txt

At runtime we do NOT rebuild the image; we use the pre-built
``t-data-processing-v1`` and seed ``/app`` + ``/tests`` ourselves:
  - mirror ``instruction.md`` → ``/app/instruction.md``
  - mirror ``environment/files/*`` → ``/app/``
  - mirror ``tests/*`` → ``/tests/`` (bench's test.sh reads from there)
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lagent.serving.sandbox.providers.gateway import GatewayProvider

from xtuner.v1.rl.agent_loop.rl_task import (
    AgentSpec,
    DetachedShellEntry,
    DownloadHook,
    EntryCapture,
    EntryDiagnostics,
    EntryFailurePolicy,
    EntryMonitor,
    EntryProcessHealthCheck,
    ExecHook,
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
    SandboxPool,
    SandboxSpec,
    SandboxStage,
    ShellEntry,
    UploadAgentConfigSource,
    UploadChosenAgent,
    UploadHook,
)

HERE = Path(__file__).resolve().parent
SETUP_DIR = HERE / "infer" / "setup"
AGENT_TEMPLATES = HERE / "infer" / "agents"
JUDGERS = HERE / "judgers"


# ─────────────────────────────────────────────────────────────────
# Sandbox runtime paths
# ─────────────────────────────────────────────────────────────────

PATHS = SimpleNamespace(
    setup_dir="/tmp/infer/setup",
    judger_dir="/tmp/judgers/rule_grader",
    agent_config="/tmp/agent_config.py",
    agent_sock="/tmp/lagent_agent.sock",
    agent_daemon_log="/tmp/agent_daemon.log",
    agent_daemon_pid="/tmp/agent_daemon.pid",
    agent_response="/tmp/agent_response.txt",
    trajectory="/tmp/trajectory.json",
    message="/tmp/message.json",
    tests="/tests",
)

SHARED_LAGENT_PYTHON = os.getenv(
    "LAGENT_PYTHON",
    "/mnt/llm-ai-infra/miniconda3/envs/train/bin/python",
)
LAGENT_PYTHONPATH = "/app:/tmp:${PYTHONPATH:-}"

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
    f"--sock {PATHS.agent_sock} --instruction-file \"/app/instruction.md\" "
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

DEFAULT_WORKSPACE = "/app"
DEFAULT_AGENTS: list[dict[str, Any]] = [
    dict(
        type=AgentSpec,
        name="interndp",
        config="config.py",
        install="install-deps.sh",
        tools="tools",
        weight=1.0,
    )
]

DEFAULT_SANDBOX = dict(
    type=SandboxSpec,
    image="t-data-processing-v1",
    ttl_seconds=11700,
    key=os.getenv("SANDBOX_PROVIDER_KEY"),
    workspace_path=DEFAULT_WORKSPACE,
)
DEFAULT_PROVIDER = {
    "type": GatewayProvider,
    "gateway_url": os.getenv("SANDBOX_GATEWAY_URL", "http://env-gateway.ailab.ailab.ai"),
}

runner = dict(
    type=Runner,
    pool=dict(
        type=SandboxPool,
        provider=DEFAULT_PROVIDER,
        specs={"main": DEFAULT_SANDBOX},
    ),
    infer=dict(
        type=SandboxStage,
        sandbox="main",
        pre=[
            # ── Stage 1: workspace setup ──────────────────────────────────
            # Create the workspace dir, upload bench-level setup scripts,
            # then run pre_entry.sh for any per-bench bootstrap.
            dict(type=ExecHook, cmd=f"mkdir -p {DEFAULT_WORKSPACE}"),
            dict(
                type=UploadHook,
                mappings=[
                    dict(base=str(SETUP_DIR), source="*", target=PATHS.setup_dir + "/", flatten=True),
                ],
            ),
            dict(
                type=ExecHook,
                cmd=f"bash {PATHS.setup_dir}/pre_entry.sh",
                env={"TASK_WORKSPACE": DEFAULT_WORKSPACE},
                timeout=300,
            ),

            # ── Stage 2: task data ────────────────────────────────────────
            # Place per-task files (instruction + environment/files/*) under
            # the workspace so the agent sees them.
            dict(type=UploadHook, mappings=[dict(source="instruction.md", target=f"{DEFAULT_WORKSPACE}/instruction.md")]),
            dict(
                type=UploadHook,
                mappings=[dict(base="environment/files", source="**/*", target=f"{DEFAULT_WORKSPACE}/")],
            ),

            # ── Stage 3: agent harness ────────────────────────────────────
            # Install lagent runtime, pick an agent variant, upload its
            # harness + config, run its install-deps.
            dict(type=InstallLagent, lagent_src_dir=os.getenv("LAGENT_SRC_DIR", "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent")),
            dict(type=PickAgent, agents=DEFAULT_AGENTS, template_root=str(AGENT_TEMPLATES)),
            dict(type=UploadChosenAgent, target_dir=f"{DEFAULT_WORKSPACE}/agent/"),
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
                    timeout=7200,
                    probes=[
                        dict(type=ReturnCodeFileCompletion, interval_sec=2.0),
                        dict(type=SandboxHealthCheck, interval_sec=10.0, probe_timeout_sec=10.0, fail_after=3),
                        dict(type=EntryProcessHealthCheck, interval_sec=10.0, probe_timeout_sec=10.0, fail_after=2),
                    ],
                ),
                failure=entry_failure(include_entry_output=True),
            ),
            # dict(
            #     type=ShellEntry,
            #     name="agent_state_dict",
            #     cmd=AGENT_STATE_DICT,
            #     timeout=300,
            #     failure=entry_failure(),
            # ),
            dict(
                type=ShellEntry,
                name="agent_get_messages",
                cmd=AGENT_GET_MESSAGES,
                timeout=300,
                failure=entry_failure(),
            ),
            # `|| true` 让 stop 失败不污染 stage status —— sandbox 一会儿
            # 也会被 pool.release_all 释放,daemon 自然死。debug 想保留
            # daemon 的话注释掉这条 entry 即可。
            dict(
                type=ShellEntry,
                name="stop_agent_daemon",
                cmd=STOP_AGENT_DAEMON + " || true",
                timeout=30,
            ),
        ],
        post=[
            dict(type=ReadFileHook, path=PATHS.message, key="message"),
            dict(type=ReadFileHook, path=PATHS.agent_response, key="agent_response"),
            # workspace tar(debug 时本地解压看产物,失败 silent + warning)
            # dict(type=DownloadHook, paths=[DEFAULT_WORKSPACE]),
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
                    type=SandboxStage,
                    sandbox="main",
                    pre=[
                        dict(
                            type=UploadHook,
                            mappings=[
                                dict(base=str(JUDGERS / "rule_grader"), source="*", target=PATHS.judger_dir + "/", flatten=True),
                                dict(base="tests", source="**/*", target=f"{PATHS.tests}/"),
                            ],
                        ),
                        dict(type=ExecHook, cmd=f"chmod +x {PATHS.tests}/test.sh || true", optional=True),
                    ],
                    entries=[
                        dict(
                            type=ShellEntry,
                            name="run_tests",
                            cmd=f"bash {PATHS.judger_dir}/run.sh",
                            env={
                                "JUDGER_NAME": "rule_grader",
                                "TASK_WORKSPACE": DEFAULT_WORKSPACE,
                                "TESTS_DIR": PATHS.tests,
                                "JUDGER_DIR": PATHS.judger_dir,
                            },
                            timeout=900,
                        )
                    ],
                    post=[dict(type=ParseJudgerStdout, judger_name="rule_grader")],
                ),
            )
        ],
    ),
)
