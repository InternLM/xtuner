"""Public config surface for RL task rollout.

This module keeps project configs readable:

    from xtuner.v1.rl.agent_loop.rl_task import Runner, SandboxStage

    runner = dict(type=Runner, infer=dict(type=SandboxStage, ...))
"""

from xtuner.v1.rl.agent_loop.rl_task.hooks import (
    BenchEnv,
    CopyInferWorkspace,
    DownloadHook,
    ExecHook,
    InstallLagent,
    ParseJudgerStdout,
    PickAgent,
    ReadFileHook,
    RenderInstruction,
    RunAgentInstallDeps,
    UploadAgentConfigSource,
    UploadChosenAgent,
    UploadHook,
)
from xtuner.v1.rl.agent_loop.rl_task.runner import Runner
from xtuner.v1.rl.agent_loop.rl_task.sandbox import (
    DetachedShellEntry,
    DiagnosticFile,
    EntryCapture,
    EntryDiagnostics,
    EntryFailurePolicy,
    EntryMonitor,
    EntryMonitorProbe,
    EntryProcessHealthCheck,
    Hook,
    ReturnCodeFileCompletion,
    SandboxHealthCheck,
    SandboxPool,
    SandboxStage,
    ShellEntry,
)
from xtuner.v1.rl.agent_loop.rl_task.schemas import (
    AgentRolloutItem,
    AgentSpec,
    RolloutError,
    RolloutStatus,
    SandboxSpec,
    StageRecord,
    StageResult,
    StageStatus,
)
from xtuner.v1.rl.agent_loop.rl_task.validator import Judger, JudgerValidator


__all__ = [
    "AgentSpec",
    "AgentRolloutItem",
    "BenchEnv",
    "CopyInferWorkspace",
    "DiagnosticFile",
    "DetachedShellEntry",
    "DownloadHook",
    "EntryDiagnostics",
    "EntryCapture",
    "EntryFailurePolicy",
    "EntryMonitor",
    "EntryMonitorProbe",
    "EntryProcessHealthCheck",
    "ExecHook",
    "Hook",
    "InstallLagent",
    "Judger",
    "JudgerValidator",
    "ParseJudgerStdout",
    "PickAgent",
    "ReadFileHook",
    "RenderInstruction",
    "RolloutError",
    "RolloutStatus",
    "RunAgentInstallDeps",
    "Runner",
    "ReturnCodeFileCompletion",
    "SandboxPool",
    "SandboxSpec",
    "SandboxStage",
    "SandboxHealthCheck",
    "ShellEntry",
    "StageRecord",
    "StageResult",
    "StageStatus",
    "UploadAgentConfigSource",
    "UploadChosenAgent",
    "UploadHook",
]
