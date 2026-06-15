"""Public config surface for RL task rollout.

This module keeps project configs readable:

    from xtuner.v1.rl.agent_loop.sandbox_agent_loop import Runner, SandboxStage

    runner = dict(type=Runner, infer=dict(type=SandboxStage, ...))
"""

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import (
    AgentInSandboxLoop,
    AgentInSandboxLoopConfig,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.hooks import (
    ConfigurePackageSources,
    DownloadHook,
    ExecHook,
    InstallLagent,
    ParseJudgerStdout,
    PickAgent,
    ReadFileHook,
    RunAgentInstallDeps,
    UploadAgentConfigSource,
    UploadChosenAgent,
    UploadHook,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.runner import Runner
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import (
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
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    AgentSpec,
    RolloutError,
    RolloutStatus,
    SandboxSpec,
    StageRecord,
    StageResult,
    StageStatus,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.validator import Judger, JudgerValidator


__all__ = [
    "AgentInSandboxLoop",
    "AgentInSandboxLoopConfig",
    "AgentSpec",
    "AgentRolloutItem",
    "ConfigurePackageSources",
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
