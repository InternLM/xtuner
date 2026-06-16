"""Public config surface for RL task rollout.

This module keeps project configs readable:

    from xtuner.v1.rl.agent_loop.sandbox_agent_loop import Runner, SandboxStage

    runner = dict(type=Runner, infer=dict(type=SandboxStage, ...), validate=...)
"""

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import (
    AgentInSandboxLoop,
    AgentInSandboxLoopConfig,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.hooks import (
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
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.judger import Judger
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.runner import Runner
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import (
    DetachedShellEntry,
    DiagnosticFile,
    EntryCapture,
    EntryDiagnostics,
    EntryFailurePolicy,
    Hook,
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


__all__ = [
    "AgentInSandboxLoop",
    "AgentInSandboxLoopConfig",
    "AgentSpec",
    "AgentRolloutItem",
    "DiagnosticFile",
    "DetachedShellEntry",
    "DownloadHook",
    "EntryDiagnostics",
    "EntryCapture",
    "EntryFailurePolicy",
    "ExecHook",
    "Hook",
    "InstallLagent",
    "Judger",
    "ParseJudgerStdout",
    "PickAgent",
    "ReadFileHook",
    "RolloutError",
    "RolloutStatus",
    "RunAgentInstallDeps",
    "Runner",
    "SandboxPool",
    "SandboxSpec",
    "SandboxStage",
    "ShellEntry",
    "StageRecord",
    "StageResult",
    "StageStatus",
    "UploadAgentConfigSource",
    "UploadChosenAgent",
    "UploadHook",
]
