"""Public config surface for RL task rollout.

This module keeps project configs readable:

    from xtuner.v1.rl_task import Runner, Stage

    runner = dict(type=Runner, infer=dict(type=Stage, ...))
"""

from xtuner.v1.ray.environment.rl_task.hooks import (
    BenchEnv,
    CopyInferWorkspace,
    DownloadHook,
    DumpDaemonLogOnFailure,
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
from lagent.serving.sandbox.providers.gateway import GatewayProvider
from xtuner.v1.ray.environment.rl_task.runner import Runner
from xtuner.v1.ray.environment.rl_task.sandbox import (
    DiagnosticFile,
    DetachedShellEntry,
    EntryCapture,
    EntryFailurePolicy,
    EntryMonitor,
    EntryProcessHealthCheck,
    EntryDiagnostics,
    ReturnCodeFileCompletion,
    SandboxStage,
    SandboxHealthCheck,
    ShellEntry,
)
from xtuner.v1.ray.environment.rl_task.schemas import (
    AgentRolloutItem,
    AgentSpec,
    EntryOutcome,
    EntryRecord,
    JudgerResult,
    RolloutError,
    RolloutStatus,
    SandboxSpec,
    SelectedAgentRecord,
    StageRecord,
    StageResult,
    StageStatus,
)
from xtuner.v1.ray.environment.rl_task.validator import Judger, JudgerValidator

Stage = SandboxStage
Validator = JudgerValidator

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
    "EntryOutcome",
    "EntryProcessHealthCheck",
    "EntryRecord",
    "DumpDaemonLogOnFailure",
    "ExecHook",
    "GatewayProvider",
    "InstallLagent",
    "Judger",
    "JudgerResult",
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
    "SandboxSpec",
    "SandboxStage",
    "SandboxHealthCheck",
    "SelectedAgentRecord",
    "ShellEntry",
    "Stage",
    "StageRecord",
    "StageResult",
    "StageStatus",
    "UploadAgentConfigSource",
    "UploadChosenAgent",
    "UploadHook",
    "Validator",
]
