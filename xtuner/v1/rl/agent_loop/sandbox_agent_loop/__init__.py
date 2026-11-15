"""Public config surface for RL task rollout.

This module keeps project configs readable:

    from xtuner.v1.rl.agent_loop.sandbox_agent_loop import Runner, SandboxStage

    runner = dict(type=Runner, infer=dict(type=SandboxStage, ...), validate=...)
"""

from importlib import import_module
from typing import Any


# Cleanup entry points are also imported by ordinary RL without Lagent.
# Resolve the config surface only when a caller actually requests it.
_EXPORT_MODULES = {
    "AgentInSandboxLoop": "agent_in_sandbox_loop",
    "AgentInSandboxLoopConfig": "agent_in_sandbox_loop",
    "DownloadHook": "hooks",
    "ExecHook": "hooks",
    "InstallLagent": "hooks",
    "ParseJudgerStdout": "hooks",
    "PickAgent": "hooks",
    "ReadFileHook": "hooks",
    "RunAgentInstallDeps": "hooks",
    "UploadAgentConfigSource": "hooks",
    "UploadChosenAgent": "hooks",
    "UploadHook": "hooks",
    "Judger": "judger",
    "Runner": "runner",
    "DetachedShellEntry": "sandbox",
    "DiagnosticFile": "sandbox",
    "EntryCapture": "sandbox",
    "EntryDiagnostics": "sandbox",
    "EntryFailurePolicy": "sandbox",
    "Hook": "sandbox",
    "SandboxPool": "sandbox",
    "SandboxStage": "sandbox",
    "ShellEntry": "sandbox",
    "AgentRolloutItem": "schemas",
    "AgentSpec": "schemas",
    "RolloutError": "schemas",
    "RolloutStatus": "schemas",
    "SandboxSpec": "schemas",
    "StageRecord": "schemas",
    "StageResult": "schemas",
    "StageStatus": "schemas",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
