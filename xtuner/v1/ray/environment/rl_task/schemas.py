"""Pydantic schemas for the RL task pipeline.

Only domain-level data shapes live here:
  - :class:`AgentRolloutItem` — one sample from dataset input through rollout
    output.
  - :class:`SandboxSpec` / :class:`AgentSpec` — reusable building blocks.
  - :class:`JudgerResult` — validate output, for both individual judgers
    and the aggregate result.

Pipeline config is intentionally just a lagent-style Python dict.  A config
can be passed directly, or by import path, and ``type`` may be either an
imported class/function object or a string path understood by
``lagent.utils.create_object``.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────
# Rollout item
# ─────────────────────────────────────────────────────────────────


class RolloutStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EntryReturnCode(IntEnum):
    PID_LOST = -1
    DAEMON_GONE = -2
    TIMEOUT = -3
    SANDBOX_UNREACHABLE = -4


class ReturnCodeKind(str, Enum):
    OK = "ok"
    SCRIPT_ERROR = "script_error"
    TIMEOUT = "timeout"
    DAEMON_GONE = "daemon_gone"
    PID_LOST = "pid_lost"
    SANDBOX_UNREACHABLE = "sandbox_unreachable"
    OOM = "oom"
    UNKNOWN = "unknown"


class RolloutError(BaseModel):
    """Structured error record for the rollout or one stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str | None = None
    category: str
    type: str | None = None
    message: str
    retryable: bool | None = None


class SelectedAgentRecord(BaseModel):
    """Serializable record of the agent selected for an infer stage."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: str = "config.py"
    install: str | None = None
    tools: str | None = None
    weight: float = 1.0
    template_root: str


class StageResult(BaseModel):
    """Raw entry-command result for one stage."""

    model_config = ConfigDict(extra="forbid")

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.return_code == 0


class EntryOutcome(BaseModel):
    """Structured reason why an entry finished.

    ``StageResult`` is the shell-level payload.  ``EntryOutcome`` records the
    observation that produced that payload: a sync command return, a detached
    rc-file completion, a monitor failure, or an exception path.
    """

    model_config = ConfigDict(extra="forbid")

    result: StageResult = Field(default_factory=StageResult)
    source: str = "entry"
    reason: str | None = None
    retryable: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result.ok


class EntryRecord(BaseModel):
    """Observable record for one stage entry execution."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    cmd: str
    mode: str = "sync"
    status: StageStatus = StageStatus.PENDING

    pid_file: str | None = None
    rc_file: str | None = None
    stdout_file: str | None = None
    stderr_file: str | None = None

    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    return_code_kind: ReturnCodeKind | None = None
    result: StageResult | None = None
    outcome: EntryOutcome | None = None
    error: RolloutError | None = None
    diagnostics: dict[str, str] = Field(default_factory=dict)
    diagnostic_errors: list[dict[str, str]] = Field(default_factory=list)


class StageRecord(BaseModel):
    """Shallow execution record for one stage.

    ``StageRecord`` is both the runtime write target for the current stage
    and the durable ledger left on :class:`AgentRolloutItem`.  Runtime-only
    fields such as ``runtime`` and ``env_vars`` are excluded from dumps so
    secrets injected by the trainer do not leak into results.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True)

    status: StageStatus = StageStatus.PENDING
    sandbox_name: str | None = None
    sandbox_image: str | None = None
    sandbox_env_id: str | None = None
    sandbox_url: str | None = None
    workspace: str | None = None

    entry_cmd: str | None = None
    return_code: int | None = None
    return_code_kind: ReturnCodeKind | None = None

    entries: list[EntryRecord] = Field(default_factory=list)
    entry_result: StageResult | None = None
    result: "StageResult | JudgerResult | None" = None
    error: RolloutError | None = None

    started_at: float | None = None
    finished_at: float | None = None

    agent: SelectedAgentRecord | None = None
    judger_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    hook_errors: list[dict[str, str]] = Field(default_factory=list)

    runtime: dict[str, Any] = Field(default_factory=dict, exclude=True)
    env_vars: dict[str, str] = Field(default_factory=dict, exclude=True)


class AgentRolloutItem(BaseModel):
    """One agent rollout sample, from input through final result.

    This is the shallow "row header" used by the rollout runner.  Fixed
    fields are limited to values that runner/stage code must read or write.
    Dataset-specific provenance such as an original source name belongs in
    ``metadata`` and is carried through opaquely.

    Fields that drive pipeline behavior:
      - ``instruction``: path (relative to task root) of the natural-language
        task the agent sees.  Pipeline exports it as ``$TASK_INSTRUCTION``.
      - ``task_root``: host task directory available to the current worker.
      - ``uid``: rollout identity supplied by the outer dataflow.
      - ``pipeline`` / ``pipeline_overrides``: lazy runner config binding.

    Runtime/result fields are filled in place by :class:`Runner.run`:
      - ``status``, ``score``, ``infer``, ``validation``, ``judgers``,
        ``artifacts`` and ``error``.

    Any judger-only files (reference answers, fixture data) are the
    judger's own concern — it uploads what it needs via its own
    pre-hooks.  No hidden ``reference`` field here.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True)

    # Stable sample identity.
    id: str
    data_source: str
    ability: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Behavior.
    instruction: str  # relative to task root
    task_root: Path | None = None
    uid: dict[str, int] = Field(default_factory=dict)
    pipeline: "PipelineConfig | Any | None" = None
    pipeline_overrides: dict[str, Any] = Field(default_factory=dict)

    # Output and observation.
    status: RolloutStatus = RolloutStatus.PENDING
    score: float | None = None
    trajectory: Any | None = None
    infer: StageRecord = Field(default_factory=StageRecord)
    validation: StageRecord = Field(default_factory=StageRecord, alias="validate")
    judgers: dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    # Opaque dataset/business metadata. Runner core does not branch on keys here.
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: RolloutError | None = None


# ─────────────────────────────────────────────────────────────────
# Stage building blocks
# ─────────────────────────────────────────────────────────────────


class SandboxSpec(BaseModel):
    """Sandbox runtime config shared by infer and isolated-judger sandboxes."""

    model_config = ConfigDict(extra="forbid")

    image: str
    ttl_seconds: int = 11700
    workspace_path: str = "/workspace"  # becomes $TASK_WORKSPACE
    env_vars: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    key: str | None = None


class AgentSpec(BaseModel):
    """One agent implementation available for the infer stage.

    Paths are relative to the agent's template directory (``<dataset.agent_template_root>/<name>/``) — NOT to the task
    dir. Each agent candidate has its own template subtree; tasks only carry data, never agent code.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    config: str = "config.py"  # relative to template dir
    install: str | None = None  # relative; optional
    tools: str | None = None  # relative dir; optional
    weight: float = 1.0


PipelineConfig = dict[str, Any] | str


# ─────────────────────────────────────────────────────────────────
# Judger result contract
# ─────────────────────────────────────────────────────────────────


class JudgerResult(BaseModel):
    """Validate output.

    A leaf result comes from one judger stage.  The aggregate result uses the
    same shape, with ``judger_name="aggregate"`` and ``per_judger`` populated.
    """

    model_config = ConfigDict(extra="forbid")

    judger_name: str
    total: float = Field(ge=0, le=1)
    criteria: dict[str, dict[str, float]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failed: bool = False
    per_judger: list["JudgerResult"] = Field(default_factory=list)


StageRecord.model_rebuild()
JudgerResult.model_rebuild()
AgentRolloutItem.model_rebuild()


__all__ = [
    "AgentRolloutItem",
    "EntryOutcome",
    "EntryRecord",
    "EntryReturnCode",
    "ReturnCodeKind",
    "RolloutError",
    "RolloutStatus",
    "SandboxSpec",
    "AgentSpec",
    "SelectedAgentRecord",
    "StageRecord",
    "StageResult",
    "StageStatus",
    "JudgerResult",
    "PipelineConfig",
]
