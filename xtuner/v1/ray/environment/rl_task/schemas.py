"""Pydantic schemas for the RL task pipeline.

Only domain-level data shapes live here:
  - :class:`TaskData` — what task.py's ``data = dict(...)`` describes.
  - :class:`SandboxSpec` / :class:`AgentSpec` — reusable building blocks.
  - :class:`JudgerResult` / :class:`AggregatedScore` — the scoring contract.

Stage-level config (infer / validate / judger) is NOT modeled here; each
stage class (``Inferencer`` subclass, ``Judger`` subclass) validates its
own kwargs.  This lets stages evolve independently.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────
# Task Data
# ─────────────────────────────────────────────────────────────────


class WorkspaceArtifact(BaseModel):
    """One required/expected product in ``$TASK_WORKSPACE`` after infer."""

    model_config = ConfigDict(extra="forbid")

    path: str                                            # relative to $TASK_WORKSPACE
    required: bool = True
    kind: Literal["json", "text", "binary", "directory"] = "text"
    schema_ref: str | None = None


class TaskData(BaseModel):
    """Per-task content contract.

    A task directory ships one ``task.py`` which constructs this with a
    ``data = dict(...)`` literal.  Only three fields drive pipeline
    behavior — the rest are metadata for result bookkeeping:

    Behavioral:
      - ``instruction``: path (relative to task root) of the natural-language
        task the agent sees.  The inferencer exports it as
        ``$TASK_INSTRUCTION``.
      - ``reference``: optional path to a ground-truth directory.  If set,
        the Validator stages it outside the agent workspace so judgers can
        read it without the agent ever seeing it.
      - ``outputs``: declared artifacts.  After infer, :meth:`Inferencer.check_outputs`
        reports which ``required`` ones are missing — a fast-fail signal
        before judgers run.

    Metadata (surfaced in result envelope):
      - ``id``, ``data_source``, ``ability``, ``tags``: routing / aggregation.
    """

    model_config = ConfigDict(extra="allow")

    # Metadata.
    id: str
    data_source: str
    ability: str | None = None
    tags: list[str] = []

    # Behavior.
    instruction: str                                     # relative to task root
    reference: str | None = None                         # relative dir; optional
    outputs: list[WorkspaceArtifact] = []


# ─────────────────────────────────────────────────────────────────
# Stage building blocks
# ─────────────────────────────────────────────────────────────────


class SandboxSpec(BaseModel):
    """Sandbox runtime config shared by infer and isolated-judger sandboxes."""

    model_config = ConfigDict(extra="forbid")

    image: str
    ttl_seconds: int = 1800
    workspace_path: str = "/workspace"                   # becomes $TASK_WORKSPACE
    env_vars: dict[str, str] = {}
    resources: dict[str, Any] = {}


class AgentSpec(BaseModel):
    """One agent implementation available for the infer stage.

    Paths are relative to the agent's template directory
    (``<dataset.agent_template_root>/<name>/``) — NOT to the task dir.
    Each agent candidate has its own template subtree; tasks only carry
    data, never agent code.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    config: str = "config.py"                            # relative to template dir
    install: str | None = None                           # relative; optional
    tools: str | None = None                             # relative dir; optional
    weight: float = 1.0


# ─────────────────────────────────────────────────────────────────
# Judger result contract
# ─────────────────────────────────────────────────────────────────


class CriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    weight: float = 1.0


class StepReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: int
    reward: float
    tag: str | None = None


class JudgerResult(BaseModel):
    """Stdout-JSON contract for a judger's script output."""

    model_config = ConfigDict(extra="forbid")

    judger_name: str
    total: float = Field(ge=0, le=1)
    criteria: dict[str, CriterionScore] = {}
    step_rewards: list[StepReward] = []
    metadata: dict[str, Any] = {}
    error: str | None = None


class AggregatedScore(BaseModel):
    """Per-item aggregate after all judgers."""

    model_config = ConfigDict(extra="forbid")

    total: float
    per_judger: list[JudgerResult]
    step_rewards: list[StepReward] = []
    failed: bool = False


__all__ = [
    "TaskData",
    "SandboxSpec",
    "AgentSpec",
    "WorkspaceArtifact",
    "CriterionScore",
    "StepReward",
    "JudgerResult",
    "AggregatedScore",
]
