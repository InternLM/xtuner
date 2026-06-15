"""Sandbox judgers."""

from __future__ import annotations

from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool, SandboxStage
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    StageRecord,
    StageStatus,
)


class Judger:
    """A named sandbox scoring stage."""

    def __init__(self, name: str, stage: SandboxStage | dict[str, Any]):
        self.name = name
        self.stage = create_object(stage)

    async def run(
        self,
        item: AgentRolloutItem,
        pool: SandboxPool,
        record: StageRecord | None = None,
    ) -> float:
        record = record or item.judgers.setdefault(self.name, StageRecord(judger_name=self.name))
        try:
            sandbox_name = _judger_sandbox_name(self.stage, pool)
            client = await pool.get(sandbox_name, record=record)
            spec = pool.spec(sandbox_name)
            record.sandbox_name = sandbox_name
            record.sandbox_image = spec.image
            record.workspace = spec.workspace_path
            record.judger_name = self.name
            await self.stage.run(client, item, record)
            if record.status == StageStatus.FAILED:
                raise RuntimeError(record.error.message if record.error is not None else "judger stage failed")
            if record.score is None:
                raise RuntimeError("judger stage did not produce record.score")
            return float(record.score)
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=self.name,
                category="judger",
                type=type(exc).__name__,
                message=str(exc),
            )
            raise


class ComposeJudger:
    """Base class for custom composed judgers.

    A composed judger is still a judger from the runner's point of view: it
    exposes ``run(item, pool, record) -> float`` and owns its top-level record.
    Child execution order, short-circuiting, and scoring policy belong in the
    concrete ``run`` implementation instead of framework presets.
    """

    def __init__(
        self,
        judgers: list[Judger | dict[str, Any]],
        *,
        name: str = "validate",
    ):
        if not judgers:
            raise ValueError("ComposeJudger.judgers is empty")
        self.name = name
        self.judgers = [create_object(judger) for judger in judgers]

    async def run(
        self,
        item: AgentRolloutItem,
        pool: SandboxPool,
        record: StageRecord | None = None,
    ) -> float:
        raise NotImplementedError(f"{type(self).__name__}.run must implement composed judging")

    def _fail(self, record: StageRecord, item: AgentRolloutItem) -> float:
        record.status = StageStatus.FAILED
        child_error = next(
            (child.error for child in item.judgers.values() if child is not record and child.error is not None),
            None,
        )
        record.error = (
            record.error
            or child_error
            or RolloutError(
                stage=self.name,
                category="validate_failed",
                type=type(self).__name__,
                message="validate failed",
            )
        )
        raise RuntimeError(record.error.message)


def _judger_sandbox_name(stage: SandboxStage, pool: SandboxPool) -> str:
    name = stage.sandbox
    if not isinstance(name, str):
        raise TypeError(f"SandboxStage.sandbox must be a sandbox name, got {type(name).__name__}")
    pool.validate_name(name)
    return name


__all__ = ["ComposeJudger", "Judger"]
