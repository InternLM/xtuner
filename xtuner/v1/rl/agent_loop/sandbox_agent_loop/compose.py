"""Composable sandbox validation stages."""

from __future__ import annotations

import time
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    StageRecord,
    StageStatus,
)


class SandboxComposeStage:
    """Compose multiple sandbox validation stages behind ``run(...) -> float``.

    Stages with ``weight=0`` still run, but do not contribute to the returned
    score. This is used for process-adv annotators that mutate rollout
    artifacts without changing outcome reward.
    """

    def __init__(
        self,
        stages: list[Any],
        *,
        name: str = "validate",
        weight: float = 1.0,
    ):
        if not stages:
            raise ValueError("SandboxComposeStage.stages is empty")
        self.name = name
        self.stages = [create_object(stage) for stage in stages]
        self.weight = weight

    async def run(self, item: AgentRolloutItem, pool: SandboxPool, record: StageRecord) -> float:
        record.status = StageStatus.RUNNING
        record.started_at = record.started_at or time.monotonic()
        record.judger_name = self.name
        try:
            weighted_score = 0.0
            total_weight = 0.0
            for stage in self.stages:
                name = getattr(stage, "name", stage.__class__.__name__)
                child_record = item.judgers.setdefault(name, StageRecord(judger_name=name))
                score = float(await stage.run(item, pool, child_record))
                stage_weight = max(float(getattr(stage, "weight", 1.0)), 0.0)
                weighted_score += score * stage_weight
                total_weight += stage_weight
            record.score = weighted_score / total_weight if total_weight > 0 else 0.0
            record.status = StageStatus.COMPLETED
            return record.score
        except Exception as exc:
            record.status = StageStatus.FAILED
            child_error = next(
                (child.error for child in item.judgers.values() if child.error is not None),
                None,
            )
            record.error = (
                record.error
                or child_error
                or RolloutError(
                    stage=self.name,
                    category="validate_failed",
                    type=type(exc).__name__,
                    message=str(exc),
                )
            )
            raise
        finally:
            record.finished_at = time.monotonic()


__all__ = ["SandboxComposeStage"]
