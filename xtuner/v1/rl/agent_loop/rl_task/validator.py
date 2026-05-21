"""Validator — fans out to a list of :class:`Judger` stages, aggregates scores.

Judgers are self-contained: each one's stage declares everything it needs
(uploads, env vars, entry, post-hooks).  The validator just orchestrates
and aggregates ``record.score`` across judger records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.rl_task.sandbox import SandboxPool, SandboxStage
from xtuner.v1.rl.agent_loop.rl_task.schemas import (
    AgentRolloutItem,
    RolloutError,
    StageRecord,
    StageStatus,
)


@dataclass
class Judger:
    """A named verifier stage.

    Sandbox selection belongs to ``stage.sandbox``.  The Judger only owns
    validation semantics: name + weight.
    """

    name: str
    stage: SandboxStage
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.stage = create_object(self.stage)


class JudgerValidator:
    """Run every judger, aggregate scores into a single ``(score, failed)``.

    Init args:
        judgers (list[Judger]): Judger instances to fan out to.
        aggregator (str): ``"weighted_sum"`` / ``"mean"`` / ``"max"`` /
            ``"min"`` / ``"all_or_nothing"``.
        on_error (str): ``"zero"`` (aggregate over usable) or ``"fail"`` (any
            judger error → ``(0.0, failed=True)``).
    """

    def __init__(
        self,
        judgers: list[Judger | dict[str, Any]],
        *,
        aggregator: Literal[
            "weighted_sum",
            "mean",
            "max",
            "min",
            "all_or_nothing",
        ] = "weighted_sum",
        on_error: Literal["zero", "fail"] = "zero",
    ):
        self.judgers = [create_object(judger) for judger in judgers]
        self.aggregator = aggregator
        self.on_error = on_error

    async def run(self, item: AgentRolloutItem, pool: SandboxPool) -> tuple[float, bool]:
        """Run every judger and aggregate.

        Returns:
            tuple[float, bool]: ``(score, failed)``. ``failed`` is True when no
            usable judger remains or ``on_error="fail"`` and any judger errored.
        """
        for j in self.judgers:
            await self._run_one(j, item, pool)
        return self._aggregate(item)

    # -- internals --

    async def _run_one(self, j: Judger, item: AgentRolloutItem, pool: SandboxPool) -> None:
        record = item.judgers.setdefault(j.name, StageRecord(judger_name=j.name))
        try:
            sandbox_name = _judger_sandbox_name(j.stage, pool)
            client = await pool.get(sandbox_name, record=record)
            spec = pool.spec(sandbox_name)
            record.sandbox_name = sandbox_name
            record.sandbox_image = spec.image
            record.workspace = spec.workspace_path
            record.judger_name = j.name
            await j.stage.run(client, item, record)
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=j.name,
                category="judger",
                type=type(exc).__name__,
                message=str(exc),
            )

    def _aggregate(self, item: AgentRolloutItem) -> tuple[float, bool]:
        weights = {j.name: j.weight for j in self.judgers}
        usable: list[tuple[str, float]] = []
        any_error = False
        for j in self.judgers:
            record = item.judgers.get(j.name)
            if record is None or record.status == StageStatus.FAILED or record.score is None:
                any_error = True
                continue
            usable.append((j.name, record.score))

        if any_error and self.on_error == "fail":
            return 0.0, True
        if not usable:
            return 0.0, True

        if self.aggregator == "weighted_sum":
            tw = sum(weights.get(name, 1.0) for name, _ in usable)
            total = sum(score * weights.get(name, 1.0) for name, score in usable) / tw if tw else 0.0
        elif self.aggregator == "mean":
            total = sum(score for _, score in usable) / len(usable)
        elif self.aggregator == "max":
            total = max(score for _, score in usable)
        elif self.aggregator == "min":
            total = min(score for _, score in usable)
        elif self.aggregator == "all_or_nothing":
            total = 1.0 if all(score >= 1.0 for _, score in usable) else 0.0
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return total, False


def _judger_sandbox_name(stage: SandboxStage, pool: SandboxPool) -> str:
    name = stage.sandbox
    if not isinstance(name, str):
        raise TypeError(f"SandboxStage.sandbox must be a sandbox name, got {type(name).__name__}")
    pool.validate_name(name)
    return name
