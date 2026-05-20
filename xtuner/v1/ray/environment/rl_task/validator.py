"""Validator — fans out to a list of :class:`Judger` stages, aggregates scores.

Judgers are self-contained: each one's stage declares everything it needs
(uploads, env vars, entry, post-hooks).  The validator just orchestrates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping

from lagent.utils import create_object

from xtuner.v1.ray.environment.rl_task.sandbox import Hook, SandboxStage
from xtuner.v1.ray.environment.rl_task.schemas import (
    AgentRolloutItem,
    JudgerResult,
    RolloutError,
    StageRecord,
    StageStatus,
)

SandboxResolver = Callable[[str], Awaitable[Any]]


@dataclass
class Judger:
    """A named verifier stage.

    Sandbox selection belongs to ``stage.sandbox``.  The Judger only owns
    validation semantics: name, weight, and optional isolated-sandbox setup.
    """

    name: str
    stage: SandboxStage
    weight: float = 1.0
    on_isolated_pre: list[Hook] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = create_object(self.stage)
        self.on_isolated_pre = [create_object(hook) for hook in self.on_isolated_pre]


class JudgerValidator:
    """Run every judger, aggregate results.

    Init args:
        judgers (list[Judger]): Judger instances to fan out to.
        aggregator (str): ``"weighted_sum"`` / ``"mean"`` / ``"max"`` /
            ``"min"`` / ``"all_or_nothing"``.
        on_error (str): ``"zero"`` (sum over usable) or ``"fail"`` (any
            error → total=0, failed=True).
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

    async def run(
        self,
        item: AgentRolloutItem,
        get_sandbox: SandboxResolver,
        *,
        sandboxes: Mapping[str, Any],
        infer_sandbox: str,
        infer_workspace: str,
    ) -> JudgerResult:
        item.validation.status = StageStatus.RUNNING
        item.validation.started_at = item.validation.started_at or time.monotonic()
        results: list[JudgerResult] = []
        try:
            for j in self.judgers:
                results.append(await self._run_one(j, item, get_sandbox, sandboxes, infer_sandbox, infer_workspace))
            aggregated = self._aggregate(results)
            item.validation.result = aggregated
            item.validation.status = StageStatus.FAILED if aggregated.failed else StageStatus.COMPLETED
            if aggregated.failed:
                item.validation.error = RolloutError(
                    stage="validate",
                    category="validate",
                    type="JudgerValidator",
                    message="all judgers failed" if not results else "validate failed",
                )
            else:
                item.validation.error = None
            return aggregated
        finally:
            item.validation.finished_at = time.monotonic()

    # -- internals --

    async def _run_one(
        self,
        j: Judger,
        item: AgentRolloutItem,
        get_sandbox: SandboxResolver,
        sandboxes: Mapping[str, Any],
        infer_sandbox: str,
        infer_workspace: str,
    ) -> JudgerResult:
        record = item.judgers.setdefault(j.name, StageRecord(judger_name=j.name))
        try:
            sandbox_name = self._stage_sandbox_name(j.stage, sandboxes)
            client = await get_sandbox(sandbox_name)
            spec = sandboxes[sandbox_name]
            isolated = sandbox_name != infer_sandbox
            j_workspace = spec.workspace_path if isolated else infer_workspace
            record.sandbox_name = sandbox_name
            record.sandbox_image = spec.image
            record.workspace = j_workspace
            record.judger_name = j.name
            if isolated:
                infer_client = await get_sandbox(infer_sandbox)
                record.runtime.update(
                    {
                        "infer_client": infer_client,
                        "infer_workspace": infer_workspace,
                        "target_workspace": j_workspace,
                    }
                )
                for hook in j.on_isolated_pre:
                    await hook(client, item, record)
            await j.stage.run(client, item, record)
            if isinstance(record.result, JudgerResult):
                if record.result.error:
                    record.status = StageStatus.FAILED
                    record.error = record.error or RolloutError(
                        stage=j.name,
                        category="judger",
                        type="JudgerResult",
                        message=record.result.error,
                    )
                return record.result
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=j.name,
                category="judger",
                type="JudgerResult",
                message="no judger_result produced",
            )
            return JudgerResult(
                judger_name=j.name,
                total=0.0,
                error="no judger_result produced",
            )
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=j.name,
                category="judger",
                type=type(exc).__name__,
                message=str(exc),
            )
            result = JudgerResult(
                judger_name=j.name,
                total=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
            record.result = result
            return result

    def _stage_sandbox_name(self, stage: SandboxStage, sandboxes: Mapping[str, Any]) -> str:
        name = stage.sandbox
        if not isinstance(name, str):
            raise TypeError(f"SandboxStage.sandbox must be a sandbox name, got {type(name).__name__}")
        if name not in sandboxes:
            raise KeyError(f"unknown sandbox {name!r}; known sandboxes: {sorted(sandboxes)}")
        return name

    def _aggregate(self, results: list[JudgerResult]) -> JudgerResult:
        weights = {j.name: j.weight for j in self.judgers}
        errors = [r for r in results if r.error]
        usable = [r for r in results if not r.error]

        if errors and self.on_error == "fail":
            total, failed = 0.0, True
        elif not usable:
            total, failed = 0.0, True
        elif self.aggregator == "weighted_sum":
            tw = sum(weights.get(r.judger_name, 1.0) for r in usable)
            total = sum(r.total * weights.get(r.judger_name, 1.0) for r in usable) / tw if tw else 0.0
            failed = False
        elif self.aggregator == "mean":
            total, failed = sum(r.total for r in usable) / len(usable), False
        elif self.aggregator == "max":
            total, failed = max(r.total for r in usable), False
        elif self.aggregator == "min":
            total, failed = min(r.total for r in usable), False
        elif self.aggregator == "all_or_nothing":
            total = 1.0 if all(r.total >= 1.0 for r in usable) else 0.0
            failed = False
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return JudgerResult(
            judger_name="aggregate",
            total=total,
            per_judger=results,
            failed=failed,
        )
