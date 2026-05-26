"""LocalhostRunner: orchestrates local infer and optional validation."""

from __future__ import annotations

import time
import traceback
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.localhost_agent_loop.stage import LocalhostStage
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    RolloutStatus,
    StageRecord,
    StageStatus,
)
from xtuner.v1.utils import get_logger


class LocalhostRunner:
    """Drive one rollout sample on a lagent agent running on this machine."""

    def __init__(
        self,
        infer: LocalhostStage | dict[str, Any],
        validate: Any | dict[str, Any] | None = None,
    ):
        self.infer: LocalhostStage = create_object(infer) if isinstance(infer, dict) else infer
        self.validate = create_object(validate) if isinstance(validate, dict) else validate

    async def run(self, item: AgentRolloutItem) -> AgentRolloutItem:
        if not item.instruction:
            raise ValueError("AgentRolloutItem.instruction is required by LocalhostRunner.run")
        item.status = RolloutStatus.RUNNING
        tid = item.id
        t_validate: float | None = None

        try:
            infer_result = await self.infer.run(item, item.infer)
            if not infer_result.ok:
                return self._fail(item, item.infer.error)

            if self.validate is not None:
                t0 = time.monotonic()
                validate_name = getattr(self.validate, "name", "validate")
                validate_record = item.judgers.setdefault(
                    validate_name,
                    StageRecord(),
                )
                try:
                    score = float(await self.validate.run(item, validate_record))
                except Exception:
                    return self._fail(
                        item,
                        validate_record.error
                        or _first_judger_error(item)
                        or RolloutError(
                            stage=validate_name,
                            category="validate_failed",
                            type=type(self.validate).__name__,
                            message="validate failed",
                        ),
                    )
                t_validate = time.monotonic() - t0
                item.reward = score

            item.status = RolloutStatus.COMPLETED
            return item
        except Exception as exc:
            promoted = (
                item.infer.error
                or _first_judger_error(item)
                or RolloutError(
                    stage="runner",
                    category="runner_exception",
                    type=type(exc).__name__,
                    message=str(exc),
                )
            )
            get_logger().error(f"[{tid}] traceback:\n{traceback.format_exc()}")
            return self._fail(item, promoted)
        finally:
            self._log_final(tid, item, t_validate)

    # -- internals --

    def _fail(self, item: AgentRolloutItem, error: RolloutError | None) -> AgentRolloutItem:
        item.status = RolloutStatus.FAILED
        if item.error is None:
            item.error = error
        if item.infer.status == StageStatus.RUNNING and item.infer.error is None:
            item.infer.status = StageStatus.FAILED
            item.infer.error = error
        if error is not None:
            get_logger().error(f"[{item.id}] failed: {error.category}: {error.message}")
        else:
            get_logger().error(f"[{item.id}] failed: unknown error")
        return item

    def _log_final(self, tid: str, item: AgentRolloutItem, t_validate: float | None) -> None:
        agent_name = item.infer.agent.name if item.infer.agent is not None else "?"
        parts = [f"status={item.status.value}", f"agent={agent_name}"]
        if item.reward is not None:
            parts.append(f"reward={item.reward:.4f}")
        if item.infer.started_at and item.infer.finished_at:
            parts.append(f"t_infer={item.infer.finished_at - item.infer.started_at:.1f}s")
        if t_validate is not None:
            parts.append(f"t_validate={t_validate:.1f}s")
        if item.status == RolloutStatus.FAILED and item.error:
            parts.append(f"error={item.error.stage or '?'}/{item.error.category}")
        get_logger().info(f"[{tid}] done {' '.join(parts)}")


def _first_judger_error(item: AgentRolloutItem) -> RolloutError | None:
    for record in item.judgers.values():
        if record.error is not None:
            return record.error
    return None


__all__ = ["LocalhostRunner"]
