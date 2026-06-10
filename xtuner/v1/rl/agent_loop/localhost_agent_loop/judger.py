"""Localhost judger stages."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    StageRecord,
    StageStatus,
)
from xtuner.v1.rl.judger.native import Judger


class LocalhostJudgerStage:
    """Run one local validation stage.

    Public stage interface is ``run(item, record) -> float``.  ``RolloutState``
    is only the internal shape needed to reuse xtuner judgers.
    """

    def __init__(
        self,
        *,
        name: str,
        judger_config: Any,
        reward_key: str = "score",
        weight: float = 1.0,
    ):
        config = create_object(deepcopy(judger_config)) if isinstance(judger_config, dict) else judger_config
        self.name = name
        self.judger: Judger = config.build()
        self.reward_key = reward_key
        self.weight = weight

    async def run(self, item: AgentRolloutItem, record: StageRecord) -> float:
        record.status = StageStatus.RUNNING
        record.started_at = record.started_at or time.monotonic()
        try:
            # reward_model stays as-is (dataset-provided ground_truth/style etc.).
            # Per-rollout artifacts (response message, agent trace) flow through extra_fields.
            reward_model = dict(item.reward_model or {})
            segment = item.artifacts["messages"][-1]
            response_message = item.artifacts.get("response_message") or {}
            content = response_message.get("content")
            response = content if isinstance(content, str) else (str(content) if content is not None else "")

            rollout_state = RolloutState(
                message=[{"role": "user", "content": item.instruction}],
                response=response,
                reward_model=reward_model,
                extra_fields={
                    "agent_messages": segment["messages"],
                    "response_message": response_message,
                },
                status=Status.COMPLETED,
            )
            judged = await self.judger.judge(rollout_state)
            reward_payload = judged.reward or {}
            if self.reward_key not in reward_payload:
                raise KeyError(f"judger reward payload has no {self.reward_key!r}: {reward_payload!r}")
            record.metadata["reward"] = reward_payload
            record.score = float(reward_payload[self.reward_key])
            record.status = StageStatus.COMPLETED
            return record.score
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=self.name,
                category="judger",
                type=type(exc).__name__,
                message=str(exc),
            )
            raise
        finally:
            record.finished_at = time.monotonic()


__all__ = ["LocalhostJudgerStage"]
