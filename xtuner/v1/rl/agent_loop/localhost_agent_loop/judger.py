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
        judger_config: Any | None = None,
        judger: Any | None = None,
        reward_key: str = "score",
        weight: float = 1.0,
    ):
        if judger_config is None:
            judger_config = judger
        if judger_config is None:
            raise ValueError("LocalhostJudgerStage requires judger_config")
        self.name = name
        self._judger_config = judger_config
        self._judger: Judger | Any | None = None
        self.reward_key = reward_key
        self.weight = weight

    async def run(self, item: AgentRolloutItem, record: StageRecord) -> float:
        record.status = StageStatus.RUNNING
        record.started_at = record.started_at or time.monotonic()
        try:
            reward_model = dict(item.reward_model or {})

            messages_artifact = item.artifacts.get("messages")
            if messages_artifact is not None:
                if not isinstance(messages_artifact, list) or not messages_artifact:
                    raise ValueError("Agent messages artifact must be a non-empty list.")
                segment = messages_artifact[-1]
                if not isinstance(segment, dict) or "messages" not in segment or "tools" not in segment:
                    raise ValueError("Agent messages trace segment must contain messages and tools.")
                messages = segment["messages"]
                if not isinstance(messages, list):
                    raise TypeError("Agent messages trace segment.messages must be a list.")
                tool_turns = sum(
                    1
                    for message in messages
                    if isinstance(message, dict)
                    and isinstance(message.get("tool_calls"), list)
                    and message["tool_calls"]
                )
                reward_model.setdefault("agent_trace", messages)
                reward_model.setdefault("num_turns", tool_turns)

            response = str(item.artifacts.get("response") or "")
            rollout_state = RolloutState(
                message=[{"role": "user", "content": item.instruction}],
                response=response,
                reward_model=reward_model,
                status=Status.COMPLETED if item.infer.status == StageStatus.COMPLETED else Status.FAILED,
            )
            judged = await self.build().judge(rollout_state)
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

    def build(self) -> Judger | Any:
        if self._judger is None:
            if isinstance(self._judger_config, dict):
                config = create_object(deepcopy(self._judger_config))
            else:
                config = self._judger_config
            if hasattr(config, "build"):
                self._judger = config.build()
            elif hasattr(config, "judge"):
                self._judger = config
            else:
                raise TypeError(
                    f"judger_config must build a Judger or be a Judger-like object, got {type(config).__name__}"
                )
        return self._judger


__all__ = ["LocalhostJudgerStage"]
