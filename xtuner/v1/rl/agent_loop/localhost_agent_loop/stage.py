"""LocalhostStage: run a local lagent agent."""

from __future__ import annotations

import importlib
import inspect
import time
from copy import deepcopy
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.localhost_agent_loop.schemas import LocalhostAgentSpec
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    SelectedAgentRecord,
    StageRecord,
    StageResult,
    StageStatus,
)
from xtuner.v1.utils import get_logger


class LocalhostStage:
    """Run one local infer stage.

    Local infer does not need a sandbox-style command layer.  The stage owns agent selection, creation, execution, and
    artifact recording directly.
    """

    def __init__(
        self,
        *,
        agents: list[LocalhostAgentSpec | dict[str, Any]],
        name: str = "infer",
    ):
        if not agents:
            raise ValueError("LocalhostStage.agents is empty")
        self.agents = [create_object(agent) if isinstance(agent, dict) else agent for agent in agents]
        self.name = name

    async def run(self, item: AgentRolloutItem, record: StageRecord) -> StageResult:
        record.status = StageStatus.RUNNING
        record.started_at = record.started_at or time.monotonic()
        agent = None
        try:
            spec = self._pick_agent(item, record)
            agent = create_object(deepcopy(_resolve_agent_config(spec.config)))
            output = await agent(item.instruction)
            content = output.content if hasattr(output, "content") else output
            item.artifacts["response"] = content if isinstance(content, str) else str(content)
            messages = agent.get_messages()
            if not isinstance(messages, list) or not messages:
                raise ValueError("Agent messages artifact must be a non-empty list.")
            segment = messages[-1]
            if not isinstance(segment, dict) or "messages" not in segment or "tools" not in segment:
                raise ValueError("Agent messages trace segment must contain messages and tools.")
            if not isinstance(segment["messages"], list):
                raise TypeError("Agent messages trace segment.messages must be a list.")
            item.artifacts["messages"] = messages
            result = StageResult(stdout=item.artifacts["response"], return_code=0)
            record.entry_result = result
            record.status = StageStatus.COMPLETED
            return result
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = record.error or RolloutError(
                stage=self.name,
                category="agent_exception",
                type=type(exc).__name__,
                message=str(exc),
            )
            result = StageResult(return_code=None, error=str(exc), stderr=str(exc))
            record.entry_result = result
            return result
        finally:
            record.finished_at = time.monotonic()
            if agent is not None:
                await _close_agent(agent)

    def _pick_agent(self, item: AgentRolloutItem, record: StageRecord) -> LocalhostAgentSpec:
        group_id = item.group_id or 0
        weights = [max(agent.weight, 0.0) for agent in self.agents]
        total = sum(weights)
        if total <= 0:
            chosen = self.agents[group_id % len(self.agents)]
        else:
            target = (group_id * 2654435761 % 2**32) / 2**32 * total
            running = 0.0
            chosen = self.agents[-1]
            for spec, weight in zip(self.agents, weights):
                running += weight
                if target < running:
                    chosen = spec
                    break

        record.agent = SelectedAgentRecord(
            name=chosen.name,
            config=chosen.config if isinstance(chosen.config, str) else "<inline>",
            weight=chosen.weight,
            template_root="",
        )
        return chosen


def _resolve_agent_config(config: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        if ":" in config:
            module_path, attr = config.split(":", 1)
        else:
            module_path, attr = config, "agent_config"
        module = importlib.import_module(module_path)
        try:
            value = getattr(module, attr)
        except AttributeError as exc:
            raise AttributeError(
                f"LocalhostAgentSpec.config: module {module_path!r} has no attribute {attr!r}"
            ) from exc
        if not isinstance(value, dict):
            raise TypeError(
                f"LocalhostAgentSpec.config: {module_path}:{attr} resolved to {type(value).__name__}, expected dict"
            )
        return value
    raise TypeError(f"LocalhostAgentSpec.config must be str or dict, got {type(config).__name__}")


async def _close_agent(agent: Any) -> None:
    for name in ("aclose", "close"):
        method = getattr(agent, name, None)
        if method is None:
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            get_logger().warning("localhost agent close failed for %s: %s", type(agent).__name__, exc)
        return


__all__ = ["LocalhostStage"]
