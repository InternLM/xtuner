from __future__ import annotations

import asyncio
import copy
import importlib
import json
import traceback
import uuid
from typing import Any

from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import create_task

from ...rollout.trace_store import get_store
from ..agent_loop import AgentLoop, AgentLoopConfig
from .schemas import AgentRolloutItem, RolloutStatus


def _import_from_path(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path: {path!r}. Expected 'module.attr'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _inject_session_id(runner_cfg: dict[str, Any], session_id: str) -> None:
    for entry in runner_cfg.get("infer", {}).get("entries", []):
        if isinstance(entry, dict) and entry.get("name") == "start_agent_daemon":
            entry.setdefault("env", {})["XTUNER_SESSION_ID"] = session_id


def _resolve_runner(pipeline: Any, session_id: str) -> Any:
    if isinstance(pipeline, str):
        pipeline = _import_from_path(pipeline)
    if isinstance(pipeline, dict):
        runner_cfg = copy.deepcopy(pipeline)
        _inject_session_id(runner_cfg, session_id)
        return create_object(runner_cfg)
    return pipeline


class AgentInSandboxLoopConfig(AgentLoopConfig):
    """Run a sandbox agent runner from ``RolloutState.extra_fields``.

    The tb2-rl tokenize function stores an :class:`AgentRolloutItem` in
    ``rollout_state.extra_fields["rollout_item"]``.  This loop executes that
    item's sandbox pipeline, then converts the resulting task reward and agent
    transcript back into the standard ``RolloutState`` fields consumed by the
    replay buffer/trainer.
    """

    max_concurrent_samples: int | None = None

    def build_local(
        self, rollout_controller: RolloutController | None = None, judger: Judger | None = None, logger=None
    ) -> AgentInSandboxLoop:
        return AgentInSandboxLoop(
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            max_concurrent_samples=self.max_concurrent_samples,
        )


class AgentInSandboxLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController | None = None,
        hf_checkpoint: str = None,
        judger: Judger | None = None,
        logger=None,
        max_concurrent_samples: int | None = None,
    ):
        super().__init__(rollout_ctl, None, hf_checkpoint, judger, logger)
        self.max_concurrent_samples = max_concurrent_samples
        self._sample_semaphore = asyncio.Semaphore(max_concurrent_samples) if max_concurrent_samples else None

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        async def generate_one(state: RolloutState) -> RolloutState:
            if self._sample_semaphore is None:
                return await self.generate_sample(state, **kwargs)
            async with self._sample_semaphore:
                return await self.generate_sample(state, **kwargs)

        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(generate_one(state))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        try:
            rollout_item = rollout_state.extra_fields["rollout_item"].model_copy(deep=True)
            if rollout_state.uid is None:
                rollout_state.uid = uuid.uuid4()
            rollout_item.uid = rollout_state.uid
            rollout_item.group_id = rollout_state.message_uid
            result = await self._run_item(rollout_item)
            await self._fill_rollout_state(rollout_state, result)
            return rollout_state
        except Exception as exc:
            rollout_state.status = Status.FAILED
            rollout_state.finish_reason = "error"
            rollout_state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[AgentInSandboxLoop] failed: {exc}\n{traceback.format_exc()}")
            return rollout_state

    async def _run_item(self, item: AgentRolloutItem) -> AgentRolloutItem:
        runner = _resolve_runner(item.pipeline, str(item.uid))
        if runner is None:
            raise ValueError("AgentRolloutItem.pipeline is required.")
        return await runner.run(item)

    async def _fill_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        rollout_state.status = Status.COMPLETED if item.status == RolloutStatus.COMPLETED else Status.FAILED
        rollout_state.finish_reason = "stop" if item.status == RolloutStatus.COMPLETED else "error"
        rollout_state.reward = {"score": item.reward} if item.reward is not None else None
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"
        if item.status != RolloutStatus.COMPLETED:
            return

        artifacts = item.artifacts
        trace = json.loads(artifacts["message"])
        if not isinstance(trace, list) or not trace:
            raise ValueError("Agent artifacts must contain at least one trainable messages trace.")
        segment = trace[-1]
        if not isinstance(segment, dict) or "messages" not in segment or "tools" not in segment:
            raise ValueError("Agent messages trace segment must contain messages and tools.")
        messages = segment["messages"]
        if not isinstance(messages, list):
            raise TypeError("Agent messages trace segment.messages must be a list.")
        session_id = rollout_state.uid

        trace_store = get_store()
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=segment["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = text[:-1] if text.endswith("\n") else text
        data = await trace_store.export_training_trace.remote(str(session_id), prompt_text)

        rollout_state.input_ids = data["input_ids"]
        rollout_state.labels = data["labels"]
        # Agentic training consumes input_ids/labels directly. response_ids is
        # filled here only so rollout throughput logging can print rollout_tgs.
        rollout_state.response_ids = [
            token_id for token_id, label in zip(data["input_ids"][1:], data["labels"][1:]) if label != -100
        ]
        rollout_state.logprobs = data["logprobs"]
        rollout_state.routed_experts = data["routed_experts"]
