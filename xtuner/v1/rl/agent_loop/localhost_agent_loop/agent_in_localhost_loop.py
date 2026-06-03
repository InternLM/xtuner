from __future__ import annotations

import asyncio
import copy
import importlib
import traceback
import uuid
from typing import Any

from lagent.utils import create_object, ctx_session_id

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutStatus,
)
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template
from xtuner.v1.rl.rollout.trace_store import get_store
from xtuner.v1.rl.utils import create_task

from ..agent_loop import AgentLoop, AgentLoopConfig


def _import_from_path(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path: {path!r}. Expected 'module.attr'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_runner(pipeline: Any) -> Any:
    if isinstance(pipeline, str):
        pipeline = _import_from_path(pipeline)
    if isinstance(pipeline, dict):
        return create_object(copy.deepcopy(pipeline))
    return pipeline


def _is_trace_key_mismatch(exc: Exception) -> bool:
    return "does not match any trace key" in str(exc)


class AgentInLocalhostLoopConfig(AgentLoopConfig):
    """Run a localhost agent runner from ``RolloutState.extra_fields``."""

    max_concurrent_samples: int | None = None

    def build_local(
        self,
        rollout_controller: RolloutController | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> AgentInLocalhostLoop:
        return AgentInLocalhostLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            max_concurrent_samples=self.max_concurrent_samples,
        )


class AgentInLocalhostLoop(AgentLoop):
    """AgentLoop adapter for localhost_agent_loop runners."""

    def __init__(
        self,
        rollout_ctl: RolloutController | None = None,
        sample_params: SampleParams | None = None,
        hf_checkpoint: str | None = None,
        judger: Judger | None = None,
        logger=None,
        max_concurrent_samples: int | None = None,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.max_concurrent_samples = max_concurrent_samples
        self._sample_semaphore = asyncio.Semaphore(max_concurrent_samples) if max_concurrent_samples else None

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        async def generate_one(state: RolloutState) -> RolloutState:
            if self._sample_semaphore is None:
                return await self.generate_sample(state, **kwargs)
            async with self._sample_semaphore:
                return await self.generate_sample(state, **kwargs)

        tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            tasks.append(create_task(generate_one(state)))
        return await asyncio.gather(*tasks)

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        try:
            item = rollout_state.extra_fields["rollout_item"].model_copy(deep=True)
            if rollout_state.uid is None:
                rollout_state.uid = uuid.uuid4().int
            item.uid = rollout_state.uid
            item.group_id = rollout_state.message_uid
            result = await self._run_item(item)
            await self._fill_rollout_state(rollout_state, result)
            return rollout_state
        except Exception as exc:
            if _is_trace_key_mismatch(exc):
                raise
            rollout_state.status = Status.FAILED
            rollout_state.finish_reason = "error"
            rollout_state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[AgentInLocalhostLoop] failed: {exc}\n{traceback.format_exc()}")
            return rollout_state

    async def _run_item(self, item: AgentRolloutItem) -> AgentRolloutItem:
        runner = _resolve_runner(item.pipeline)
        if runner is None:
            raise ValueError("AgentRolloutItem.pipeline is required.")
        with ctx_session_id.set(str(item.uid)):
            return await runner.run(item)

    async def _fill_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        rollout_state.status = Status.COMPLETED if item.status == RolloutStatus.COMPLETED else Status.FAILED
        rollout_state.finish_reason = "stop" if item.status == RolloutStatus.COMPLETED else "error"
        rollout_state.reward = {"score": item.reward} if item.reward is not None else None
        rollout_state.extra_fields["agent_artifacts"] = item.artifacts
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"
        if item.status != RolloutStatus.COMPLETED:
            return

        segment = item.artifacts["messages"][-1]
        text = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(segment["messages"]),
            tools=segment["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = text[:-1] if text.endswith("\n") else text
        data = await get_store().export_training_trace.remote(str(rollout_state.uid), prompt_text)

        rollout_state.input_ids = data["input_ids"]
        rollout_state.labels = data["labels"]
        rollout_state.response_ids = [
            token_id for token_id, label in zip(data["input_ids"][1:], data["labels"][1:]) if label != -100
        ]
        rollout_state.logprobs = data["logprobs"]
        rollout_state.routed_experts = data["routed_experts"]
        rollout_state.response = str(item.artifacts.get("response") or "")
        rollout_state.extra_fields["raw_prompt"] = prompt_text


__all__ = ["AgentInLocalhostLoop", "AgentInLocalhostLoopConfig"]
