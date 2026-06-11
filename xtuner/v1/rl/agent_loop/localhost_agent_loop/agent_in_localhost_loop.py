from __future__ import annotations

import asyncio
import copy
import importlib
import traceback
import uuid
from typing import Any, Literal

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


def _load_eval_trace_segment(artifacts: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    trace = artifacts.get("messages") or []
    if not isinstance(trace, list) or not trace:
        return [], None
    segment = trace[-1]
    if not isinstance(segment, dict) or "messages" not in segment:
        return [], None
    messages = segment.get("messages") or []
    if not isinstance(messages, list):
        return [], None
    if not all(isinstance(message, dict) for message in messages):
        return [], None
    return messages, segment.get("tools")


def _count_tool_turns(messages: list[dict[str, Any]]) -> int:
    """Number of assistant turns that emitted at least one tool_call."""
    return sum(
        1 for m in messages if isinstance(m, dict) and isinstance(m.get("tool_calls"), list) and m["tool_calls"]
    )


def _extract_reward_payload(item: AgentRolloutItem) -> dict[str, Any] | None:
    for record in item.judgers.values():
        reward = record.metadata.get("reward")
        if isinstance(reward, dict):
            payload = dict(reward)
            if item.reward is not None:
                payload.setdefault("score", item.reward)
            return payload
    if item.reward is not None:
        return {"score": item.reward}
    return None


class AgentInLocalhostLoopConfig(AgentLoopConfig):
    """Run a localhost agent runner from ``RolloutState.extra_fields``."""

    max_concurrent_samples: int | None = None
    sample_timeout_s: float | None = None
    mode: Literal["train", "eval"] = "train"

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
            sample_timeout_s=self.sample_timeout_s,
            mode=self.mode,
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
        sample_timeout_s: float | None = None,
        mode: Literal["train", "eval"] = "train",
    ):
        if hf_checkpoint is None:
            raise ValueError("hf_checkpoint must be provided for AgentInLocalhostLoop.")
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.max_concurrent_samples = max_concurrent_samples
        self.sample_timeout_s = sample_timeout_s
        self._sample_semaphore = asyncio.Semaphore(max_concurrent_samples) if max_concurrent_samples else None
        self.mode = mode

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        async def generate_one(state: RolloutState) -> RolloutState:
            if self._sample_semaphore is None:
                return await self.generate_sample(state, **kwargs)
            async with self._sample_semaphore:
                return await self.generate_sample(state, **kwargs)

        tasks: list[asyncio.Task[RolloutState]] = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(generate_one(state))
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        try:
            if self.sample_timeout_s is not None and self.sample_timeout_s > 0:
                return await asyncio.wait_for(
                    self._generate_sample_impl(rollout_state),
                    timeout=self.sample_timeout_s,
                )
            return await self._generate_sample_impl(rollout_state)
        except asyncio.TimeoutError:
            self.logger.warning(
                f"[AgentInLocalhostLoop] sample timed out after {self.sample_timeout_s:.1f}s "
                f"(uid={rollout_state.uid}, group_id={rollout_state.message_uid})."
            )
            return self._fail_rollout_state(
                rollout_state,
                finish_reason="timeout",
                error_msg=f"TimeoutError: localhost agent sample exceeded {self.sample_timeout_s:.1f}s",
                agent_status="timeout",
            )
        except Exception as exc:
            if self.mode == "train" and _is_trace_key_mismatch(exc):
                raise
            self.logger.error(f"[AgentInLocalhostLoop] failed: {exc}\n{traceback.format_exc()}")
            return self._fail_rollout_state(
                rollout_state,
                finish_reason="error",
                error_msg=f"{type(exc).__name__}: {exc}",
                agent_status="exception",
            )

    async def _generate_sample_impl(self, rollout_state: RolloutState) -> RolloutState:
        item = rollout_state.extra_fields["rollout_item"].model_copy(deep=True)
        if rollout_state.uid is None:
            rollout_state.uid = uuid.uuid4().int
        item.uid = rollout_state.uid
        item.group_id = rollout_state.message_uid
        result = await self._run_item(item)
        await self._fill_rollout_state(rollout_state, result)
        return rollout_state

    def _fail_rollout_state(
        self,
        rollout_state: RolloutState,
        *,
        finish_reason: str,
        error_msg: str,
        agent_status: str,
    ) -> RolloutState:
        if rollout_state.uid is None:
            rollout_state.uid = uuid.uuid4().int
        rollout_state.status = Status.COMPLETED if self.mode == "eval" else Status.FAILED
        rollout_state.finish_reason = finish_reason
        if self.mode == "eval":
            rollout_state.reward = {"score": 0.0}
            rollout_state.response = ""
            rollout_state.extra_fields["agent_status"] = agent_status
        rollout_state.error_msg = error_msg
        return rollout_state

    async def _run_item(self, item: AgentRolloutItem) -> AgentRolloutItem:
        runner = _resolve_runner(item.pipeline)
        if runner is None:
            raise ValueError("AgentRolloutItem.pipeline is required.")
        with ctx_session_id.set(str(item.uid)):
            return await runner.run(item)

    async def _fill_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        if self.mode == "eval":
            self._fill_eval_rollout_state(rollout_state, item)
            return

        response_message = item.artifacts.get("response_message") or {}
        rollout_state.status = Status.COMPLETED if item.status == RolloutStatus.COMPLETED else Status.FAILED
        rollout_state.finish_reason = str(
            response_message.get("finish_reason") or ("stop" if item.status == RolloutStatus.COMPLETED else "error")
        )
        rollout_state.reward = _extract_reward_payload(item)
        rollout_state.extra_fields["agent_status"] = item.status.value
        rollout_state.extra_fields["agent_artifacts"] = item.artifacts
        rollout_state.extra_fields["agent_judgers"] = {
            name: record.model_dump(mode="json") for name, record in item.judgers.items()
        }
        messages, tools = _load_eval_trace_segment(item.artifacts)
        if messages:
            rollout_state.extra_fields["agent_messages"] = messages
            rollout_state.extra_fields["agent_tools"] = tools
            rollout_state.extra_fields["agent_tool_turns"] = _count_tool_turns(messages)
        finish_info = response_message.get("finish_info")
        if isinstance(finish_info, dict) and finish_info:
            rollout_state.extra_fields["agent_finish_info"] = finish_info
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
        content = response_message.get("content")
        rollout_state.response = content if isinstance(content, str) else (str(content) if content is not None else "")

    def _fill_eval_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        is_success = item.status == RolloutStatus.COMPLETED
        response_message = item.artifacts.get("response_message") or {}
        rollout_state.status = Status.COMPLETED
        rollout_state.finish_reason = str(response_message.get("finish_reason") or ("stop" if is_success else "error"))
        rollout_state.reward = {"score": item.reward if is_success and item.reward is not None else 0.0}
        rollout_state.input_ids = None
        rollout_state.labels = None
        rollout_state.response_ids = None
        rollout_state.logprobs = None
        rollout_state.routed_experts = None
        rollout_state.response_mask = None
        rollout_state.response_model_steps = None
        rollout_state.extra_fields["agent_status"] = item.status.value
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"

        messages, tools = _load_eval_trace_segment(item.artifacts)
        content = response_message.get("content")
        rollout_state.response = content if isinstance(content, str) else (str(content) if content is not None else "")
        if messages:
            rollout_state.extra_fields["agent_messages"] = messages
            rollout_state.extra_fields["agent_tools"] = tools
            rollout_state.extra_fields["agent_tool_turns"] = _count_tool_turns(messages)
        finish_info = response_message.get("finish_info")
        if isinstance(finish_info, dict) and finish_info:
            rollout_state.extra_fields["agent_finish_info"] = finish_info


__all__ = ["AgentInLocalhostLoop", "AgentInLocalhostLoopConfig"]
