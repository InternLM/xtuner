from __future__ import annotations

import asyncio
import copy
import importlib
import json
import traceback
import uuid
from typing import Any, Literal

from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import create_task

from ...rollout.chat_template import canonicalize_messages_for_chat_template
from ...rollout.trace_store import get_store
from ..agent_loop import AgentLoop, AgentLoopConfig
from .schemas import AgentRolloutItem, RolloutStatus


_MISSING = object()


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


def _drop_failed_train_samples(samples: list[RolloutState], mode: Literal["train", "eval"]) -> list[RolloutState]:
    if mode != "train":
        return samples
    filtered = [sample for sample in samples if sample.status != Status.FAILED]
    return filtered or samples


def _validate_trace_segment(
    segment: dict[str, Any],
    *,
    require_tools: bool = False,
) -> tuple[list[dict[str, Any]], Any]:
    if "messages" not in segment:
        raise ValueError("Agent messages trace segment must contain messages.")
    messages = segment["messages"]
    tools = segment.get("tools", _MISSING)
    if not isinstance(messages, list):
        raise TypeError("Agent messages trace must be a list.")
    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("Agent messages trace must contain only dict messages.")
    if require_tools and tools is _MISSING:
        raise ValueError("Agent messages trace segment must contain tools.")
    return messages, None if tools is _MISSING else tools


def _trim_to_last_assistant_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return messages[: index + 1]
    return []


def _load_train_trace_segments(artifacts: dict[str, Any]) -> list[tuple[list[dict[str, Any]], Any]]:
    trace = _load_messages_artifact(artifacts)
    if not trace:
        raise ValueError("Agent artifacts must contain at least one messages trace segment.")
    segments: list[tuple[list[dict[str, Any]], Any]] = []
    for segment in trace:
        if not isinstance(segment, dict):
            raise TypeError("Agent messages trace segment must be a dict.")
        messages, tools = _validate_trace_segment(segment, require_tools=True)
        messages = _trim_to_last_assistant_turn(messages)
        if messages:
            segments.append((messages, tools))
    return segments


def _load_eval_trace_segment(artifacts: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    trace = _load_messages_artifact(artifacts, required=False)
    if not trace:
        return [], None
    segment = trace[-1]
    if not isinstance(segment, dict):
        raise TypeError("Agent messages trace segment must be a dict.")
    messages = segment.get("messages") or []
    tools = segment.get("tools")
    if not isinstance(messages, list):
        raise TypeError("Agent messages trace must be a list.")
    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("Agent messages trace must contain only dict messages.")
    return messages, tools


def _load_messages_artifact(artifacts: dict[str, Any], *, required: bool = True) -> list[dict[str, Any]] | None:
    if "messages" not in artifacts:
        if required:
            raise ValueError("Agent artifacts must contain 'messages'.")
        return None
    trace = artifacts["messages"]
    if not isinstance(trace, list):
        raise TypeError("Agent artifact 'messages' must be a list.")
    return trace


def _response_message(artifacts: dict[str, Any], *, required: bool) -> dict[str, Any]:
    if "response_message" not in artifacts:
        if required:
            raise KeyError("Agent artifacts must contain 'response_message'.")
        return {}
    response_message = artifacts["response_message"]
    if not isinstance(response_message, dict):
        raise TypeError("Agent artifact 'response_message' must be a dict.")
    return response_message


def _response_text(response_message: dict[str, Any]) -> str:
    content = response_message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError("Agent response_message['content'] must be a string.")
    return content


def _finish_info(response_message: dict[str, Any]) -> dict[str, Any] | None:
    finish_info = response_message.get("finish_info")
    if finish_info is None:
        return None
    if not isinstance(finish_info, dict):
        raise TypeError("Agent response_message['finish_info'] must be a dict.")
    return finish_info or None


def _to_json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _selected_agent(item: AgentRolloutItem) -> dict[str, Any] | None:
    if item.infer.agent is None:
        return None
    return item.infer.agent.model_dump(mode="json")


def _count_tool_turns(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list) and message["tool_calls"]
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


class AgentInSandboxLoopConfig(AgentLoopConfig):
    """Run a sandbox agent runner from ``RolloutState.extra_fields``.

    The tb2-rl tokenize function stores an :class:`AgentRolloutItem` in
    ``rollout_state.extra_fields["rollout_item"]``.  This loop executes that
    item's sandbox pipeline, then converts the resulting task reward and agent
    transcript back into the standard ``RolloutState`` fields consumed by the
    replay buffer/trainer.
    """

    max_concurrent_samples: int | None = None
    mode: Literal["train", "eval"] = "train"
    requires_rollout_proxy: bool = True

    def build_local(
        self, rollout_controller: RolloutController | None = None, judger: Judger | None = None, logger=None
    ) -> AgentInSandboxLoop:
        return AgentInSandboxLoop(
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
            logger=logger,
            max_concurrent_samples=self.max_concurrent_samples,
            mode=self.mode,
        )


class AgentInSandboxLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController | None = None,
        hf_checkpoint: str | None = None,
        sample_params: SampleParams | None = None,
        judger: Judger | None = None,
        logger=None,
        max_concurrent_samples: int | None = None,
        mode: Literal["train", "eval"] = "train",
    ):
        if hf_checkpoint is None:
            raise ValueError("hf_checkpoint must be provided for AgentInSandboxLoop.")
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.max_concurrent_samples = max_concurrent_samples
        self._sample_semaphore = asyncio.Semaphore(max_concurrent_samples) if max_concurrent_samples else None
        self.mode = mode

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        async def generate_one(state: RolloutState) -> list[RolloutState]:
            if self._sample_semaphore is None:
                return await self.generate_sample(state)
            async with self._sample_semaphore:
                return await self.generate_sample(state)

        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(generate_one(state))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        sample_groups = await generated_samples
        samples = [sample for sample_group in sample_groups for sample in sample_group]
        return _drop_failed_train_samples(samples, self.mode)

    # NOTE: A single sandbox session may yield multiple trainable segments, so this returns a list
    # rather than the base class's single RolloutState. The base contract is never exercised for
    # this loop (generate_group is the only entry point), so we override the return type here.
    async def generate_sample(  # type: ignore[override]
        self, rollout_state: RolloutState, **kwargs
    ) -> list[RolloutState]:
        try:
            rollout_item = rollout_state.extra_fields["rollout_item"].model_copy(deep=True)
            if rollout_state.session_id is None:
                rollout_state.session_id = uuid.uuid4().int
            rollout_item.uid = rollout_state.session_id
            rollout_item.group_id = rollout_state.group_id
            result = await self._run_item(rollout_item)
            return await self._build_rollout_states(rollout_state, result)
        except Exception as exc:
            rollout_state.status = Status.COMPLETED if self.mode == "eval" else Status.FAILED
            rollout_state.finish_reason = "error"
            if self.mode == "eval":
                rollout_state.reward = {"score": 0.0}
                rollout_state.response = ""
                rollout_state.extra_fields["agent_status"] = "exception"
            rollout_state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[AgentInSandboxLoop] failed: {exc}\n{traceback.format_exc()}")
            return [rollout_state]

    async def _run_item(self, item: AgentRolloutItem) -> AgentRolloutItem:
        runner = _resolve_runner(item.pipeline, str(item.uid))
        if runner is None:
            raise ValueError("AgentRolloutItem.pipeline is required.")
        return await runner.run(item)

    async def _build_rollout_states(self, rollout_state: RolloutState, item: AgentRolloutItem) -> list[RolloutState]:
        if self.mode == "eval":
            self._fill_eval_rollout_state(rollout_state, item)
            return [rollout_state]

        response_message = _response_message(item.artifacts, required=item.status == RolloutStatus.COMPLETED)
        rollout_state.status = Status.COMPLETED if item.status == RolloutStatus.COMPLETED else Status.FAILED
        rollout_state.finish_reason = str(
            response_message.get("finish_reason") or ("stop" if item.status == RolloutStatus.COMPLETED else "error")
        )
        rollout_state.reward = _extract_reward_payload(item)
        rollout_state.extra_fields["agent_status"] = item.status.value
        selected_agent = _selected_agent(item)
        if selected_agent is not None:
            rollout_state.extra_fields["agent_name"] = selected_agent.get("name")
            rollout_state.extra_fields["agent_selected"] = _to_json_safe(selected_agent)
        rollout_state.extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
        rollout_state.extra_fields["agent_judgers"] = {
            name: record.model_dump(mode="json") for name, record in item.judgers.items()
        }
        finish_info = _finish_info(response_message)
        if finish_info:
            rollout_state.extra_fields["agent_finish_info"] = finish_info
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"
        if item.status != RolloutStatus.COMPLETED:
            return [rollout_state]

        segments = _load_train_trace_segments(item.artifacts)
        if not segments:
            raise ValueError("Agent artifacts must contain at least one trainable messages trace.")

        rollout_states: list[RolloutState] = []
        trace_store = get_store()
        for segment_index, (messages, tools) in enumerate(segments):
            if not messages:
                raise ValueError("Agent artifacts must contain at least one trainable messages trace.")
            segment_state = rollout_state.model_copy(deep=True)
            segment_state.extra_fields["agent_messages"] = messages
            segment_state.extra_fields["agent_tools"] = tools
            segment_state.extra_fields["agent_tool_turns"] = _count_tool_turns(messages)
            segment_state.extra_fields["agent_trace_segment_index"] = segment_index
            segment_state.extra_fields["agent_trace_segment_count"] = len(segments)
            segment_state.extra_fields["agent_session_id"] = rollout_state.session_id

            text = self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(messages),
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = text[:-1] if text.endswith("\n") else text
            data = await trace_store.export_training_trace.remote(str(rollout_state.session_id), prompt_text)
            segment_state.input_ids = data["input_ids"]
            segment_state.labels = data["labels"]
            # Agentic training consumes input_ids/labels directly. response_ids is
            # filled here only so rollout throughput logging can print rollout_tgs.
            segment_state.response_ids = [
                token_id for token_id, label in zip(data["input_ids"][1:], data["labels"][1:]) if label != -100
            ]
            segment_state.logprobs = data["logprobs"]
            segment_state.routed_experts = data["routed_experts"]
            if segment_state.response_ids:
                segment_state.response = self.tokenizer.decode(segment_state.response_ids)
            else:
                segment_state.response = _response_text(response_message)
            rollout_states.append(segment_state)

        return rollout_states

    def _fill_eval_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        is_success = item.status == RolloutStatus.COMPLETED
        response_message = _response_message(item.artifacts, required=is_success)
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
        selected_agent = _selected_agent(item)
        if selected_agent is not None:
            rollout_state.extra_fields["agent_name"] = selected_agent.get("name")
            rollout_state.extra_fields["agent_selected"] = _to_json_safe(selected_agent)
        rollout_state.extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
        rollout_state.extra_fields["agent_judgers"] = {
            name: record.model_dump(mode="json") for name, record in item.judgers.items()
        }
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"

        rollout_state.response = _response_text(response_message)
        finish_info = _finish_info(response_message)
        if finish_info:
            rollout_state.extra_fields["agent_finish_info"] = finish_info
        if not is_success:
            return

        messages, tools = _load_eval_trace_segment(item.artifacts)
        if messages:
            rollout_state.extra_fields["agent_messages"] = messages
            rollout_state.extra_fields["agent_tools"] = tools
            rollout_state.extra_fields["agent_tool_turns"] = _count_tool_turns(messages)
