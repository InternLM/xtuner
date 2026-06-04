from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import time
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
_DEFAULT_SANDBOX_CREATES_PER_SEC = 3.0
_DEFAULT_SANDBOX_CREATES_BURST = 8


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


def _env_float(names: tuple[str, ...], default: float | None = None) -> float | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return float(value)
    return default


def _env_int(names: tuple[str, ...], default: int | None = None) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return int(value)
    return default


class _TokenBucket:
    """Pace sandbox rollout starts from the long-lived agent loop."""

    def __init__(self, rate_per_sec: float, capacity: int):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._rate = float(rate_per_sec)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
            self._last = now + wait
        await asyncio.sleep(wait)


def _resolve_runner(pipeline: Any, session_id: str) -> Any:
    if isinstance(pipeline, str):
        pipeline = _import_from_path(pipeline)
    if isinstance(pipeline, dict):
        runner_cfg = copy.deepcopy(pipeline)
        _inject_session_id(runner_cfg, session_id)
        return create_object(runner_cfg)
    return pipeline


def _load_latest_trace_segment(
    artifacts: dict[str, Any], *, require_tools: bool = False
) -> tuple[list[dict[str, Any]], Any]:
    raw_message = artifacts.get("message")
    if raw_message is None:
        return [], None
    trace = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
    if isinstance(trace, list) and trace:
        segment = trace[-1]
        if not isinstance(segment, dict) or "messages" not in segment:
            raise ValueError("Agent messages trace segment must contain messages.")
        messages = segment["messages"]
        tools = segment.get("tools", _MISSING)
    elif isinstance(trace, dict):
        messages = trace.get("messages")
        tools = trace.get("tools", _MISSING)
    else:
        raise ValueError("Agent artifacts must contain a messages trace.")
    if not isinstance(messages, list):
        raise TypeError("Agent messages trace must be a list.")
    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("Agent messages trace must contain only dict messages.")
    if require_tools and tools is _MISSING:
        raise ValueError("Agent messages trace segment must contain tools.")
    return messages, None if tools is _MISSING else tools


def _load_eval_trace_segment(artifacts: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    raw_message = artifacts.get("message")
    if raw_message is None:
        return [], None
    try:
        trace = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
    except json.JSONDecodeError:
        return [], None
    if isinstance(trace, list) and trace:
        segment = trace[-1]
        if not isinstance(segment, dict):
            return [], None
        messages = segment.get("messages") or []
        tools = segment.get("tools")
    elif isinstance(trace, dict):
        messages = trace.get("messages") or []
        tools = trace.get("tools")
    else:
        return [], None
    if not isinstance(messages, list):
        return [], None
    if not all(isinstance(message, dict) for message in messages):
        return [], None
    return messages, tools


def _to_json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class AgentInSandboxLoopConfig(AgentLoopConfig):
    """Run a sandbox agent runner from ``RolloutState.extra_fields``.

    The tb2-rl tokenize function stores an :class:`AgentRolloutItem` in
    ``rollout_state.extra_fields["rollout_item"]``.  This loop executes that
    item's sandbox pipeline, then converts the resulting task reward and agent
    transcript back into the standard ``RolloutState`` fields consumed by the
    replay buffer/trainer.
    """

    max_concurrent_samples: int | None = None
    sandbox_creates_per_sec: float | None = None
    sandbox_creates_burst: int | None = None
    mode: Literal["train", "eval"] = "train"

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
            sandbox_creates_per_sec=self.sandbox_creates_per_sec,
            sandbox_creates_burst=self.sandbox_creates_burst,
            mode=self.mode,
        )


class AgentInSandboxLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController | None = None,
        hf_checkpoint: str = None,
        sample_params: SampleParams | None = None,
        judger: Judger | None = None,
        logger=None,
        max_concurrent_samples: int | None = None,
        sandbox_creates_per_sec: float | None = None,
        sandbox_creates_burst: int | None = None,
        mode: Literal["train", "eval"] = "train",
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.max_concurrent_samples = max_concurrent_samples
        self._sample_semaphore = asyncio.Semaphore(max_concurrent_samples) if max_concurrent_samples else None
        self.sandbox_creates_per_sec = self._resolve_sandbox_creates_per_sec(sandbox_creates_per_sec)
        self.sandbox_creates_burst = (
            self._resolve_sandbox_creates_burst(sandbox_creates_burst)
            if self.sandbox_creates_per_sec is not None
            else None
        )
        self._sandbox_create_limiter = (
            _TokenBucket(self.sandbox_creates_per_sec, self.sandbox_creates_burst)
            if self.sandbox_creates_per_sec is not None and self.sandbox_creates_per_sec > 0
            else None
        )
        if self._sandbox_create_limiter is not None:
            self.logger.info(
                "[AgentInSandboxLoop] sandbox rollout start rate limit: "
                f"{self.sandbox_creates_per_sec:.2f}/s burst={self.sandbox_creates_burst}"
            )
        self.mode = mode

    @staticmethod
    def _resolve_sandbox_creates_per_sec(value: float | None) -> float | None:
        if value is None:
            value = _env_float(
                (
                    "SANDBOX_AGENT_LOOP_CREATES_PER_SEC",
                    "SANDBOX_CREATES_PER_SEC",
                    "GATEWAY_CREATES_PER_SEC",
                ),
                _DEFAULT_SANDBOX_CREATES_PER_SEC,
            )
        if value is not None and value <= 0:
            return None
        return value

    @staticmethod
    def _resolve_sandbox_creates_burst(value: int | None) -> int:
        if value is None:
            value = _env_int(
                (
                    "SANDBOX_AGENT_LOOP_CREATES_BURST",
                    "SANDBOX_CREATES_BURST",
                    "GATEWAY_CREATES_BURST",
                ),
                _DEFAULT_SANDBOX_CREATES_BURST,
            )
        if value is None or value <= 0:
            raise ValueError("sandbox_creates_burst must be > 0 when sandbox create rate limiting is enabled.")
        return value

    async def _throttle_sandbox_create(self) -> None:
        if self._sandbox_create_limiter is not None:
            await self._sandbox_create_limiter.acquire()

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
                rollout_state.uid = uuid.uuid4().int
            rollout_item.uid = rollout_state.uid
            rollout_item.group_id = rollout_state.message_uid
            await self._throttle_sandbox_create()
            result = await self._run_item(rollout_item)
            await self._fill_rollout_state(rollout_state, result)
            return rollout_state
        except Exception as exc:
            rollout_state.status = Status.COMPLETED if self.mode == "eval" else Status.FAILED
            rollout_state.finish_reason = "error"
            if self.mode == "eval":
                rollout_state.reward = {"score": 0.0}
                rollout_state.response = ""
                rollout_state.extra_fields["agent_status"] = "exception"
            rollout_state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[AgentInSandboxLoop] failed: {exc}\n{traceback.format_exc()}")
            return rollout_state

    async def _run_item(self, item: AgentRolloutItem) -> AgentRolloutItem:
        runner = _resolve_runner(item.pipeline, str(item.uid))
        if runner is None:
            raise ValueError("AgentRolloutItem.pipeline is required.")
        return await runner.run(item)

    async def _fill_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        if self.mode == "eval":
            self._fill_eval_rollout_state(rollout_state, item)
            return

        rollout_state.status = Status.COMPLETED if item.status == RolloutStatus.COMPLETED else Status.FAILED
        rollout_state.finish_reason = "stop" if item.status == RolloutStatus.COMPLETED else "error"
        rollout_state.reward = {"score": item.reward} if item.reward is not None else None
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"
        if item.status != RolloutStatus.COMPLETED:
            return

        messages, tools = _load_latest_trace_segment(item.artifacts, require_tools=True)
        if not messages:
            raise ValueError("Agent artifacts must contain at least one trainable messages trace.")
        session_id = rollout_state.uid

        trace_store = get_store()
        text = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(messages),
            tools=tools,
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

    def _fill_eval_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        is_success = item.status == RolloutStatus.COMPLETED
        rollout_state.status = Status.COMPLETED
        rollout_state.finish_reason = "stop" if is_success else "error"
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

        rollout_state.response = str(item.artifacts.get("agent_response") or "")
        messages, tools = _load_eval_trace_segment(item.artifacts)
        if messages:
            rollout_state.extra_fields["agent_trajectory"] = _to_json_safe({"messages": messages, "tools": tools})
