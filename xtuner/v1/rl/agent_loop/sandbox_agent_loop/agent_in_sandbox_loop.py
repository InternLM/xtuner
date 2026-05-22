from __future__ import annotations

import copy
import importlib
import json
import traceback
from typing import Any

from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController

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

    response_artifact_key: str = "agent_response"
    messages_artifact_key: str = "message"

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "AgentInSandboxLoop":
        return AgentInSandboxLoop(
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            response_artifact_key=self.response_artifact_key,
            messages_artifact_key=self.messages_artifact_key,
        )


class AgentInSandboxLoop(AgentLoop):
    def __init__(
        self,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        *,
        response_artifact_key: str = "agent_response",
        messages_artifact_key: str = "message",
    ):
        super().__init__(None, None, hf_checkpoint, judger, logger)
        self.response_artifact_key = response_artifact_key
        self.messages_artifact_key = messages_artifact_key

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        try:
            rollout_item = rollout_state.extra_fields["rollout_item"].model_copy(deep=True)
            rollout_item.uid = rollout_state.uid
            rollout_item.group_id = rollout_state.message_uid
            result = await self._run_item(rollout_item)
            self._fill_rollout_state(rollout_state, result)
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

    def _fill_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        response = self._extract_response(item)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        rollout_state.response = response
        rollout_state.response_ids = response_ids
        rollout_state.logprobs = [0.0] * len(response_ids)
        rollout_state.response_mask = [1] * len(response_ids)
        rollout_state.finish_reason = "stop" if item.status == RolloutStatus.COMPLETED else "error"
        rollout_state.status = Status.COMPLETED if item.status == RolloutStatus.COMPLETED else Status.FAILED
        rollout_state.reward = {"score": item.reward if item.reward is not None else 0.0}
        rollout_state.extra_fields["agent_rollout_item"] = item
        rollout_state.extra_fields["agent_artifacts"] = item.artifacts

        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"

    def _extract_response(self, item: AgentRolloutItem) -> str:
        response = item.artifacts.get(self.response_artifact_key)
        if isinstance(response, str) and response:
            return response

        messages = item.artifacts.get(self.messages_artifact_key)
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                return messages

        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if role == "assistant" and content:
                    return str(content)
        return ""
