from __future__ import annotations

import copy
import importlib
import json
import traceback
from typing import Any
import uuid
from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController

from ..agent_loop import AgentLoop, AgentLoopConfig
from .schemas import AgentRolloutItem, RolloutStatus
from ...rollout.trace_store import get_store

import ray

_STORE_NAME = "rollout_trace_store"

def get_store1():
    try:
        return ray.get_actor(_STORE_NAME)
    except ValueError:
        pass

    from ray.util.state import list_actors

    actors = list_actors(filters=[("name", "=", _STORE_NAME)], detail=True)
    if not actors:
        raise RuntimeError(f"cannot find ray actor: {_STORE_NAME}")

    actor = actors[0]
    namespace = actor.get("ray_namespace") if isinstance(actor, dict) else actor.ray_namespace
    print(f"found {_STORE_NAME} in namespace={namespace}")
    return ray.get_actor(_STORE_NAME, namespace=namespace)


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
    def build_local(self, rollout_controller: RolloutController | None = None, judger: Judger | None = None, logger=None) -> "AgentInSandboxLoop":
        return AgentInSandboxLoop(
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
        )


class AgentInSandboxLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController | None = None,
        hf_checkpoint: str = None,
        judger: Judger | None = None,
        logger=None
    ):
        super().__init__(rollout_ctl, None, hf_checkpoint, judger, logger)

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
        artifacts = item.artifacts
        message=json.loads(artifacts["message"])
        messages = message['policy_agent.messages']
        tools = message.get("tools", None)
        session_id = rollout_state.uid
        
        # import debugpy
        # debugpy.connect(('10.102.250.69', 5680))

        # 获取数据
        trace_store = get_store()
        text = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        data = await trace_store.export_training_trace.remote(str(session_id), text[:-1]) # '\n'
        
        rollout_state.input_ids = data['input_ids']
        rollout_state.labels = data['labels']
        rollout_state.logprobs = data['logprobs']
        rollout_state.routed_experts = data['routed_experts']
        rollout_state.finish_reason = 'stop' if item.status == RolloutStatus.COMPLETED else 'error'
        rollout_state.status = item.status
        rollout_state.reward = {"score": item.reward}
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"
