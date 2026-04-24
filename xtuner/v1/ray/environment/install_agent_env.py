import asyncio
import importlib
import inspect
import os
import traceback
from typing import Callable, List, Self, Tuple

import ray
from lagent.serving.sandbox.providers.gateway import GatewayProvider

from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLJudgerResponseItem,
    RolloutState,
    update_dataflow_item,
)
from xtuner.v1.ray.environment.lagent.schema import AgentMessage
from xtuner.v1.utils import get_logger

from .base_env import BaseEnvironment


def check_dead_actors():
    # 获取所有 Actor 的列表
    from ray.util.state import list_actors

    all_actors = list_actors()

    dead_actors = []
    for actor_info in all_actors:
        # 状态通常是 "ALIVE", "DEAD", "RECONSTRUCTING" 等
        if actor_info["state"] == "DEAD":
            dead_actors.append(actor_info)

    return dead_actors


DEFAULT_GATEWAY = "http://env-gateway.ailab.ailab.ai"
DEFAULT_LAGENT_SRC = "/mnt/shared-storage-user/llmit/user/wangziyi/projs/lagent"


def _import_from_path(path: str):
    """Import an object from a dotted path like 'pkg.mod.attr'."""
    if not path or not isinstance(path, str):
        raise TypeError(f"pipeline must be a non-empty str, got {type(path)}")
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path: {path!r}. Expected 'module.attr'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_pipeline(pipeline_spec):
    """Resolve pipeline spec into a Runner-like object with run_single()."""
    obj = _import_from_path(pipeline_spec)
    # Most configs point to a factory function (e.g., claw_pipeline()).
    return obj() if callable(obj) else obj


@ray.remote(max_concurrency=int(os.environ.get("XTUNER_MAX_CONCURRENCY", 2000)))  # type: ignore[call-overload]
class InstallAgentEnvironment(BaseEnvironment):
    def __init__(
        self,
        environment: str,
        rollout_controller,
        preprocess_func: Callable[[Self, RLDataFlowItem], Tuple[AgentMessage]] = lambda _, item: (
            AgentMessage(role="user", content=item.data.messages[0]["content"]),  # type: ignore[index]
        ),
        postprocess_func: Callable[[Self, List[RLDataFlowItem]], List[RLDataFlowItem]] = lambda _, items: items,
    ):
        super().__init__(environment, None, None, None, None)
        self.rollout_controller = rollout_controller
        self.provider = GatewayProvider(DEFAULT_GATEWAY)
        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func

    async def generate(  # type: ignore[override]
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        sample_params = sample_params.model_dump() if sample_params else {}

        async def _inner_agent_call(item):
            if item.env.rollout.state == RolloutState.COMPLETED:
                get_logger().debug(f"Rollout already completed for item {item.uid.observation_id}, skip agent call.")
                return "Passed"

            # agent_inputs = self.preprocess_func(self, deepcopy(item))
            pipeline = item.data.extra_info.get("pipeline", None)
            pipeline = _resolve_pipeline(pipeline)

            task_dir = item.data.extra_info.get("task_dir", None)
            data = item.data.extra_info.get("task_data", None)
            uid = {
                "root_id": item.uid.root_id,
                "action_id": item.uid.action_id,
                "observation_id": item.uid.observation_id,
            }
            get_logger().info(f"running task={task_dir} (dataset={data.id})")

            # import debugpy
            # debugpy.connect(('10.102.151.17', 5680))
            try:
                return await pipeline.run_single(
                    task_dir,
                    data,
                    uid,
                    provider=self.provider,
                    lagent_src_dir=DEFAULT_LAGENT_SRC,
                    llm_base_url=None,
                    llm_api_key=None,
                )
            except BaseException as exc:
                get_logger().error(
                    f"[Agent Inference Error] {exc}. Dead actors: {check_dead_actors()}\n{traceback.format_exc()}"
                )
                return "Failed"

        results = await asyncio.gather(*[_inner_agent_call(sample) for sample in group_data_items])
        passed_data_items, completed_data_items = [], []
        for sample, result in zip(group_data_items, results):
            if result == "Failed":
                continue
            if result == "Passed":
                passed_data_items.append(sample)
            elif result["env"]["rollout"]["finish_reason"] == "abort":
                # sample.env.rollout.state = RolloutState.ABORTED
                # agent_state_dict = self.agent.state_dict(sample.uid.observation_id)
                # remove routed_experts from message extra_info to avoid serialization issue
                # for state in agent_state_dict.values():
                #     for msg in state:
                #         msg["extra_info"].pop("routed_experts", None)
                # sample.env.rollout.extra_info["agent_state_dict"] = agent_state_dict  # type: ignore[typeddict-unknown-key]
                # passed_data_items.append(sample)
                continue
            else:
                sample.env.agent.extra_info["message_dict"] = result["env"]["agent"]["message_dict"]
                sample.env.judger.extra_info.update(result["env"]["judger"])
                completed_data_items.append(sample)
        completed_data_items_result = self.postprocess_func(self, completed_data_items)  # type: ignore[arg-type]
        if inspect.iscoroutinefunction(self.postprocess_func):
            completed_data_items_result = await completed_data_items_result  # type: ignore[misc]
        return passed_data_items + completed_data_items_result

    async def run(  # type: ignore[override]
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        group_data_items = await self.generate(group_data_items, sample_params, extra_params)
        skip_judger = any(
            item.env.rollout.finish_reason == "abort" or item.env.rollout.finish_reason == "failed"
            for item in group_data_items
        )
        if self.judger_controller and not skip_judger:
            try:
                judger_responses: List[RLJudgerResponseItem] = await asyncio.wait_for(
                    self.judger_controller.run.remote(group_data_items), timeout=1200.0
                )
            except asyncio.TimeoutError:
                judger_responses = [RLJudgerResponseItem(extra_info={"state": "failed"}) for _ in group_data_items]
            group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_responses)
        return group_data_items
