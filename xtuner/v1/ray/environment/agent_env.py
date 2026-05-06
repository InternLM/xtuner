import asyncio
import inspect
import os
import pickle
import traceback
from copy import deepcopy
from typing import Callable, List, Self, Tuple

import ray
from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLJudgerResponseItem,
    RolloutState,
    update_dataflow_item,
)
from xtuner.v1.ray.environment.lagent.schema import AgentMessage
from xtuner.v1.utils import get_logger

from .base_env import BaseEnvironment


# Memory diagnostics thresholds (bytes). Set via env var to disable (0).
_ITEM_SIZE_LOG_THRESHOLD = int(os.environ.get("XTUNER_ITEM_SIZE_LOG_MB", "1")) * 1024 * 1024
_RSS_MONITOR_INTERVAL = int(os.environ.get("XTUNER_RSS_MONITOR_SEC", "60"))


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


def _log_item_size_if_large(item) -> None:
    """Diagnostic: pickle-measure item + per-field breakdown when total exceeds threshold.

    Runs on every ``_inner_agent_call`` iteration; on fresh samples the total is a few KB
    so the early-exit keeps steady-state overhead negligible.  On resume / abort paths the
    item carries ``agent_state_dict`` / ``agent_message_dict`` from prior turns — we want
    visibility into which field dominates.
    """
    try:
        total_size = len(pickle.dumps(item))
    except Exception as exc:
        get_logger().debug(f"[item-size] pickle.dumps failed: {exc}")
        return
    if total_size < _ITEM_SIZE_LOG_THRESHOLD:
        return
    field_sizes: dict = {}
    try:
        for key, val in item.env.rollout.extra_info.items():
            try:
                field_sizes[key] = len(pickle.dumps(val))
            except Exception:
                field_sizes[key] = -1  # type: ignore[assignment]
    except Exception:
        pass
    get_logger().warning(
        f"[item-size] sample={getattr(item.uid, 'observation_id', '?')} "
        f"total={total_size / 1e6:.1f}MB "
        f"state={item.env.rollout.state} "
        f"fields={ {k: f'{v / 1e6:.1f}MB' for k, v in field_sizes.items()} }"
    )


@ray.remote(max_concurrency=int(os.environ.get("XTUNER_MAX_CONCURRENCY", 2000)))  # type: ignore[call-overload]
class AgentEnvironment(BaseEnvironment):
    def __init__(
        self,
        environment: str,
        agent_cfg: dict,
        rollout_controller,
        judger_pg=None,
        judger_cfg=None,
        preprocess_func: Callable[[Self, RLDataFlowItem], Tuple[AgentMessage]] = lambda _, item: (
            AgentMessage(role="user", content=item.data.messages[0]["content"]),  # type: ignore[index]
        ),
        postprocess_func: Callable[[Self, List[RLDataFlowItem]], List[RLDataFlowItem]] = lambda _, items: items,
    ):
        super().__init__(environment, None, None, judger_pg, judger_cfg)
        self.rollout_controller = rollout_controller
        self.agent_cfg = agent_cfg
        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func
        if _RSS_MONITOR_INTERVAL > 0:
            self._rss_task = asyncio.get_event_loop().create_task(self._rss_monitor())

    async def _rss_monitor(self):
        """Diagnostic: periodically log this actor's RSS so cross-sample leaks surface."""
        try:
            import psutil
        except ImportError:
            get_logger().warning("[actor-rss] psutil not available, disabling RSS monitor")
            return
        proc = psutil.Process()
        while True:
            try:
                rss_gb = proc.memory_info().rss / 1e9
                get_logger().info(f"[actor-rss] env={self.environment} pid={proc.pid} rss={rss_gb:.2f}GB")
            except Exception as exc:
                get_logger().warning(f"[actor-rss] read failed: {exc}")
            await asyncio.sleep(_RSS_MONITOR_INTERVAL)

    async def generate(  # type: ignore[override]
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        sample_params = sample_params.model_dump() if sample_params else {}

        async def _inner_agent_call(item):
            if item.env.rollout.state == RolloutState.COMPLETED:
                get_logger().debug(f"Rollout already completed for item {item.uid.observation_id}, skip agent call.")
                return "Passed"
            agent = create_object(self.agent_cfg)
            if item.env.rollout.state == RolloutState.ABORTED:
                agent.load_state_dict(item.env.rollout.extra_info["agent_state_dict"])  # type: ignore[operator]

            if _ITEM_SIZE_LOG_THRESHOLD > 0:
                _log_item_size_if_large(item)
            _item = deepcopy(item)
            _item.env.agent.extra_info["agent"] = agent
            agent_inputs = self.preprocess_func(self, _item)
            try:
                return await agent(*agent_inputs, **sample_params)
            except BaseException as exc:
                get_logger().error(
                    f"[Agent Inference Error] {exc}. Dead actors: {check_dead_actors()}\n{traceback.format_exc()}"
                )
                return "Failed"
            finally:
                item.env.rollout.extra_info["agent_state_dict"] = agent.state_dict()  # type: ignore[typeddict-unknown-key]
                item.env.rollout.extra_info["agent_message_dict"] = agent.get_messages()  # type: ignore[typeddict-unknown-key]

        results = await asyncio.gather(*[_inner_agent_call(sample) for sample in group_data_items])
        passed_data_items, completed_data_items = [], []
        for sample, message in zip(group_data_items, results):
            if message == "Failed":
                continue
            if message == "Passed":
                passed_data_items.append(sample)
            elif message.finish_reason == "abort":
                sample.env.rollout.state = RolloutState.ABORTED
                passed_data_items.append(sample)
            else:
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
