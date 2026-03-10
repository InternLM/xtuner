import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, TypeAlias, cast

import httpx
import ray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils.logger import get_logger
from xtuner.v1.utils.type_helper import ray_method


logger = get_logger()


class Judger(ABC):
    @abstractmethod
    async def judge(self, rollout_state: RolloutState) -> RolloutState: ...


class NativeJudger(Judger):
    """Base class for judgers, providing a standard interface for executing a
    judging process, which can be either a local function or a remote service.

    The judger orchestrates a three-step pipeline:
    1. Pre-process the input data.
    2. Execute the core logic (local function or remote HTTP call).
    3. Post-process the result.
    """

    def __init__(
        self,
        judger_name: str = "native_judger",
        reward_handler: Callable | str | None = None,  # 支持函数/url，其他类型可以后面再加上
        extra_info: dict = {},
        request_timeout: float = 30.0,
    ):
        self._judger_name = judger_name
        self.extra_info = extra_info
        self.reward_handler = reward_handler
        self.request_timeout = request_timeout

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:  # type: ignore[override]
        # native preprocess
        assert rollout_state.response is not None, (
            "RolloutState must have a response for judging. You should detokenize the response_ids in AgentLoop"
        )
        # 传入rollout_state方便用户从rollout_state挑选自己想要的字段
        info = {**self.extra_info, "rollout_state": rollout_state.model_dump(mode="json")}
        assert rollout_state.reward_model is not None and "ground_truth" in rollout_state.reward_model, (
            "RolloutState must have reward_model with 'ground_truth' for judging. You should set reward_model in AgentLoop"
        )
        input_kwargs = {
            "response": rollout_state.response,
            "label": rollout_state.reward_model["ground_truth"],
            "extra_info": info,
        }
        # actual judger function
        judger_response = None
        if isinstance(self.reward_handler, str):
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(self.reward_handler, json=input_kwargs)
                response.raise_for_status()
                judger_response = response.json()
        elif callable(self.reward_handler):
            if inspect.iscoroutinefunction(self.reward_handler):
                judger_response = await self.reward_handler(**input_kwargs)
            else:
                judger_response = self.reward_handler(**input_kwargs)
        assert judger_response is not None, "Reward handler did not return a response."
        # native postprocess
        rollout_state.reward = judger_response
        return rollout_state

    def get_judger_name(self) -> str:
        """Get the name of the judger.

        Returns:
            str: The name of the judger.
        """
        return self._judger_name


# For type hint and IDE support. For more info, please refer to:
# 1. https://docs.ray.io/en/latest/ray-core/actors.html#type-hints-and-static-typing-for-actors
# 2. https://github.com/InternLM/xtuner/pull/1349
RayJudger = cast(ActorClass[NativeJudger], ray.remote(NativeJudger))
RayJudgerProxy: TypeAlias = ActorProxy[NativeJudger]


class RouterJudger(Judger):
    """NativeJudger 路由管理器。

    功能：
    1. 通过维护 worker 负载实现负载均衡（Least Loaded）。
    2. 当负载相同时，通过轮询（Round-robin）分配任务。
    """

    def __init__(self, workers: List[RayJudgerProxy], judger_name: str):
        self.workers = workers
        self._worker_loads = dict.fromkeys(workers, 0)
        self._rr_index = 0
        self._lock = asyncio.Lock()
        self._judger_name = judger_name

    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        async with self._lock:
            min_load = min(self._worker_loads.values())
            candidates: list[RayJudgerProxy] = [w for w in self.workers if self._worker_loads[w] == min_load]
            worker: RayJudgerProxy = candidates[self._rr_index % len(candidates)]
            self._rr_index = (self._rr_index + 1) % len(self.workers)
            self._worker_loads[worker] += 1
        try:
            return await worker.judge.remote(rollout_state)
        finally:
            async with self._lock:
                self._worker_loads[worker] -= 1

    def get_worker_status(self) -> dict[str, int]:
        return {str(w): load for w, load in self._worker_loads.items()}

    def get_judger_name(self) -> str:
        return self._judger_name


class JudgerConfig(BaseModel):
    """Configuration class for NativeJudger / Ray actor / RouterJudger."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    judger_name: str
    judger_type: Literal["native", "ray.actor", "router"] = "native"
    reward_handler: Callable | str = Field(default=None, exclude=True)
    request_timeout: float = 30.0
    extra_info: dict = Field(default={}, exclude=True)
    num_ray_actors: int = Field(
        default=1, description="Number of Ray actor instances. Must be 1 when judger_type is 'ray.actor'."
    )
    num_cpus_per_actor: int = Field(
        default=1, description="CPU cores per Ray actor. Only for 'ray.actor' or 'router'."
    )
    cpu_memory_per_actor: int = Field(
        default=1024**3, description="CPU memory in bytes per Ray actor. Only for 'ray.actor' or 'router'."
    )

    @model_validator(mode="after")
    def _validate_ray_actor_config(self) -> "JudgerConfig":
        if self.judger_type == "ray.actor" and self.num_ray_actors > 1:
            logger.warning(
                "num_ray_actors will be set to 1 when judger_type is 'ray.actor'."
            )
            self.num_ray_actors = 1
        if self.judger_type == "native":
            if self.num_ray_actors > 1 or self.num_cpus_per_actor > 1 or self.cpu_memory_per_actor != 1024**3:
                logger.warning(
                    "num_ray_actors, num_cpus_per_actor and cpu_memory_per_actor settings will be ignored when judger_type is 'native'."
                )
        return self

    def _build_worker(self, pg: PlacementGroup | None = None, bundle_idx: int = 0) -> RayJudgerProxy:
        pg_options = {"num_cpus": self.num_cpus_per_actor, "memory": self.cpu_memory_per_actor}
        if pg is None:
            # NOTE: 保持与 router 构建逻辑一致，默认创建 PlacementGroup。
            from xtuner.v1.rl.utils.ray_worker import CPUResourcesConfig

            cpu_resource_cfg = CPUResourcesConfig(
                num_workers=self.num_ray_actors,
                num_cpus_per_worker=self.num_cpus_per_actor,
                cpu_memory_per_worker=self.cpu_memory_per_actor,
            )
            pg = cpu_resource_cfg.build_placement_group()
            ray.get(pg.ready())
            bundle_idx = 0

        assert len(pg.bundle_specs) > bundle_idx, "Placement group does not have enough bundles for ray actor."
        assert pg.bundle_specs[bundle_idx].get("CPU", 1) >= self.num_cpus_per_actor, (
            f"Placement group bundle {bundle_idx} does not have enough CPU resources."
        )
        assert pg.bundle_specs[bundle_idx].get("memory", 0) >= self.cpu_memory_per_actor, (
            f"Placement group bundle {bundle_idx} does not have enough memory resources."
        )
        return RayJudger.options(
            placement_group=pg,
            placement_group_bundle_index=bundle_idx,
            **pg_options,
        ).remote(
            judger_name=self.judger_name,
            reward_handler=self.reward_handler,
            request_timeout=self.request_timeout,
            extra_info=self.extra_info,
        )

    def _build_workers(self, pg: PlacementGroup | None = None, start_bundle_idx: int = 0) -> list[RayJudgerProxy]:
        """Create and launch Ray actor instances for router workers.

        This method instantiates multiple NativeJudger Ray actors according to `num_ray_actors`,
        assigning each to a specific bundle in the provided placement group for resource isolation.
        Each actor is initialized with the judger's configuration and reward function.

        Args:
            pg: The Ray PlacementGroup used to allocate resources for the actors.
            start_bundle_idx: The starting bundle index in the placement group for actor placement.

        Returns:
            List[ActorClass]: A list of Ray actor handles representing the launched judger workers.
        """
        if pg is None:
            # NOTE: 这里直接在build_workers里创建PlacementGroup是为了简化用户使用，用户不需要关心PlacementGroup的细节。
            from xtuner.v1.rl.utils.ray_worker import CPUResourcesConfig

            cpu_resource_cfg = CPUResourcesConfig(
                num_workers=self.num_ray_actors,
                num_cpus_per_worker=self.num_cpus_per_actor,
                cpu_memory_per_worker=self.cpu_memory_per_actor,
            )
            pg = cpu_resource_cfg.build_placement_group()
            ray.get(pg.ready())
            start_bundle_idx = 0

        workers_list = []
        assert len(pg.bundle_specs) >= self.num_ray_actors, (
            "Placement group does not have enough bundles for the number of ray actors."
        )
        for idx in range(self.num_ray_actors):
            workers_list.append(self._build_worker(pg=pg, bundle_idx=start_bundle_idx + idx))
        return workers_list

    def build(
        self,
        pg: PlacementGroup | None = None,
        start_bundle_idx: int = 0,
    ) -> NativeJudger | RayJudgerProxy | RouterJudger:
        if self.judger_type == "native":
            return NativeJudger(
                judger_name=self.judger_name,
                reward_handler=self.reward_handler,
                request_timeout=self.request_timeout,
                extra_info=self.extra_info,
            )

        if self.judger_type == "ray.actor":
            return self._build_worker(pg=pg, bundle_idx=start_bundle_idx)

        workers_list = self._build_workers(pg=pg, start_bundle_idx=start_bundle_idx)
        return RouterJudger(workers=workers_list, judger_name=self.judger_name)
