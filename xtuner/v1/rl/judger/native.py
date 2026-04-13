from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Callable, Literal, TypeAlias, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.utils import CPUActorLauncher
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
        info = {**self.extra_info}
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
        assert isinstance(judger_response, dict), (
            f"Reward handler must return a dict, but got {type(judger_response)}."
        )
        rollout_state.reward = judger_response
        return rollout_state

    def get_judger_name(self) -> str:
        """Get the name of the judger.

        Returns:
            str: The name of the judger.
        """
        return self._judger_name


class RouterJudger(Judger):
    """NativeJudger 路由管理器。

    功能：
    1. 通过维护 worker 负载实现负载均衡（Least Loaded）。
    2. 当负载相同时，通过轮询（Round-robin）分配任务。
    """

    def __init__(self, workers: list[RayJudgerProxy], judger_name: str):
        self.workers = workers
        self._worker_loads = dict.fromkeys(workers, 0)
        self._rr_index = 0
        self._lock = asyncio.Lock()
        self._judger_name = judger_name

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:  # type: ignore[override]
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
    def _validate_ray_actor_config(self) -> JudgerConfig:
        if self.judger_type == "native":
            if self.num_ray_actors > 1 or self.num_cpus_per_actor > 1 or self.cpu_memory_per_actor != 1024**3:
                logger.warning(
                    "num_ray_actors, num_cpus_per_actor and cpu_memory_per_actor settings will be ignored when judger_type is 'native'."
                )
        return self

    def get_num_placement_group_bundles(self) -> int:
        if self.judger_type == "native":
            return 0
        return self.num_ray_actors

    def get_cpu_bundles(self) -> list[dict[str, float | int]]:
        return [
            {
                "CPU": self.num_cpus_per_actor,
                "memory": self.cpu_memory_per_actor,
            }
            for _ in range(self.get_num_placement_group_bundles())
        ]

    def build_local(self) -> Judger:
        return NativeJudger(
            judger_name=self.judger_name,
            reward_handler=self.reward_handler,
            request_timeout=self.request_timeout,
            extra_info=self.extra_info,
        )

    def _build_ray_actor(self, pg: PlacementGroup | None = None, bundle_idx: int = 0) -> RayJudgerProxy:
        return CPUActorLauncher.build_actor(
            JudgerActor,
            self,
            pg=pg,
            bundle_idx=bundle_idx,
            actor_num_cpus=self.num_cpus_per_actor,
            actor_memory=self.cpu_memory_per_actor,
        )

    def _build_ray_actor_list(
        self,
        pg: PlacementGroup | None = None,
        start_bundle_idx: int = 0,
    ) -> list[RayJudgerProxy]:
        return CPUActorLauncher.build_actors(
            JudgerActor,
            self,
            pg=pg,
            start_bundle_idx=start_bundle_idx,
            num_workers=self.num_ray_actors,
            actor_num_cpus_per_worker=self.num_cpus_per_actor,
            actor_memory_per_worker=self.cpu_memory_per_actor,
        )

    def _build_router_workers(
        self, pg: PlacementGroup | None = None, start_bundle_idx: int = 0
    ) -> list[RayJudgerProxy]:
        return self._build_ray_actor_list(pg=pg, start_bundle_idx=start_bundle_idx)

    def _build_router(self, pg: PlacementGroup | None = None, start_bundle_idx: int = 0) -> RayJudgerProxy:
        workers_list = self._build_router_workers(pg=pg, start_bundle_idx=start_bundle_idx)
        return cast(
            RayJudgerProxy,
            CPUActorLauncher.build_actor(
                RouterJudger,
                workers_list,
                self.judger_name,
                actor_num_cpus=0,
                actor_memory=0,
            ),
        )

    def build(
        self,
        pg: PlacementGroup | None = None,
        start_bundle_idx: int = 0,
    ) -> Judger | RayJudgerProxy:
        if self.judger_type == "native":
            return self.build_local()

        if self.judger_type == "ray.actor":
            if self.num_ray_actors > 1:
                return self._build_router(pg=pg, start_bundle_idx=start_bundle_idx)
            return self._build_ray_actor(pg=pg, bundle_idx=start_bundle_idx)

        return self._build_router(pg=pg, start_bundle_idx=start_bundle_idx)


class JudgerActor:
    def __init__(self, judger_config: JudgerConfig):
        self.judger = judger_config.build_local()

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        return await self.judger.judge(rollout_state)


# For type hint and IDE support. For more info, please refer to:
# 1. https://docs.ray.io/en/latest/ray-core/actors.html#type-hints-and-static-typing-for-actors
# 2. https://github.com/InternLM/xtuner/pull/1349
RayJudger = cast(ActorClass[JudgerActor], CPUActorLauncher.to_actor_class(JudgerActor))
RayRouterJudger = cast(ActorClass[RouterJudger], CPUActorLauncher.to_actor_class(RouterJudger))
RayJudgerProxy: TypeAlias = ActorProxy[JudgerActor] | ActorProxy[RouterJudger]
