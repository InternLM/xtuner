import asyncio
import inspect
from typing import Callable, List

import httpx
import ray
from pydantic import BaseModel, ConfigDict, Field
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState


class NativeJudger:
    # 默认使用NativeJudger的Judger为ray.actor，如果为function，则通过用户自己调用
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

    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        # native preprocess
        assert rollout_state.response is not None, (
            "RolloutState must have a response for judging. You should detokenize the response_ids in AgentLoop"
        )
        # 传入rollout_state方便用户从rollout_state挑选自己想要的字段
        info = {**self.extra_info, "rollout_state": rollout_state}
        input_kwargs = {
            "response": rollout_state.response,
            "label": rollout_state.reward_model["ground_truth"],
            "extra_info": info,
        }
        # actual judger function
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
        # native postprocess
        rollout_state.reward = judger_response
        return rollout_state

    def get_judger_name(self) -> str:
        """Get the name of the judger.

        Returns:
            str: The name of the judger.
        """
        return self._judger_name


class NativeJudgerRouter:
    """NativeJudger 路由管理器。

    功能：
    1. 通过维护 worker 负载实现负载均衡（Least Loaded）。
    2. 当负载相同时，通过轮询（Round-robin）分配任务。
    """

    def __init__(self, workers: List[ray.actor.ActorHandle], judger_name: str):
        self.workers = workers
        self._worker_loads = {worker: 0 for worker in workers}
        self._rr_index = 0
        self._lock = asyncio.Lock()
        self._judger_name = judger_name

    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        async with self._lock:
            min_load = min(self._worker_loads.values())
            candidates = [w for w in self.workers if self._worker_loads[w] == min_load]
            worker = candidates[self._rr_index % len(candidates)]
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


class NativeJudgerConfig(BaseModel):
    """Configuration class for NativeJudger.

    This class defines the configuration options for initializing a NativeJudger,
    including resource allocation (number of Ray actors and CPUs per actor),
    reward function or remote judging service, optional pre/post-processing functions,
    request timeout, and any extra information needed for judging.

    Attributes:
        judger_name (str): Name identifier for the judger.
        num_ray_actors (int): Number of Ray actor instances to launch.
        num_cpus_per_actor (int): Number of CPUs allocated per actor.
        reward_func (Optional[Callable]): Local reward function for judging.
            Exactly one of reward_func or remote_url must be provided.
        remote_url (Optional[str]): Remote service URL for judging.
            Exactly one of reward_func or remote_url must be provided.
        preprocess_func (Optional[Callable]): Function to preprocess input data before judging.
        postprocess_func (Optional[Callable]): Function to postprocess the judging result.
        request_timeout (float): Timeout (in seconds) for remote requests.
        extra_info (dict): Additional information to be passed to the judger or reward function.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    judger_name: str
    num_ray_actors: int = 1
    num_cpus_per_actor: int = 1
    cpu_memory_per_actor: int = 1024**3
    reward_handler: Callable | str = Field(default=None, exclude=True)
    request_timeout: float = 30.0
    extra_info: dict = Field(default={}, exclude=True)

    def build(self) -> NativeJudger:
        return NativeJudger(
            judger_name=self.judger_name,
            reward_handler=self.reward_handler,
            request_timeout=self.request_timeout,
            extra_info=self.extra_info,
        )

    def build_router(self, pg: PlacementGroup, start_bundle_idx: int) -> NativeJudgerRouter:
        """Create and launch Ray actor instances for the GSM8K judger.

        This method instantiates multiple NativeJudger Ray actors according to `num_ray_actors`,
        assigning each to a specific bundle in the provided placement group for resource isolation.
        Each actor is initialized with the judger's configuration and reward function.

        Args:
            pg: The Ray PlacementGroup used to allocate resources for the actors.
            start_bundle_idx: The starting bundle index in the placement group for actor placement.

        Returns:
            List[ActorClass]: A list of Ray actor handles representing the launched judger workers.
        """
        workers_list = []
        for idx in range(self.num_ray_actors):
            bundle_idx = start_bundle_idx + idx
            pg_options = {"num_cpus": self.num_cpus_per_actor, "memory": self.cpu_memory_per_actor}
            assert pg.bundle_specs[bundle_idx].get("CPU", 1) >= self.num_cpus_per_actor, (
                f"Placement group bundle {bundle_idx} does not have enough CPU resources."
            )
            assert pg.bundle_specs[bundle_idx].get("memory", 0) >= self.cpu_memory_per_actor, (
                f"Placement group bundle {bundle_idx} does not have enough memory resources."
            )
            worker = (
                ray.remote(NativeJudger)
                .options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    **pg_options,
                )
                .remote(
                    judger_name=self.judger_name,
                    reward_handler=self.reward_handler,
                    request_timeout=self.request_timeout,
                    extra_info=self.extra_info,
                )
            )
            workers_list.append(worker)
        judger_router = NativeJudgerRouter(workers=workers_list, judger_name=self.judger_name)
        return judger_router
