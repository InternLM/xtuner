from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutEndpoint
from xtuner.v1.rl.utils import CPUActorLauncher, create_task
from xtuner.v1.utils import get_logger, ray_method
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams
    num_ray_actors: int = Field(
        default=0,
        ge=0,
        description="Number of AgentLoop Ray actor instances. 0 means local mode.",
    )
    num_cpus: float = Field(default=1, gt=0, description="CPU cores required by the AgentLoop actor itself.")
    cpu_memory: int = Field(default=1024**3, gt=0, description="CPU memory in bytes required by AgentLoop.")

    @model_validator(mode="after")
    def _validate_ray_actor_config(self) -> AgentLoopConfig:
        if self.num_ray_actors == 0 and (self.num_cpus != 1 or self.cpu_memory != 1024**3):
            logger = get_logger()
            logger.warning("num_cpus and cpu_memory are ignored when AgentLoop runs in local mode.")
        return self

    def build(
        self,
        rollout_endpoint: RolloutEndpoint,
        judger: Judger | None = None,
        logger=None,
    ) -> AgentLoopSpec:
        if self.num_ray_actors == 0:
            return self.build_local(
                rollout_endpoint=rollout_endpoint,
                judger=judger,
                logger=logger,
            )
        if self.num_ray_actors > 1:
            return self._build_router(
                rollout_endpoint=rollout_endpoint,
                judger=judger,
                logger=logger,
            )
        return self._build_ray_actor(
            rollout_endpoint=rollout_endpoint,
            judger=judger,
            logger=logger,
        )

    @abstractmethod
    def build_local(
        self,
        rollout_endpoint: RolloutEndpoint,
        judger: Judger | None = None,
        logger=None,
    ) -> AgentLoop: ...

    def _build_ray_actor(
        self,
        rollout_endpoint: RolloutEndpoint,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> RayAgentLoopProxy:
        return cast(
            "RayAgentLoopProxy",
            CPUActorLauncher.build_actor(
                AgentLoopActor,
                self,
                rollout_endpoint,
                judger,
                logger,
                pg=pg,
                bundle_idx=0,
                actor_num_cpus=self.num_cpus,
                actor_memory=self.cpu_memory,
                capture_child_tasks=True,
            ),
        )

    def _build_ray_actors(
        self,
        rollout_endpoint: RolloutEndpoint,
        num_actors: int,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> list[RayAgentLoopProxy]:
        return cast(
            list["RayAgentLoopProxy"],
            CPUActorLauncher.build_actors(
                AgentLoopActor,
                self,
                rollout_endpoint,
                judger,
                logger,
                pg=pg,
                start_bundle_idx=start_bundle_idx,
                num_workers=num_actors,
                actor_num_cpus_per_worker=self.num_cpus,
                actor_memory_per_worker=self.cpu_memory,
                capture_child_tasks=True,
            ),
        )

    def _build_router(
        self,
        rollout_endpoint: RolloutEndpoint,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> RouterAgentLoop:
        return RouterAgentLoop(
            workers=self._build_ray_actors(
                rollout_endpoint=rollout_endpoint,
                num_actors=self.num_ray_actors,
                pg=pg,
                judger=judger,
                logger=logger,
                start_bundle_idx=start_bundle_idx,
            ),
        )


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_endpoint: RolloutEndpoint,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
    ) -> None:
        self.rollout_endpoint = rollout_endpoint
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judger = judger
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState: ...

    async def rollout_generate(self, rollout_state: RolloutState) -> RolloutState:
        if self.rollout_endpoint.kind == "worker_local":
            return await self.rollout_endpoint.require_worker_local_router().generate(rollout_state)
        return await self.rollout_generate_from_url(
            rollout_state=rollout_state,
            base_url=self.rollout_endpoint.require_base_url(),
        )
    
    # 这个地方还需要思考，rollout_state 如何发送给 worker 的 url？ 应该还有一层 gateway，因为要做 token 记录+session 隔离
    async def rollout_generate_from_url(self, rollout_state: RolloutState, base_url: str) -> RolloutState:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement URL rollout generation for endpoint "
            f"{self.rollout_endpoint.kind!r} at {base_url!r}."
        )

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples


class RouterAgentLoop:
    def __init__(self, workers: list[RayAgentLoopProxy]):
        self.workers = workers
        self._worker_loads = dict.fromkeys(workers, 0)
        self._rr_index = 0
        self._lock = asyncio.Lock()

    async def _pick_worker(self) -> RayAgentLoopProxy:
        async with self._lock:
            min_load = min(self._worker_loads.values())
            candidates = [worker for worker in self.workers if self._worker_loads[worker] == min_load]
            worker = candidates[self._rr_index % len(candidates)]
            self._rr_index = (self._rr_index + 1) % len(self.workers)
            self._worker_loads[worker] += 1
            return worker

    async def _release_worker(self, worker: RayAgentLoopProxy) -> None:
        async with self._lock:
            self._worker_loads[worker] -= 1

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        worker = await self._pick_worker()
        try:
            return await worker.generate_sample.remote(rollout_state, **kwargs)
        finally:
            await self._release_worker(worker)

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        worker = await self._pick_worker()
        try:
            return await worker.generate_group.remote(rollout_state, **kwargs)
        finally:
            await self._release_worker(worker)

    def get_worker_status(self) -> dict[str, int]:
        return {str(worker): load for worker, load in self._worker_loads.items()}


class AgentLoopActor:
    def __init__(
        self,
        agent_loop_config: AgentLoopConfig,
        rollout_endpoint: RolloutEndpoint,
        judger: Judger | None = None,
        logger=None,
    ):
        self.agent_loop_config = agent_loop_config
        self.agent_loop = agent_loop_config.build_local(
            rollout_endpoint=rollout_endpoint,
            judger=judger,
            logger=logger,
        )

    @ray_method
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        return await self.agent_loop.generate_sample(rollout_state, **kwargs)

    @ray_method
    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        return await self.agent_loop.generate_group(rollout_state, **kwargs)


RayAgentLoop = cast(ActorClass[AgentLoopActor], CPUActorLauncher.to_actor_class(AgentLoopActor))
RayAgentLoopProxy: TypeAlias = ActorProxy[AgentLoopActor]
AgentLoopSpec: TypeAlias = AgentLoop | RayAgentLoopProxy | RouterAgentLoop
