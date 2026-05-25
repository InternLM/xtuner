from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController, RolloutGenerateHandle, RolloutGenerateHandleConfig
from xtuner.v1.rl.utils import (
    CPUActorLauncher,
    CPUResourcesConfig,
    create_task,
    register_cpu_resources,
)
from xtuner.v1.utils import get_logger, ray_method
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams | None = None
    cpu_resources: CPUResourcesConfig | None = None

    def build(
        self,
        rollout_controller=None,
        rollout_generator: RolloutGenerateHandle | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> AgentLoopSpec:
        if rollout_generator is None:
            if rollout_controller is None:
                raise ValueError("Either rollout_controller or rollout_generator must be provided.")
            rollout_generator = RolloutGenerateHandleConfig().build(rollout_controller)
        if self.cpu_resources is None:
            return self.build_local(
                rollout_controller=rollout_controller,
                rollout_generator=rollout_generator,
                judger=judger,
                logger=logger,
            )

        register_cpu_resources(
            name=f"agent_loop:{self.__class__.__name__}",
            cpu_resources=self.cpu_resources,
        )

        if self.cpu_resources.num_workers > 1:
            return self._build_router(
                rollout_controller=rollout_controller,
                rollout_generator=rollout_generator,
                cpu_resources=self.cpu_resources,
                judger=judger,
                logger=logger,
            )
        return self._build_ray_actor(
            rollout_controller=rollout_controller,
            rollout_generator=rollout_generator,
            cpu_resources=self.cpu_resources,
            judger=judger,
            logger=logger,
        )

    @abstractmethod
    def build_local(
        self,
        rollout_controller,
        rollout_generator: RolloutGenerateHandle | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> AgentLoop: ...

    def _build_ray_actor(
        self,
        rollout_controller: RolloutController,
        rollout_generator: RolloutGenerateHandle,
        cpu_resources: CPUResourcesConfig,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> RayAgentLoopProxy:
        return cast(
            "RayAgentLoopProxy",
            CPUActorLauncher.build_actor(
                AgentLoopActor,
                self,
                rollout_controller,
                rollout_generator,
                judger,
                pg=pg,
                bundle_idx=0,
                actor_num_cpus=cpu_resources.num_cpus_per_worker,
                actor_memory=cpu_resources.cpu_memory_per_worker,
                capture_child_tasks=True,
            ),
        )

    def _build_ray_actors(
        self,
        rollout_controller: RolloutController,
        rollout_generator: RolloutGenerateHandle,
        cpu_resources: CPUResourcesConfig,
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
                rollout_controller,
                rollout_generator,
                judger,
                pg=pg,
                start_bundle_idx=start_bundle_idx,
                num_workers=cpu_resources.num_workers,
                actor_num_cpus_per_worker=cpu_resources.num_cpus_per_worker,
                actor_memory_per_worker=cpu_resources.cpu_memory_per_worker,
                capture_child_tasks=True,
            ),
        )

    def _build_router(
        self,
        rollout_controller: RolloutController,
        rollout_generator: RolloutGenerateHandle,
        cpu_resources: CPUResourcesConfig,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> RouterAgentLoop:
        return RouterAgentLoop(
            workers=self._build_ray_actors(
                rollout_controller=rollout_controller,
                rollout_generator=rollout_generator,
                cpu_resources=cpu_resources,
                pg=pg,
                judger=judger,
                logger=logger,
                start_bundle_idx=start_bundle_idx,
            ),
            rollout_ctl=rollout_controller,
        )


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController | None,
        rollout_generator: RolloutGenerateHandle | None,
        sample_params: SampleParams | None,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        ) -> None:
        self.rollout_ctl = rollout_ctl
        self.rollout_generator = rollout_generator
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

    async def rollout_generate(self, rollout_state: RolloutState, *, enable_partial_rollout: bool = False) -> RolloutState:
        if self.rollout_generator is None:
            raise RuntimeError("AgentLoop requires rollout_generator; RolloutController.generate has been removed.")
        if self.rollout_generator.kind == "local":
            return await self.rollout_generator.require_local_generator().generate(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        return await self.rollout_generate_from_url(
            rollout_state=rollout_state,
            base_url=self.rollout_generator.require_base_url(),
        )

    async def rollout_generate_from_url(self, rollout_state: RolloutState, base_url: str) -> RolloutState:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement URL rollout generation for endpoint at {base_url!r}."
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
    def __init__(self, workers: list[RayAgentLoopProxy], rollout_ctl: RolloutController):
        self.workers = workers
        self.rollout_ctl = rollout_ctl
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


async def get_agent_loop_rollout_ctl(agent_loop: AgentLoopSpec) -> RolloutController:
    rollout_ctl = getattr(agent_loop, "rollout_ctl", None)
    if rollout_ctl is not None:
        return rollout_ctl

    get_rollout_ctl = getattr(agent_loop, "get_rollout_ctl", None)
    if get_rollout_ctl is None or not hasattr(get_rollout_ctl, "remote"):
        raise AttributeError(f"Agent loop {type(agent_loop)} does not expose rollout_ctl or get_rollout_ctl().")
    return await get_rollout_ctl.remote()


class AgentLoopActor:
    def __init__(
        self,
        agent_loop_config: AgentLoopConfig,
        rollout_controller: RolloutController,
        rollout_generator: RolloutGenerateHandle,
        judger: Judger | None = None,
        logger=None,
    ):
        self.agent_loop = agent_loop_config.build_local(
            rollout_controller=rollout_controller,
            rollout_generator=rollout_generator,
            judger=judger,
            logger=logger,
        )

    @ray_method
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        return await self.agent_loop.generate_sample(rollout_state, **kwargs)

    @ray_method
    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        return await self.agent_loop.generate_group(rollout_state, **kwargs)

    @ray_method
    async def get_rollout_ctl(self):
        return self.agent_loop.rollout_ctl


RayAgentLoop = cast(ActorClass[AgentLoopActor], CPUActorLauncher.to_actor_class(AgentLoopActor))
RayAgentLoopProxy: TypeAlias = ActorProxy[AgentLoopActor]
AgentLoopSpec: TypeAlias = AgentLoop | RayAgentLoopProxy | RouterAgentLoop
