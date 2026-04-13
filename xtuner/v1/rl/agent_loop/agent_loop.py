from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias, cast

import ray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto import RolloutState, SampleParams
from xtuner.v1.rl.judger import Judger, JudgerSpec
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import CPUActorLauncher, create_task
from xtuner.v1.utils import get_logger, ray_method
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams
    type: Literal["local", "ray.actor"] = "local"
    num_ray_actors: int = Field(default=1, ge=1, description="Number of AgentLoop Ray actor instances.")
    num_cpus: float = Field(default=1, gt=0, description="CPU cores required by the AgentLoop actor itself.")
    cpu_memory: int = Field(default=1024**3, gt=0, description="CPU memory in bytes required by AgentLoop.")

    @model_validator(mode="after")
    def _validate_ray_actor_config(self) -> AgentLoopConfig:
        if self.type == "local" and self.num_ray_actors != 1:
            logger = get_logger()
            logger.warning("num_ray_actors will be ignored when AgentLoop type is 'local'.")
        return self

    def build(self, rollout_controller, judger: JudgerSpec = None, logger=None) -> AgentLoopSpec:
        if self.type == "local":
            return self.build_local(
                rollout_controller=rollout_controller,
                judger=judger,
                logger=logger,
            )
        if self.type == "ray.actor":
            if self.num_ray_actors > 1:
                return self._build_router(
                    rollout_controller=rollout_controller,
                    judger=judger,
                    logger=logger,
                )
            return self._build_ray_actor(
                rollout_controller=rollout_controller,
                judger=judger,
                logger=logger,
            )
        raise ValueError(f"Invalid agent loop type: {self.type}")

    @abstractmethod
    def build_local(
        self,
        rollout_controller,
        judger: JudgerSpec = None,
        logger=None,
    ) -> AgentLoop: ...

    def _build_ray_actor(
        self,
        rollout_controller: RolloutController,
        pg: PlacementGroup | None = None,
        judger: JudgerSpec = None,
        logger=None,
    ) -> RayAgentLoopProxy:
        return cast(
            "RayAgentLoopProxy",
            CPUActorLauncher.build_actor(
                AgentLoopActor,
                self,
                rollout_controller,
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
        rollout_controller: RolloutController,
        num_actors: int,
        pg: PlacementGroup | None = None,
        judger: JudgerSpec = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> list[RayAgentLoopProxy]:
        return cast(
            list["RayAgentLoopProxy"],
            CPUActorLauncher.build_actors(
                AgentLoopActor,
                self,
                rollout_controller,
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
        rollout_controller: RolloutController,
        pg: PlacementGroup | None = None,
        judger: JudgerSpec = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> RouterAgentLoop:
        return RouterAgentLoop(
            workers=self._build_ray_actors(
                rollout_controller=rollout_controller,
                num_actors=self.num_ray_actors,
                pg=pg,
                judger=judger,
                logger=logger,
                start_bundle_idx=start_bundle_idx,
            ),
            rollout_ctl=rollout_controller,
        )

    def build_ray_actor_list(
        self,
        rollout_controller,
        num_actors: int,
        judger: JudgerSpec = None,
        logger=None,
        pg: PlacementGroup | None = None,
    ) -> list[RayAgentLoopProxy]:
        if self.type != "ray.actor":
            raise ValueError(f"build_ray_actor_list requires type='ray.actor', got {self.type}")
        if num_actors <= 0:
            raise ValueError(f"num_actors must be positive, got {num_actors}")
        return self._build_ray_actors(
            rollout_controller=rollout_controller,
            pg=pg,
            num_actors=num_actors,
            judger=judger,
            logger=logger,
        )


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: JudgerSpec = None,
        logger=None,
    ) -> None:
        self.rollout_ctl = rollout_ctl
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

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples

    async def judge_sample(self, rollout_state: RolloutState) -> RolloutState:
        if self.judger is None:
            return rollout_state

        judger = self.judger
        if isinstance(judger, dict):
            if len(judger) > 1:
                raise NotImplementedError("Multiple judgers require a custom AgentLoop.judge_sample implementation.")
            judger = next(iter(judger.values()))

        if isinstance(judger, Judger):
            rollout_state = await judger.judge(rollout_state)
        elif isinstance(judger, ray.actor.ActorHandle):
            rollout_state = await judger.judge.remote(rollout_state)
        elif callable(judger):
            judger_result = judger(rollout_state)
            if inspect.isawaitable(judger_result):
                rollout_state = await judger_result
            else:
                rollout_state = judger_result
        else:
            raise ValueError(f"Invalid judger type: {type(judger)}")

        if not isinstance(rollout_state, RolloutState):
            raise TypeError(f"Judger must return RolloutState, but got {type(rollout_state)}")
        return rollout_state


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
        judger: JudgerSpec = None,
        logger=None,
    ):
        self.agent_loop = agent_loop_config.build_local(
            rollout_controller=rollout_controller,
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
