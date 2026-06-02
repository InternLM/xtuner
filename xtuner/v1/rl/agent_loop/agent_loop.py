from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from typing import TypeAlias, cast, overload

import ray
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import (
    CPUActorLauncher,
    CPUResourcesConfig,
    cancel_and_drain,
    create_task,
    register_cpu_resources,
)
from xtuner.v1.utils import get_logger, ray_method
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


AGENT_LOOP_CONCURRENCY_GROUP_GENERATE = "generate"
DEFAULT_JUDGER_CANCEL_TIMEOUT_S = 5.0


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams
    cpu_resources: CPUResourcesConfig | None = None
    enable_batch_judge: bool = False

    def build(self, rollout_controller, judger: Judger | None = None, logger=None) -> AgentLoopSpec:
        if self.cpu_resources is None:
            return self.build_local(
                rollout_controller=rollout_controller,
                judger=judger,
                logger=logger,
            )

        get_generate_concurrency = rollout_controller.get_generate_concurrency
        if hasattr(get_generate_concurrency, "remote"):
            total_generate_concurrency = ray.get(get_generate_concurrency.remote())
        else:
            total_generate_concurrency = get_generate_concurrency()
        concurrency = max(1, math.ceil(total_generate_concurrency / self.cpu_resources.num_workers))

        register_cpu_resources(
            name=f"agent_loop:{self.__class__.__name__}",
            cpu_resources=self.cpu_resources,
        )

        if self.cpu_resources.num_workers > 1:
            return self._build_router(
                rollout_controller=rollout_controller,
                cpu_resources=self.cpu_resources,
                concurrency=concurrency,
                judger=judger,
                logger=logger,
            )
        return self._build_ray_actor(
            rollout_controller=rollout_controller,
            cpu_resources=self.cpu_resources,
            concurrency=concurrency,
            judger=judger,
            logger=logger,
        )

    @abstractmethod
    def build_local(
        self,
        rollout_controller,
        judger: Judger | None = None,
        logger=None,
    ) -> AgentLoop: ...

    def _build_ray_actor(
        self,
        rollout_controller: RolloutController,
        cpu_resources: CPUResourcesConfig,
        concurrency: int,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> RayAgentLoopProxy:
        ray_agent_loop = ray.remote(
            concurrency_groups={
                AGENT_LOOP_CONCURRENCY_GROUP_GENERATE: concurrency,
            },
        )(AgentLoopActor)
        return cast(
            "RayAgentLoopProxy",
            CPUActorLauncher.build_actor(
                ray_agent_loop,
                self,
                rollout_controller,
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
        cpu_resources: CPUResourcesConfig,
        concurrency: int,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> list[RayAgentLoopProxy]:
        ray_agent_loop = ray.remote(
            concurrency_groups={
                AGENT_LOOP_CONCURRENCY_GROUP_GENERATE: concurrency,
            },
        )(AgentLoopActor)
        return cast(
            list["RayAgentLoopProxy"],
            CPUActorLauncher.build_actors(
                ray_agent_loop,
                self,
                rollout_controller,
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
        cpu_resources: CPUResourcesConfig,
        concurrency: int,
        pg: PlacementGroup | None = None,
        judger: Judger | None = None,
        logger=None,
        start_bundle_idx: int = 0,
    ) -> RouterAgentLoop:
        return RouterAgentLoop(
            workers=self._build_ray_actors(
                rollout_controller=rollout_controller,
                cpu_resources=cpu_resources,
                concurrency=concurrency,
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
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = False,
    ) -> None:
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judger = judger
        self.enable_batch_judge = enable_batch_judge
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
        self._judger_pause_event = asyncio.Event()

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
        if self.judger is not None and self.enable_batch_judge:
            if not any(sample.status == Status.ABORTED for sample in group_samples):
                group_samples = await self.run_judger(group_samples)
        return group_samples

    @overload
    async def run_judger(self, rollout_state: RolloutState) -> RolloutState: ...

    @overload
    async def run_judger(self, rollout_state: list[RolloutState]) -> list[RolloutState]: ...

    async def run_judger(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:
        assert self.judger is not None
        if isinstance(rollout_state, list):
            judge_task = create_task(self.judger.batch_judge(rollout_state))
        else:
            judge_task = create_task(self.judger.judge(rollout_state))
        pause_task = create_task(self._judger_pause_event.wait())
        try:
            done, _ = await asyncio.wait({judge_task, pause_task}, return_when=asyncio.FIRST_COMPLETED)
            if judge_task in done:
                return await judge_task
            try:
                return await asyncio.wait_for(
                    asyncio.shield(judge_task),
                    timeout=DEFAULT_JUDGER_CANCEL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await cancel_and_drain([judge_task])
                for sample in rollout_state if isinstance(rollout_state, list) else [rollout_state]:
                    sample.status = Status.ABORTED
                    sample.finish_reason = "abort"
                    sample.reward = None
                return rollout_state
        except asyncio.CancelledError:
            await cancel_and_drain([judge_task])
            for sample in rollout_state if isinstance(rollout_state, list) else [rollout_state]:
                sample.status = Status.ABORTED
                sample.finish_reason = "abort"
                sample.reward = None
            return rollout_state
        finally:
            await cancel_and_drain([pause_task])

    async def pause(self) -> None:
        self._judger_pause_event.set()
        try:
            await self.rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]
        finally:
            self._judger_pause_event.clear()


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

    async def pause(self) -> None:
        await asyncio.gather(
            *(worker.pause.remote() for worker in self.workers),
        )


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
        judger: Judger | None = None,
        logger=None,
    ):
        self.agent_loop = agent_loop_config.build_local(
            rollout_controller=rollout_controller,
            judger=judger,
            logger=logger,
        )

    @ray_method(concurrency_group=AGENT_LOOP_CONCURRENCY_GROUP_GENERATE)
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        return await self.agent_loop.generate_sample(rollout_state, **kwargs)

    @ray_method(concurrency_group=AGENT_LOOP_CONCURRENCY_GROUP_GENERATE)
    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        return await self.agent_loop.generate_group(rollout_state, **kwargs)

    @ray_method
    async def get_rollout_ctl(self):
        return self.agent_loop.rollout_ctl

    @ray_method
    async def pause(self) -> None:
        return await self.agent_loop.pause()


RayAgentLoop = cast(
    ActorClass[AgentLoopActor],
    ray.remote(
        concurrency_groups={
            AGENT_LOOP_CONCURRENCY_GROUP_GENERATE: 1000,
        },
    )(AgentLoopActor),
)
RayAgentLoopProxy: TypeAlias = ActorProxy[AgentLoopActor]
AgentLoopSpec: TypeAlias = AgentLoop | RayAgentLoopProxy | RouterAgentLoop
