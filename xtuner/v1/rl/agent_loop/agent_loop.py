from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias, cast, overload

import ray
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status, get_group_status
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherClient,
    route_rollout_teacher_client,
)
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.constants import AGENT_LOOP_RAY_GENERATE_MAX_CONCURRENCY
from xtuner.v1.rl.trace.rollout_api import (
    trace_rollout_endpoint,
)
from xtuner.v1.rl.utils import (
    JUDGER_PAUSE_JUDGE_TASK_TIMEOUT_S,
    CPUActorLauncher,
    CPUResourcesConfig,
    cancel_and_drain,
    create_task,
    register_cpu_resources,
)
from xtuner.v1.utils import get_logger, ray_method
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


AGENT_LOOP_CONCURRENCY_GROUP_GENERATE = "generate"
IsValidSampleFn: TypeAlias = Callable[[list[RolloutState]], bool]


def maybe_filter_invalid_sample(
    group: list[RolloutState],
    is_valid_sample_fn: IsValidSampleFn | None,
    logger,
) -> list[RolloutState]:
    """Apply task-specific group validation after generation and judging.

    Teacher scoring may run before this helper when no validity check is configured. When a validity check is
    configured, call this helper before sending the completed group to the Teacher.
    """
    if get_group_status(group) != Status.COMPLETED:
        return group
    if is_valid_sample_fn is None or is_valid_sample_fn(group):
        return group

    for state in group:
        state.status = Status.FILTERED
    group_id = group[0].group_id if group else None
    rollout_ids = [state.rollout_id for state in group]
    logger.info(f"Filtered invalid rollout group: group_id={group_id}, rollout_ids={rollout_ids}.")
    return group


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams | None = None
    cpu_resources: CPUResourcesConfig | None = None
    enable_batch_judge: bool = False
    requires_rollout_proxy: bool = False

    def build(
        self,
        rollout_controller,
        judger: Judger | None = None,
        logger=None,
        *,
        is_valid_sample_fn: IsValidSampleFn | None = None,
        distillation_config: DistillationConfig | None = None,
    ) -> AgentLoopSpec:
        if self.cpu_resources is None:
            agent_loop = self.build_local(
                rollout_controller=rollout_controller,
                judger=judger,
                logger=logger,
            )
            agent_loop.is_valid_sample_fn = is_valid_sample_fn
            agent_loop.configure_distillation(distillation_config)
            return agent_loop

        concurrency = AGENT_LOOP_RAY_GENERATE_MAX_CONCURRENCY

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
                is_valid_sample_fn=is_valid_sample_fn,
                distillation_config=distillation_config,
            )
        return self._build_ray_actor(
            rollout_controller=rollout_controller,
            cpu_resources=self.cpu_resources,
            concurrency=concurrency,
            judger=judger,
            logger=logger,
            is_valid_sample_fn=is_valid_sample_fn,
            distillation_config=distillation_config,
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
        is_valid_sample_fn: IsValidSampleFn | None = None,
        distillation_config: DistillationConfig | None = None,
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
                is_valid_sample_fn=is_valid_sample_fn,
                distillation_config=distillation_config,
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
        is_valid_sample_fn: IsValidSampleFn | None = None,
        distillation_config: DistillationConfig | None = None,
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
                is_valid_sample_fn=is_valid_sample_fn,
                distillation_config=distillation_config,
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
        is_valid_sample_fn: IsValidSampleFn | None = None,
        distillation_config: DistillationConfig | None = None,
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
                is_valid_sample_fn=is_valid_sample_fn,
                distillation_config=distillation_config,
            ),
            rollout_ctl=rollout_controller,
        )


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController | None,
        sample_params: SampleParams | None,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = False,
    ) -> None:
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params: SampleParams = sample_params if sample_params is not None else SampleParams()
        self.judger = judger
        self.enable_batch_judge = enable_batch_judge
        self.is_valid_sample_fn: IsValidSampleFn | None = None
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
        self._judger_pause_event = asyncio.Event()
        self.teacher_clients: dict[str, RolloutTeacherClient] = {}
        self.data_source_teacher_map: dict[str, str] = {}

    def configure_distillation(self, distillation_config: DistillationConfig | None) -> None:
        if distillation_config is None:
            return

        self.teacher_clients = {
            teacher.name: RolloutTeacherClient(teacher, distillation_config.loss_config)
            for teacher in distillation_config.rollout_teachers
        }
        self.data_source_teacher_map = dict(distillation_config.data_source_teacher_map)

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        """Generate one rollout sample without group-level post-processing."""

        ...

    async def maybe_compute_teacher_logprob(self, state: RolloutState) -> RolloutState:
        if state.status != Status.COMPLETED or not self.teacher_clients:
            return state
        teacher = route_rollout_teacher_client(
            state,
            data_source_teacher_map=self.data_source_teacher_map,
            teacher_clients=self.teacher_clients,
        )
        return await teacher.compute_logprobs(state)

    async def maybe_compute_teacher_logprobs(self, group: list[RolloutState]) -> list[RolloutState]:
        return list(await asyncio.gather(*(create_task(self.maybe_compute_teacher_logprob(state)) for state in group)))

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        """Generate one rollout group.

        Warning:
            Subclasses overriding this method must preserve the Teacher/Judger/filter
            ordering described here. Without a validity check, Teacher scoring should
            start as soon as each sample finishes generation. With a validity check,
            only a completed group that passes filtering should be sent to the Teacher.
        """
        filter_before_teacher = self.is_valid_sample_fn is not None

        async def generate_one(state: RolloutState) -> RolloutState:
            state.sample_params = self.sample_params
            state = await self.generate_sample(state, **kwargs)
            # Fast path: overlap Teacher scoring with the remaining generations.
            if not filter_before_teacher:
                state = await self.maybe_compute_teacher_logprob(state)
            if state.status == Status.COMPLETED and self.judger is not None and not self.enable_batch_judge:
                state = await self.run_judger(state)
            return state

        group = list(await asyncio.gather(*(create_task(generate_one(state)) for state in rollout_state)))
        if self.judger is not None and self.enable_batch_judge and get_group_status(group) == Status.COMPLETED:
            group = await self.run_judger(group)

        group = maybe_filter_invalid_sample(group, self.is_valid_sample_fn, self.logger)
        # Filter path: avoid Teacher work for a group that will be discarded.
        if filter_before_teacher and get_group_status(group) == Status.COMPLETED:
            group = await self.maybe_compute_teacher_logprobs(group)
        return group

    @overload
    async def run_judger(self, rollout_state: RolloutState) -> RolloutState: ...

    @overload
    async def run_judger(self, rollout_state: list[RolloutState]) -> list[RolloutState]: ...

    @trace_rollout_endpoint("judger.run")
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
                    timeout=JUDGER_PAUSE_JUDGE_TASK_TIMEOUT_S,
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
            rollout_ctl = self.rollout_ctl
            if rollout_ctl is None:
                return
            await cast(Any, rollout_ctl.pause_generation).remote()
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
        is_valid_sample_fn: IsValidSampleFn | None = None,
        *,
        distillation_config: DistillationConfig | None = None,
    ):
        self.agent_loop = agent_loop_config.build_local(
            rollout_controller=rollout_controller,
            judger=judger,
            logger=logger,
        )
        self.agent_loop.is_valid_sample_fn = is_valid_sample_fn
        self.agent_loop.configure_distillation(distillation_config)

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
