from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias, cast, overload

import ray
import torch
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status, get_group_status
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


def normalize_token_ids(value: Any) -> list[int]:
    """Convert list-like token IDs, including CPU tensors, to Python
    integers."""
    if value is None:
        raise ValueError("token IDs must be provided.")
    if isinstance(value, torch.Tensor):
        value = value.flatten().tolist()
    return [int(item) for item in value]


def validate_training_artifacts(rollout_state: RolloutState) -> None:
    """Validate the final fields required by the train worker."""
    if rollout_state.input_ids is None or rollout_state.labels is None:
        raise ValueError("training artifacts must provide input_ids and labels.")
    if len(rollout_state.input_ids) != len(rollout_state.labels):
        raise ValueError("training artifacts input_ids and labels must have the same length.")
    if rollout_state.logprobs is not None and len(rollout_state.logprobs) != len(rollout_state.input_ids):
        raise ValueError("training artifacts logprobs and input_ids must have the same length.")


def mark_training_artifacts_failed(rollout_state: RolloutState, error: Exception, logger) -> RolloutState:
    """Mark artifact preparation failure on a rollout state without raising
    it."""
    rollout_state.status = Status.FAILED
    rollout_state.finish_reason = "error"
    rollout_state.error_msg = f"{type(error).__name__}: {error}"
    logger.error(
        f"[AgentLoop] failed to prepare training artifacts for rollout_id={rollout_state.rollout_id}: "
        f"{type(error).__name__}: {error}"
    )
    return rollout_state


def maybe_filter_invalid_sample(
    group: list[RolloutState],
    is_valid_sample_fn: IsValidSampleFn | None,
    logger,
) -> list[RolloutState]:
    """Finalize rollout-group validity after all generation post-processing.

    Every custom ``AgentLoop.generate_group`` implementation must return through
    this helper after generation, judging, flattening, and failure cleanup::

        # Keep sample validation as the final group-generation step.
        return maybe_filter_invalid_sample(
            samples,
            self.is_valid_sample_fn,
            self.logger,
        )
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
    ) -> AgentLoopSpec:
        if self.cpu_resources is None:
            agent_loop = self.build_local(
                rollout_controller=rollout_controller,
                judger=judger,
                logger=logger,
            )
            agent_loop.is_valid_sample_fn = is_valid_sample_fn
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
            )
        return self._build_ray_actor(
            rollout_controller=rollout_controller,
            cpu_resources=self.cpu_resources,
            concurrency=concurrency,
            judger=judger,
            logger=logger,
            is_valid_sample_fn=is_valid_sample_fn,
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

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState: ...

    @abstractmethod
    async def prepare_training_artifacts(self, rollout_state: RolloutState) -> RolloutState:
        """Prepare and validate the training fields for one completed rollout.

        Args:
            rollout_state (RolloutState): Completed rollout state to prepare.

        Returns:
            RolloutState: The rollout state with canonical training fields.
        """
        ...

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        """Generate one rollout group.

        Warning:
            Subclasses overriding this method MUST call
            ``maybe_filter_invalid_sample`` as the final step before returning.
            Otherwise ``TaskSpecConfig.is_valid_sample_fn`` will be silently ignored.
        """
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        if self.judger is not None and self.enable_batch_judge:
            if all(sample.status == Status.COMPLETED for sample in group_samples):
                group_samples = await self.run_judger(group_samples)
        # Filter completed groups before preparing training artifacts.
        group_samples = maybe_filter_invalid_sample(group_samples, self.is_valid_sample_fn, self.logger)
        return await asyncio.gather(*(self.prepare_training_artifacts(sample) for sample in group_samples))

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
    ):
        self.agent_loop = agent_loop_config.build_local(
            rollout_controller=rollout_controller,
            judger=judger,
            logger=logger,
        )
        self.agent_loop.is_valid_sample_fn = is_valid_sample_fn

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
