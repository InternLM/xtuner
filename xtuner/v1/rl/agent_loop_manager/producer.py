import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger

from .produce_utils import (
    PERIODIC_ABORT_INTERVAL_S,
    BaseProduceContext,
    IsValidSampleFn,
    ShouldContinueFn,
    _ProgressDisplayer,
    _put_claimed_tasks,
    calculate_stale_threshold,
    default_is_valid_sample_fn,
    default_should_continue_fn,
    pause_pending_tasks,
)


if TYPE_CHECKING:
    from xtuner.v1.rl.rollout.controller import RolloutControllerProxy


logger = get_logger()


@dataclass
class ProduceProgress:
    """共卡单次 produce_batch 的局部指标，不进入 checkpoint。"""

    target_samples: dict[str, int] = field(default_factory=dict)
    raw_rewards_sum: dict[str, float] = field(default_factory=dict)
    raw_rewards_count: dict[str, int] = field(default_factory=dict)
    produced_samples: dict[str, int] = field(default_factory=dict)
    produced_tokens: dict[str, int] = field(default_factory=dict)
    failed_samples: dict[str, int] = field(default_factory=dict)
    filtered_samples: dict[str, int] = field(default_factory=dict)
    produce_time_s: float = 0.0

    @classmethod
    def build(
        cls,
        *,
        task_names: list[str],
        target_samples: dict[str, int],
    ) -> "ProduceProgress":
        return cls(
            target_samples=dict(target_samples),
            raw_rewards_sum={task_name: 0.0 for task_name in task_names},
            raw_rewards_count={task_name: 0 for task_name in task_names},
            produced_samples={task_name: 0 for task_name in task_names},
            produced_tokens={task_name: 0 for task_name in task_names},
            failed_samples={task_name: 0 for task_name in task_names},
            filtered_samples={task_name: 0 for task_name in task_names},
        )

    def add_raw_rewards(self, task_name: str, rewards_sum: float, rewards_count: int) -> None:
        self.raw_rewards_sum[task_name] += rewards_sum
        self.raw_rewards_count[task_name] += rewards_count

    def add_produced(self, task_name: str, samples: int, tokens: int) -> None:
        self.produced_samples[task_name] += samples
        self.produced_tokens[task_name] += tokens

    def add_discarded(self, task_name: str, status: Status, *, samples: int = 1) -> None:
        if status == Status.FAILED:
            self.failed_samples[task_name] += samples
            return
        if status == Status.FILTERED:
            self.filtered_samples[task_name] += samples
            return
        raise ValueError(f"Discarded status must be FAILED or FILTERED, got {status}.")

    def add_produce_time(self, elapsed_s: float) -> None:
        self.produce_time_s += elapsed_s

    def consume_produced(self, task_name: str) -> tuple[int, int]:
        samples = self.produced_samples[task_name]
        tokens = self.produced_tokens[task_name]
        self.produced_samples[task_name] = 0
        self.produced_tokens[task_name] = 0
        return samples, tokens

    def consume_produce_time(self) -> float:
        produce_time_s = self.produce_time_s
        self.produce_time_s = 0.0
        return produce_time_s

    def consume_discarded(self, task_name: str) -> tuple[int, int]:
        failed = self.failed_samples[task_name]
        filtered = self.filtered_samples[task_name]
        self.failed_samples[task_name] = 0
        self.filtered_samples[task_name] = 0
        return failed, filtered

    def consume_raw_rewards(self, task_name: str) -> tuple[float, int]:
        rewards_sum = self.raw_rewards_sum[task_name]
        rewards_count = self.raw_rewards_count[task_name]
        self.raw_rewards_sum[task_name] = 0.0
        self.raw_rewards_count[task_name] = 0
        return rewards_sum, rewards_count


@dataclass(kw_only=True)
class ProduceContext(BaseProduceContext):
    """共卡本地生产窗口；不暴露非共卡状态机字段。"""

    @property
    def batch_target(self) -> int:
        return self.progress.target_samples[self.task_name]

    async def completed_count(self) -> int:
        return await self.replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED)


class ProduceStrategyConfig(ABC, BaseModel):
    """Base configuration for rollout production strategies.

    Production strategies decide how the agent loop fills the replay buffer and
    when it should stop producing samples for the current training step.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    """Configuration for synchronous rollout production.

    The synchronous strategy produces samples on demand for the current training
    step. It is simpler and is the default choice when rollout and training run
    in a colocated or tightly synchronized workflow.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.

    **Examples:**

    Example synchronous strategy::

        config = SyncProduceStrategyConfig()
    """

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    """Configuration for colocated asynchronous rollout production.

    The colocated asynchronous strategy produces rollout samples concurrently
    within one ``AgentLoopManager.produce_batch`` call and stores them in the
    replay buffer. It can oversample, allow partial rollout continuation, and
    discard samples that are too stale relative to the current training step.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.
        over_sample_threshold (float): Extra completed-sample ratio allowed
            before the producer stops. Defaults to 0.0.
        enable_partial_rollout (bool): Whether unfinished rollouts can be
            continued after a weight sync. Defaults to False.
        max_staleness (int): Maximum allowed model-step staleness for replayed
            samples. Defaults to 0.
        tail_batch_trigger_size (int): Minimum pending tail size that can
            trigger a final batch. Defaults to 0.

    **Examples:**

    Example asynchronous strategy::

        config = AsyncProduceStrategyConfig(
            over_sample_threshold=0.2,
            enable_partial_rollout=True,
            max_staleness=1,
        )
    """

    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = Field(default=0, ge=0)
    tail_batch_trigger_size: int = 0

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "AsyncProduceStrategy":
        if rollout_controller is not None:
            import ray

            ray.get(rollout_controller.set_enable_partial_rollout.remote(self.enable_partial_rollout))
        return AsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(self, ctx: ProduceContext) -> None: ...

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return 0.0

    def pending_task_count(self) -> int:
        return 0


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(self, ctx: ProduceContext) -> None:
        pending_tasks = set()
        completed_sample_count = await ctx.replay_buffer.count(task_name=ctx.task_name, group_status=Status.COMPLETED)

        for _ in range(ctx.task_batch_size):
            rollout_state = await ctx.sampler.sample(task_name=ctx.task_name)
            task = create_task(ctx.generate_group(rollout_state))
            pending_tasks.add(task)

        logger.info(f"[SyncProduceStrategy] Started {len(pending_tasks)} initial tasks.")

        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.batch_target,
            initial=completed_sample_count,
        )
        while self.should_continue_fn(completed_sample_count, ctx.task_batch_size):
            if not pending_tasks:
                logger.warning("[SyncProduceStrategy] All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            # put_generated_group 负责过滤和入库。
            for task in done_tasks:
                items = task.result()

                is_completed = await ctx.put_generated_group(items)
                if not is_completed:
                    continue

                completed_sample_count += 1
                progress_displayer.update(completed_sample_count)

            while len(pending_tasks) + completed_sample_count < ctx.task_batch_size and self.should_continue_fn(
                completed_sample_count, ctx.task_batch_size
            ):
                rollout_state = await ctx.sampler.sample(task_name=ctx.task_name)
                task = create_task(ctx.generate_group(rollout_state))
                pending_tasks.add(task)
        progress_displayer.close()


class AsyncProduceStrategy(ProduceStrategy):
    PERIODIC_ABORT_INTERVAL_S = PERIODIC_ABORT_INTERVAL_S

    def __init__(
        self,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        tail_batch_trigger_size: int,
        max_staleness: int,
        sync_weights_interval: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn)

        # TODO: 需要添加 tail_batch_max_tries
        # 作用是：如果一个样本多次重试，则将它置为特殊状态 MAX_TRIES，这类样本和过期样本一起触发tail batch逻辑
        # 这个依赖：RolloutState 添加并维护一个新的属性 num_tries，每次打断时加1，达到 max_tries 时置为 MAX_TRIES
        # 如果 enable_partial_rollout=True，不会触发这个逻辑，所以不受此影响
        # 如果 enable_partial_rollout=False，分两种情况：
        # 1) staleness = 0，即不允许过期样本，此时过期触发tail batch逻辑已经cover了tail batch逻辑
        # 2) staleness > 0，此时需要 重试tail batch逻辑，否则多次重试的样本会影响rollout 效率
        if not enable_partial_rollout and max_staleness > 0:
            logger.warning(
                "max_staleness > 0, enable_partial_rollout is False, this will affect rollout efficiency because not support tail_batch_max_tries logic now"
            )

        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self._local_pending_tasks: set[asyncio.Task] = set()

    def pending_task_count(self) -> int:
        return len(self._local_pending_tasks)

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return await pause_pending_tasks(
            pending_tasks=self._local_pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: ctx.put_generated_group(task.result()),
        )

    async def produce_batch(self, ctx: ProduceContext) -> None:
        if ctx.task_name not in ctx.progress.target_samples:
            raise KeyError(f"ProduceProgress.target_samples missing task_name={ctx.task_name!r}")

        # 共卡 async 的 pending 只属于本次 produce_batch。
        self._local_pending_tasks = set()

        if ctx.batch_target <= 0:
            return

        expired_count = await ctx.expired_count()
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size
        if sample_from_expired:
            logger.info(
                f"Tail batch trigger condition met: {expired_count} expired samples "
                f"(threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )

        # normal 使用固定超发预算；tail-batch 只补必要缺口。
        batch_target = ctx.batch_target
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = batch_target + oversample_budget
        logger.info(
            f"Starting produce_batch for task {ctx.task_name} with batch_target={batch_target}, "
            f"oversample_budget={oversample_budget}, scheduled_target={scheduled_target}."
        )

        async def spawn_one() -> asyncio.Task:
            rollout_state = await ctx.sample_group(from_expired_pool=sample_from_expired)
            return create_task(
                ctx.generate_group(
                    rollout_state,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )

        initial_available = await ctx.completed_count()
        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.batch_target,
            initial=initial_available,
        )
        while True:
            available = await ctx.completed_count()
            progress_displayer.update(available)
            if not self.should_continue_fn(available, batch_target):
                break

            pending_count = len(self._local_pending_tasks)
            desired_pending = max(0, scheduled_target - available)
            if available + pending_count < scheduled_target:
                while len(self._local_pending_tasks) < desired_pending:
                    self._local_pending_tasks.add(await spawn_one())

            if not self._local_pending_tasks:
                logger.warning("All tasks are done but not enough samples collected.")
                break

            done_tasks, _ = await asyncio.wait(
                set(self._local_pending_tasks), timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            self._local_pending_tasks.difference_update(done_tasks)
            await _put_claimed_tasks(
                done_tasks,
                ctx,
                available_base=available,
                progress_displayer=progress_displayer,
            )
        progress_displayer.close()
