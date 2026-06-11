import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.utils import calculate_seq_staleness, create_task
from xtuner.v1.utils import get_logger

from .produce_utils import (
    PERIODIC_ABORT_INTERVAL_S,
    BaseProduceContext,
    IsValidSampleFn,
    ProduceBatchStatus,
    ShouldContinueFn,
    _PendingTasks,
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
class DisaggProduceProgress:
    """非共卡 producer / consumer 共享的绝对进度。"""

    task_names: list[str] = field(default_factory=list)
    producer_future_step: int = 1
    next_consumer_step: int = 1
    target_upto_future_step: int = 0
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    raw_rewards_sum: dict[str, float] = field(default_factory=dict)
    raw_rewards_count: dict[str, int] = field(default_factory=dict)
    produced_samples: dict[str, int] = field(default_factory=dict)
    produced_tokens: dict[str, int] = field(default_factory=dict)
    produce_time_s: float = 0.0

    @classmethod
    def build(cls, task_names: list[str]) -> "DisaggProduceProgress":
        return cls(
            task_names=list(task_names),
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples={task_name: 0 for task_name in task_names},
            raw_rewards_sum={task_name: 0.0 for task_name in task_names},
            raw_rewards_count={task_name: 0 for task_name in task_names},
            produced_samples={task_name: 0 for task_name in task_names},
            produced_tokens={task_name: 0 for task_name in task_names},
        )

    def ensure_target_upto(
        self,
        *,
        batch_size: int,
        future_step: int,
        allocate_batch_sizes: Callable[[int, int], dict[str, int]],
    ) -> dict[str, int]:
        """把累计 target 推进到指定 future step，并返回该 step 的 task batch size。"""

        current_task_batch_sizes: dict[str, int] | None = None
        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                current_task_batch_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, task_batch_size in current_task_batch_sizes.items():
                    self.target_samples[task_name] += task_batch_size
            self.target_upto_future_step = future_step

        if current_task_batch_sizes is None:
            current_task_batch_sizes = allocate_batch_sizes(batch_size, future_step)
        return current_task_batch_sizes

    def begin_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step

    def mark_consumed(self, consumed_counts: dict[str, int]) -> None:
        # target 不回退；producer 用 consumed + completed 判断真实缺口。
        for task_name, count in consumed_counts.items():
            self.consumed_samples[task_name] += count

    def finish_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step + 1

    def advance_future_step(self) -> None:
        self.producer_future_step += 1

    def add_raw_rewards(self, task_name: str, rewards_sum: float, rewards_count: int) -> None:
        self.raw_rewards_sum[task_name] += rewards_sum
        self.raw_rewards_count[task_name] += rewards_count

    def add_produced(self, task_name: str, samples: int, tokens: int) -> None:
        self.produced_samples[task_name] += samples
        self.produced_tokens[task_name] += tokens

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

    def consume_raw_rewards(self, task_name: str) -> tuple[float, int]:
        rewards_sum = self.raw_rewards_sum[task_name]
        rewards_count = self.raw_rewards_count[task_name]
        self.raw_rewards_sum[task_name] = 0.0
        self.raw_rewards_count[task_name] = 0
        return rewards_sum, rewards_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "producer_future_step": self.producer_future_step,
            "next_consumer_step": self.next_consumer_step,
            "target_upto_future_step": self.target_upto_future_step,
            "consumed_samples": dict(self.consumed_samples),
            "target_samples": dict(self.target_samples),
            "raw_rewards_sum": dict(self.raw_rewards_sum),
            "raw_rewards_count": dict(self.raw_rewards_count),
            "produced_samples": dict(self.produced_samples),
            "produced_tokens": dict(self.produced_tokens),
            "produce_time_s": self.produce_time_s,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # 原地更新，避免 strategy / context 持有旧引用。
        self.producer_future_step = state["producer_future_step"]
        self.next_consumer_step = state["next_consumer_step"]
        self.target_upto_future_step = state["target_upto_future_step"]
        self.consumed_samples.clear()
        self.consumed_samples.update(state["consumed_samples"])
        self.target_samples.clear()
        self.target_samples.update(state["target_samples"])
        task_names = set(self.consumed_samples) | set(self.target_samples)
        self.raw_rewards_sum.clear()
        self.raw_rewards_sum.update(
            {task_name: float(state.get("raw_rewards_sum", {}).get(task_name, 0.0)) for task_name in task_names}
        )
        self.raw_rewards_count.clear()
        self.raw_rewards_count.update(
            {task_name: int(state.get("raw_rewards_count", {}).get(task_name, 0)) for task_name in task_names}
        )
        produced_samples_state = state.get("produced_samples", {})
        produced_tokens_state = state.get("produced_tokens", {})
        self.produced_samples.clear()
        self.produced_samples.update(
            {task_name: int(produced_samples_state.get(task_name, 0)) for task_name in task_names}
        )
        self.produced_tokens.clear()
        self.produced_tokens.update(
            {task_name: int(produced_tokens_state.get(task_name, 0)) for task_name in task_names}
        )
        self.produce_time_s = float(state.get("produce_time_s", 0.0))


@dataclass(kw_only=True)
class DisaggProduceContext(BaseProduceContext):
    """非共卡后台生产上下文。"""

    progress: DisaggProduceProgress
    update_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def consumer_step(self) -> int:
        return self.progress.next_consumer_step

    @property
    def total_target(self) -> int:
        return self.progress.target_samples[self.task_name]

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    async def available_count(self) -> int:
        completed_count = await self.replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed_count


class DisaggProduceStrategyConfig(ABC, BaseModel):
    """非共卡后台 producer strategy 配置。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "DisaggProduceStrategy": ...


class DisaggAsyncProduceStrategyConfig(DisaggProduceStrategyConfig):
    """非共卡异步生产配置。"""

    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = Field(default=0, ge=0)
    tail_batch_trigger_size: int = 0

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "DisaggAsyncProduceStrategy":
        if rollout_controller is not None:
            import ray

            ray.get(rollout_controller.set_enable_partial_rollout.remote(self.enable_partial_rollout))
        return DisaggAsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class DisaggProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(self, ctx: DisaggProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: DisaggProduceContext) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False

    def pending_task_count(self) -> int:
        return 0


class DisaggAsyncProduceStrategy(DisaggProduceStrategy):
    """非共卡 async strategy；pending 跨后台生产轮次存在。"""

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
        self._pending_tasks = _PendingTasks()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        staleness = calculate_seq_staleness(model_step, train_step)
        return staleness >= self.stale_threshold

    def pending_task_count(self) -> int:
        return self._pending_tasks.count()

    async def pause_produce(self, ctx: DisaggProduceContext) -> float:
        return await pause_pending_tasks(
            pending_tasks=self._pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: ctx.put_generated_group(task.result()),
        )

    async def produce_batch(self, ctx: DisaggProduceContext) -> ProduceBatchStatus:
        if ctx.task_name not in ctx.progress.consumed_samples:
            raise KeyError(f"DisaggProduceProgress.consumed_samples missing task_name={ctx.task_name!r}")
        if ctx.task_name not in ctx.progress.target_samples:
            raise KeyError(f"DisaggProduceProgress.target_samples missing task_name={ctx.task_name!r}")

        if ctx.total_target <= 0:
            return ProduceBatchStatus.NORMAL

        # TODO: place this check just before while loop
        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.train_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        # 进入下一轮前先回收已完成的旧 pending。
        claimed_done = await self._pending_tasks.claim_ready()
        await _put_claimed_tasks(claimed_done, ctx)

        # TODO: remove this check
        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.train_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        expired_count = await ctx.expired_count()
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size
        if sample_from_expired:
            logger.info(
                f"Tail batch trigger condition met: {expired_count} expired samples "
                f"(threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )

        # normal 使用固定超发预算；tail-batch 只补必要缺口。
        total_target = ctx.total_target
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = total_target + oversample_budget
        logger.info(
            f"Starting produce_batch for task {ctx.task_name} with total_target={total_target}, "
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

        initial_available = await ctx.available_count()
        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.total_target,
            initial=initial_available,
        )
        produce_status = ProduceBatchStatus.NORMAL
        while True:
            if ctx.should_abort():
                produce_status = ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                break
            if self.is_model_expired(ctx.train_step, ctx.model_step):
                produce_status = ProduceBatchStatus.EXPIRED_BATCH
                break

            available = await ctx.available_count()
            progress_displayer.update(available)
            if not self.should_continue_fn(available, total_target):
                break

            pending_count = self._pending_tasks.count()
            desired_pending = max(0, scheduled_target - available)
            if available + pending_count < scheduled_target:
                while await self._pending_tasks.schedule_one(
                    max_pending=desired_pending,
                    should_abort=ctx.should_abort,
                    spawn_one=spawn_one,
                ):
                    pass
                if ctx.should_abort():
                    produce_status = ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                    break

            # TODO: remove this check, because will check it when exit if statement, it's redundant
            if ctx.should_abort():
                produce_status = ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                break
            if self._pending_tasks.count() == 0:
                logger.warning("All tasks are done but not enough samples collected.")
                break

            claimed_done = await self._pending_tasks.wait_and_claim(timeout_s=1)
            await _put_claimed_tasks(
                claimed_done,
                ctx,
                available_base=available,
                progress_displayer=progress_displayer,
            )
        progress_displayer.close()
        return produce_status
