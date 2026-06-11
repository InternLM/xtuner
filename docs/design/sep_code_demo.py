"""共卡 / 非共卡生产代码拆分伪代码。

说明：
- 这是设计伪代码，用来展示 Module、Interface 和 Adapter 关系，不是可直接运行实现。
- 重点是把共卡同步生产和非共卡 Background Producer / Training Consumer 分开。
- 共卡 AsyncProduceStrategyConfig 和非共卡 DisaggAsyncProduceStrategyConfig 是不同配置类型，
  不在 strategy config.build(...) 里用 mode 切换。
"""

from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeAlias


class Status(Enum):
    INIT = auto()
    COMPLETED = auto()
    ABORTED = auto()
    EXPIRED = auto()
    FAILED = auto()
    FILTERED = auto()


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()


ProducerMode: TypeAlias = Literal["colocate", "disaggregated"]


class DisaggManagerStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


def get_group_status(group: list[Any]) -> Status:
    """聚合 rollout group 状态。

    这里只读状态，不修改样本。过滤和过期翻转必须发生在显式业务逻辑里。
    """

    ...


def calculate_seq_staleness(model_step: int, train_step: int) -> int:
    ...


AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S = 10.0
PERIODIC_ABORT_INTERVAL_S = 5.0
PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S = 60.0


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    return (max_staleness + 1) * sync_weights_interval


@dataclass
class ProduceBatchResult:
    rollout_states: list[list[Any]]
    status: ProduceBatchStatus = ProduceBatchStatus.NORMAL
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    leftover_init: int = 0
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0
    leftover_failed: int = 0
    leftover_filtered: int = 0
    raw_rewards_sum: float = 0.0
    raw_rewards_count: int = 0
    produced_samples: int = 0
    produced_tokens: int = 0
    produce_time_s: float = 0.0
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


class ReplayBuffer(Protocol):
    async def put(
        self,
        group: list[Any],
        task_name: str,
        *,
        model_step: int | None = None,
        current_train_step: int | None = None,
        stale_threshold: int | None = None,
    ) -> None: ...

    async def count(self, task_name: str, group_status: Status) -> int: ...

    async def refresh_staleness(
        self,
        *,
        task_stale_thresholds: dict[str, int],
        current_train_step: int,
        statuses: list[Status],
    ) -> dict[str, int]: ...

    async def is_ready(self, task_batch_sizes: dict[str, int]) -> bool: ...

    async def take_batch(
        self,
        task_batch_sizes: dict[str, int],
    ) -> tuple[dict[str, list[list[Any]]], dict[str, int]]: ...

    async def count_statuses(
        self,
        task_names: list[str],
        statuses: list[Status],
    ) -> dict[str, dict[Status, int]]: ...

    async def save(self, checkpoint_path: Path) -> None: ...

    async def resume(self, checkpoint_path: Path) -> None: ...


class Sampler(Protocol):
    async def sample(
        self,
        *,
        task_name: str,
        group_status: Status | list[Status] | None = None,
    ) -> list[Any]: ...

    def save(self, checkpoint_path: Path) -> None: ...

    def resume(self, checkpoint_path: Path) -> None: ...


class AgentLoop(Protocol):
    async def generate_group(
        self,
        group: list[Any],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[Any]: ...

    async def pause(self) -> None: ...


class RolloutController(Protocol):
    async def continue_generation(self) -> None: ...

    async def pause_generation(self) -> None: ...


class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs: Any) -> bool: ...


class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[Any]) -> bool: ...


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs: Any) -> bool:
    return completed_count < batch_size


def default_is_valid_sample_fn(samples: list[Any]) -> bool:
    return True


@dataclass
class ProduceProgress:
    """共卡单次 produce_batch 的局部进度。

    中文不变量：
    - 只表达本次调用，不进入 checkpoint。
    - pending task 由具体 strategy 在本次调用内持有。
    - 裁剪非共卡需要的 producer_future_step / next_consumer_step / consumed_samples / target_upto_future_step / state_dict。
    - 不新增 model_step，model_step 仍由 manager 放进 ProduceContext。
    """

    target_samples: dict[str, int]
    raw_rewards_sum: dict[str, float] = field(default_factory=dict)
    raw_rewards_count: dict[str, int] = field(default_factory=dict)
    produced_samples: dict[str, int] = field(default_factory=dict)
    produced_tokens: dict[str, int] = field(default_factory=dict)
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
            raw_rewards_sum={name: 0.0 for name in task_names},
            raw_rewards_count={name: 0 for name in task_names},
            produced_samples={name: 0 for name in task_names},
            produced_tokens={name: 0 for name in task_names},
        )

    def add_raw_rewards(self, task_name: str, rewards_sum: float, rewards_count: int) -> None:
        self.raw_rewards_sum[task_name] += rewards_sum
        self.raw_rewards_count[task_name] += rewards_count

    def add_produced(self, task_name: str, samples: int, tokens: int) -> None:
        self.produced_samples[task_name] += samples
        self.produced_tokens[task_name] += tokens

    def add_produce_time(self, elapsed_s: float) -> None:
        self.produce_time_s += elapsed_s


@dataclass
class DisaggProduceProgress:
    """非共卡 Background Producer / Training Consumer 共享进度。

    中文不变量：
    - target_samples / consumed_samples 使用绝对累计口径。
    - consumer 从 replay buffer 取走样本后只增加 consumed，不回退 target。
    - producer_future_step 只由后台 producer 正常完成生产后推进。
    - 该对象会进入 checkpoint/resume。
    """

    task_names: list[str]
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
            task_names=task_names,
            consumed_samples={name: 0 for name in task_names},
            target_samples={name: 0 for name in task_names},
            raw_rewards_sum={name: 0.0 for name in task_names},
            raw_rewards_count={name: 0 for name in task_names},
            produced_samples={name: 0 for name in task_names},
            produced_tokens={name: 0 for name in task_names},
        )

    def ensure_target_upto(
        self,
        *,
        batch_size: int,
        future_step: int,
        allocate_batch_sizes: Callable[[int, int], dict[str, int]],
    ) -> dict[str, int]:
        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                task_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, task_size in task_sizes.items():
                    self.target_samples[task_name] += task_size
            self.target_upto_future_step = future_step
        return allocate_batch_sizes(batch_size, future_step)

    def begin_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step

    def mark_consumed(self, consumed_counts: dict[str, int]) -> None:
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
        self.producer_future_step = state["producer_future_step"]
        self.next_consumer_step = state["next_consumer_step"]
        self.target_upto_future_step = state["target_upto_future_step"]
        self.consumed_samples.clear()
        self.consumed_samples.update(state["consumed_samples"])
        self.target_samples.clear()
        self.target_samples.update(state["target_samples"])
        self.raw_rewards_sum.clear()
        self.raw_rewards_sum.update(state.get("raw_rewards_sum", {}))
        self.raw_rewards_count.clear()
        self.raw_rewards_count.update(state.get("raw_rewards_count", {}))
        self.produced_samples.clear()
        self.produced_samples.update(state.get("produced_samples", {}))
        self.produced_tokens.clear()
        self.produced_tokens.update(state.get("produced_tokens", {}))
        self.produce_time_s = state.get("produce_time_s", 0.0)


@dataclass
class BaseProduceContext:
    """strategy 生产一个 task 时看到的公共上下文。

    共卡和非共卡共享生成、采样、入库能力；具体 target / abort 语义由子类表达。
    """

    task_name: str
    agent_loop: AgentLoop
    sampler: Sampler
    replay_buffer: ReplayBuffer
    task_batch_size: int
    train_step: int
    model_step: int
    progress: ProduceProgress | DisaggProduceProgress
    is_valid_sample_fn: IsValidSampleFn
    stale_threshold: int | None

    @property
    def current_train_step_for_staleness(self) -> int:
        return self.train_step

    async def sample_group(self, *, from_expired_pool: bool) -> list[Any]:
        statuses = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.sampler.sample(task_name=self.task_name, group_status=statuses)

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(self.task_name, Status.EXPIRED)

    async def generate_group(
        self,
        group: list[Any],
        *,
        enable_partial_rollout: bool,
    ) -> list[Any]:
        start = time.perf_counter()
        result = await self.agent_loop.generate_group(
            group,
            enable_partial_rollout=enable_partial_rollout,
        )
        self.progress.add_produce_time(time.perf_counter() - start)
        return result

    async def put_generated_group(self, group: list[Any]) -> bool:
        """统一处理生成结果过滤、统计和入库。

        中文设计点：
        - 只有 completed group 才执行业务过滤。
        - ReplayBuffer.put 负责写 model_step、刷新 staleness、按阈值转 expired。
        - put 之后重新判断 group 状态，因为 completed 可能在入库前被转成 expired。
        """

        is_completed = get_group_status(group) == Status.COMPLETED
        produced_tokens = sum(len(getattr(item, "response_ids", []) or []) for item in group)
        if is_completed:
            # 真实实现按当前字段结构写 raw_rewards_sum/raw_rewards_count，不把这些字段重构成 metrics 对象。
            self.progress.add_raw_rewards(self.task_name, rewards_sum=0.0, rewards_count=0)
            if not self.is_valid_sample_fn(group):
                for item in group:
                    item.status = Status.FILTERED

        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.current_train_step_for_staleness,
            stale_threshold=self.stale_threshold,
        )
        self.progress.add_produced(self.task_name, samples=len(group), tokens=produced_tokens)
        return get_group_status(group) == Status.COMPLETED


@dataclass
class ProduceContext(BaseProduceContext):
    """共卡生产 context。

    中文设计点：
    - 去掉非共卡 update_event / absolute consumed / checkpoint progress 语义。
    - 保留当前 ProduceContext 内部字段结构：task_batch_size、progress、stale_threshold 等仍按原形状传递。
    """

    def should_abort(self) -> bool:
        return False


@dataclass
class DisaggProduceContext(BaseProduceContext):
    update_event: asyncio.Event
    progress: DisaggProduceProgress

    @property
    def current_train_step_for_staleness(self) -> int:
        return self.progress.next_consumer_step

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    async def available_count(self) -> int:
        completed = await self.replay_buffer.count(self.task_name, Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed

    @property
    def target_abs(self) -> int:
        return self.progress.target_samples[self.task_name]


class ProduceStrategy(ABC):
    @abstractmethod
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False


class DisaggProduceStrategy(ABC):
    @abstractmethod
    async def produce_batch(self, ctx: DisaggProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: DisaggProduceContext) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False

    def pending_task_count(self) -> int:
        return 0


ModeSpecificProduceStrategy = ProduceStrategy | DisaggProduceStrategy


class ProduceStrategyConfig(Protocol):
    def build(
        self,
        *,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> ProduceStrategy: ...


class DisaggProduceStrategyConfig(Protocol):
    def build(
        self,
        *,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> DisaggProduceStrategy: ...


@dataclass
class SyncProduceStrategyConfig:
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    def build(
        self,
        *,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> ProduceStrategy:
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


@dataclass
class AsyncProduceStrategyConfig:
    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = 0
    tail_batch_trigger_size: int = 0
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    def build(
        self,
        *,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> ProduceStrategy:
        return AsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


@dataclass
class DisaggAsyncProduceStrategyConfig:
    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = 0
    tail_batch_trigger_size: int = 0
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    def build(
        self,
        *,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> DisaggProduceStrategy:
        return DisaggAsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class SyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        *,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        pending: set[asyncio.Task] = set()
        completed = await ctx.replay_buffer.count(ctx.task_name, Status.COMPLETED)

        for _ in range(ctx.task_batch_size):
            group = await ctx.sampler.sample(task_name=ctx.task_name)
            pending.add(asyncio.create_task(ctx.generate_group(group, enable_partial_rollout=False)))

        while self.should_continue_fn(completed, ctx.task_batch_size):
            if not pending:
                break
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group = task.result()
                if await ctx.put_generated_group(group):
                    completed += 1

            while len(pending) + completed < ctx.task_batch_size and self.should_continue_fn(
                completed,
                ctx.task_batch_size,
            ):
                group = await ctx.sampler.sample(task_name=ctx.task_name)
                pending.add(asyncio.create_task(ctx.generate_group(group, enable_partial_rollout=False)))

        return ProduceBatchStatus.NORMAL


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        *,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        max_staleness: int,
        sync_weights_interval: int,
        tail_batch_trigger_size: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)
        self._pending_tasks: set[asyncio.Task] = set()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return calculate_seq_staleness(model_step, train_step) >= self.stale_threshold

    async def produce_batch(
        self,
        ctx: ProduceContext,
    ) -> ProduceBatchStatus:
        """共卡 async 生产。

        中文不变量：
        - pending 只属于本次 manager.produce_batch 调用，不跨 manager 调用保存。
        - 本函数只生产到 replay buffer，不在这里 pause/drain。
        - manager 必须等所有 task 的 produce_batch 都返回后，再调用 pause_produce 收尾 pending。
        - 不读取 update_event，不返回 UPDATE_WEIGHT_AND_ABORT。
        """

        self._pending_tasks = set()
        expired_count = await ctx.expired_count()
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size

        # 保持当前实现语义：normal 模式只按本 task batch size 追加固定超发预算；
        # tail-batch 模式只补必要缺口，并固定从 expired/aborted pool 取样。
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = ctx.task_batch_size + oversample_budget
        completed = await ctx.replay_buffer.count(ctx.task_name, Status.COMPLETED)

        async def schedule_one() -> None:
            group = await ctx.sample_group(from_expired_pool=sample_from_expired)
            self._pending_tasks.add(
                asyncio.create_task(
                    ctx.generate_group(
                        group,
                        enable_partial_rollout=self.enable_partial_rollout,
                    )
                )
            )

        while len(self._pending_tasks) + completed < scheduled_target:
            await schedule_one()

        while self.should_continue_fn(completed, ctx.task_batch_size):
            if not self._pending_tasks:
                break
            done, pending = await asyncio.wait(self._pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            self._pending_tasks = pending
            for task in done:
                if await ctx.put_generated_group(task.result()):
                    completed += 1

            while len(self._pending_tasks) + completed < scheduled_target and self.should_continue_fn(
                completed, ctx.task_batch_size
            ):
                await schedule_one()

        return ProduceBatchStatus.NORMAL

    async def pause_produce(self, ctx: ProduceContext) -> float:
        pending_tasks = self._pending_tasks
        self._pending_tasks = set()
        return await pause_pending_tasks(
            pending_tasks=pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: ctx.put_generated_group(task.result()),
        )


class _LocalPendingTasks:
    """把共卡本次调用的局部 set 包装成 pause helper 可使用的形状。"""

    def __init__(self, tasks: set[asyncio.Task]) -> None:
        self._tasks = tasks

    def count(self) -> int:
        return len(self._tasks)

    async def wait_and_claim(self, timeout_s: float) -> set[asyncio.Task]:
        if not self._tasks:
            return set()
        done, _ = await asyncio.wait(self._tasks, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        self._tasks.difference_update(done)
        return done

    async def cancel_all(self) -> int:
        tasks = set(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        return len(tasks)


class _PendingTasks:
    """非共卡专用 pending 集合。

    共卡不使用它，因为共卡 pending 不跨 produce_batch 调用。
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def count(self) -> int:
        return len(self._tasks)

    async def claim_ready(self) -> set[asyncio.Task]:
        async with self._lock:
            ready = {task for task in self._tasks if task.done()}
            self._tasks.difference_update(ready)
            return ready

    async def schedule_one(
        self,
        *,
        max_pending: int,
        should_abort: Callable[[], bool],
        spawn_one: Callable[[], Awaitable[asyncio.Task]],
    ) -> bool:
        async with self._lock:
            if should_abort() or len(self._tasks) >= max_pending:
                return False
            self._tasks.add(await spawn_one())
            return True

    async def wait_and_claim(self, timeout_s: float) -> set[asyncio.Task]:
        async with self._lock:
            snapshot = set(self._tasks)
        if not snapshot:
            return set()
        done, _ = await asyncio.wait(snapshot, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        async with self._lock:
            claimed = done & self._tasks
            self._tasks.difference_update(claimed)
            return claimed

    async def cancel_all(self) -> int:
        async with self._lock:
            tasks = set(self._tasks)
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        return len(tasks)


PendingTasksInput = set[asyncio.Task] | _PendingTasks


async def request_agent_loop_pause(ctx: BaseProduceContext, *, pending_count: int) -> None:
    """发送一次 agent loop pause 请求。

    最新生产代码里 pause_produce 会周期性调用 agent_loop.pause()，这里把这段协议抽成全局工具函数，
    让共卡本地 pending 收尾和非共卡后台 pending drain 使用同一套超时/日志语义。
    """

    try:
        await asyncio.wait_for(ctx.agent_loop.pause(), timeout=AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        # 真实实现写 logger.warning，伪代码只保留关键上下文。
        print(
            f"Agent loop pause timed out: task={ctx.task_name}, "
            f"timeout_s={AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S}, pending={pending_count}"
        )
    except Exception:
        print(f"Agent loop pause failed: task={ctx.task_name}, pending={pending_count}")


async def pause_pending_tasks(
    *,
    pending_tasks: PendingTasksInput,
    ctx: BaseProduceContext,
    put_claimed_task: Callable[[asyncio.Task], Awaitable[Any]],
) -> float:
    """复用当前 pause_produce 的 pending drain 协议。

    中文不变量：
    - 先发 pause，再等待 pending 产出。
    - pending 没清空时周期性补发 pause，兼容后端 abort 信号丢失或延迟。
    - 超时后 cancel 剩余 pending，避免 checkpoint/save 前仍有任务写 buffer。
    - 已完成任务必须 claim 后再 put，避免 produce 和 pause 重复入库同一个 done task。
    """

    pending = _LocalPendingTasks(pending_tasks) if isinstance(pending_tasks, set) else pending_tasks
    pause_start = time.perf_counter()
    if pending.count() == 0:
        return 0.0

    pending_pause_tasks = {
        asyncio.create_task(request_agent_loop_pause(ctx, pending_count=pending.count()))
    }
    cleanup_start_time = time.perf_counter()
    next_periodic_abort_time = cleanup_start_time + PERIODIC_ABORT_INTERVAL_S

    while True:
        elapsed_time = time.perf_counter() - cleanup_start_time
        if elapsed_time > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
            cancelled_count = await pending.cancel_all()
            print(
                f"Cleanup timeout reached. Forcefully cancelling {cancelled_count} "
                f"remaining tasks for task={ctx.task_name}."
            )
            break

        if pending.count() == 0:
            break

        current_time = time.perf_counter()
        pending_pause_tasks = {task for task in pending_pause_tasks if not task.done()}
        if PERIODIC_ABORT_INTERVAL_S > 0 and current_time >= next_periodic_abort_time:
            pending_pause_tasks.add(
                asyncio.create_task(request_agent_loop_pause(ctx, pending_count=pending.count()))
            )
            next_periodic_abort_time += PERIODIC_ABORT_INTERVAL_S

        claimed_done = await pending.wait_and_claim(timeout_s=1.0)
        for task in claimed_done:
            await put_claimed_task(task)

    for task in pending_pause_tasks:
        task.cancel()
    if pending_pause_tasks:
        await asyncio.gather(*pending_pause_tasks, return_exceptions=True)

    return time.perf_counter() - pause_start


class DisaggAsyncProduceStrategy(DisaggProduceStrategy):
    def __init__(
        self,
        *,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        max_staleness: int,
        sync_weights_interval: int,
        tail_batch_trigger_size: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)
        self._pending_tasks = _PendingTasks()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return calculate_seq_staleness(model_step, train_step) >= self.stale_threshold

    def pending_task_count(self) -> int:
        return self._pending_tasks.count()

    async def produce_batch(
        self,
        ctx: DisaggProduceContext,
    ) -> ProduceBatchStatus:
        """非共卡后台 async 生产。

        中文不变量：
        - pending 可以跨多次 produce_batch 调用存在。
        - 每轮循环都观察 update_event 和 model expired。
        - 只负责生产到 replay buffer，不取训练 batch。
        """

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.progress.producer_future_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        await self._put_claimed(await self._pending_tasks.claim_ready(), ctx)

        expired_count = await ctx.expired_count()
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size

        # 保持当前实现语义：normal 模式只按本 task batch size 追加固定超发预算；
        # tail-batch 模式只补必要缺口，并固定从 expired/aborted pool 取样。
        target_abs = ctx.target_abs
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = target_abs + oversample_budget

        async def spawn_one() -> asyncio.Task:
            group = await ctx.sample_group(from_expired_pool=sample_from_expired)
            return asyncio.create_task(
                ctx.generate_group(
                    group,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )

        while True:
            if ctx.should_abort():
                return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
            if self.is_model_expired(ctx.progress.producer_future_step, ctx.model_step):
                return ProduceBatchStatus.EXPIRED_BATCH

            available = await ctx.available_count()
            if not self.should_continue_fn(available, target_abs):
                return ProduceBatchStatus.NORMAL

            desired_pending = max(0, scheduled_target - available)
            while await self._pending_tasks.schedule_one(
                max_pending=desired_pending,
                should_abort=ctx.should_abort,
                spawn_one=spawn_one,
            ):
                pass

            claimed = await self._pending_tasks.wait_and_claim(timeout_s=1.0)
            await self._put_claimed(claimed, ctx)

    async def pause_produce(
        self,
        ctx: DisaggProduceContext,
    ) -> float:
        return await pause_pending_tasks(
            pending_tasks=self._pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: ctx.put_generated_group(task.result()),
        )

    async def _put_claimed(
        self,
        claimed: set[asyncio.Task],
        ctx: BaseProduceContext,
    ) -> None:
        for task in claimed:
            await ctx.put_generated_group(task.result())


@dataclass(frozen=True)
class TaskRunner:
    task_name: str
    agent_loop: AgentLoop
    sampler: Sampler
    produce_strategy: ModeSpecificProduceStrategy
    weight: float = 1.0
    order: int = 0

    @property
    def stale_threshold(self) -> int | None:
        return getattr(self.produce_strategy, "stale_threshold", None)


class AgentLoopManagerConfig:
    def __init__(self, tasks: list[Any], mode: ProducerMode = "colocate") -> None:
        self.tasks = tasks
        self.mode = mode

    def build(
        self,
        *,
        rollout_controller: RolloutController,
        tokenizer: Any,
        replay_buffer: ReplayBuffer,
        logger: Any,
        sync_weights_interval: int,
    ) -> "AgentLoopManager | DisaggAgentLoopManager":
        mode = self.mode
        runners = self._build_task_runners(
            mode=mode,
            rollout_controller=rollout_controller,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
            sync_weights_interval=sync_weights_interval,
        )
        if mode == "colocate":
            return AgentLoopManager(runners, replay_buffer, rollout_controller, logger)
        return DisaggAgentLoopManager(runners, replay_buffer, rollout_controller, logger)

    def _build_task_runners(
        self,
        *,
        mode: ProducerMode,
        rollout_controller: RolloutController,
        tokenizer: Any,
        replay_buffer: ReplayBuffer,
        logger: Any,
        sync_weights_interval: int,
    ) -> list[TaskRunner]:
        runners: list[TaskRunner] = []
        for task_cfg in self.tasks:
            # manager mode 只选择 manager 类型；strategy 的执行环境由 config 类型表达。
            if mode == "colocate" and not isinstance(
                task_cfg.produce_strategy_config,
                (SyncProduceStrategyConfig, AsyncProduceStrategyConfig),
            ):
                raise ValueError("colocate mode expects ProduceStrategyConfig")
            if mode == "disaggregated" and not isinstance(
                task_cfg.produce_strategy_config,
                DisaggAsyncProduceStrategyConfig,
            ):
                raise ValueError("disaggregated mode expects DisaggProduceStrategyConfig")
            strategy = task_cfg.produce_strategy_config.build(
                sync_weights_interval=sync_weights_interval,
                rollout_controller=rollout_controller,
            )
            runners.append(
                TaskRunner(
                    task_name=task_cfg.task_name,
                    agent_loop=task_cfg.agent_loop_config.build(rollout_controller, logger),
                    sampler=task_cfg.sampler_config.build(tokenizer, replay_buffer),
                    produce_strategy=strategy,
                    weight=task_cfg.weight,
                    order=len(runners),
                )
            )
        return runners


def allocate_task_batch_sizes(
    task_runners: list[TaskRunner],
    global_batch_size: int,
    train_step: int,
) -> dict[str, int]:
    # 真实实现沿用当前按 task weight 分配的逻辑；保持为全局 helper，避免两个 manager 继承公共父类。
    ...


def validate_task_batch_sizes(
    task_runners: list[TaskRunner],
    task_sizes: dict[str, int],
    global_batch_size: int,
) -> None:
    ...


async def refresh_for_all_tasks(
    *,
    task_runners: list[TaskRunner],
    replay_buffer: ReplayBuffer,
    train_step: int,
) -> None:
    thresholds = {
        task.task_name: task.stale_threshold or 1
        for task in task_runners
    }
    await replay_buffer.refresh_staleness(
        task_stale_thresholds=thresholds,
        current_train_step=train_step,
        statuses=[Status.COMPLETED, Status.ABORTED],
    )


async def take_train_batch(
    *,
    task_runners: list[TaskRunner],
    replay_buffer: ReplayBuffer,
    task_sizes: dict[str, int],
    progress: ProduceProgress | DisaggProduceProgress,
) -> ProduceBatchResult:
    batch_by_task, consumed_counts = await replay_buffer.take_batch(task_sizes)
    if isinstance(progress, DisaggProduceProgress):
        progress.mark_consumed(consumed_counts)

    counts = await replay_buffer.count_statuses(
        [task.task_name for task in task_runners],
        [Status.INIT, Status.COMPLETED, Status.ABORTED, Status.EXPIRED, Status.FAILED, Status.FILTERED],
    )
    return build_produce_batch_result(
        task_runners=task_runners,
        batch_by_task=batch_by_task,
        leftover_counts=counts,
        progress=progress,
    )


def build_produce_batch_result(
    *,
    task_runners: list[TaskRunner],
    batch_by_task: dict[str, list[list[Any]]],
    leftover_counts: dict[str, dict[Status, int]],
    progress: ProduceProgress | DisaggProduceProgress,
) -> ProduceBatchResult:
    # 真实实现负责 task result 聚合、timing 聚合、leftover 聚合。
    ...


class AgentLoopManager:
    def __init__(
        self,
        task_runners: list[TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger: Any,
    ) -> None:
        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self.rollout_controller = rollout_controller
        self.logger = logger
        self.task_names = [task.task_name for task in task_runners]

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        return allocate_task_batch_sizes(self.task_runners, global_batch_size, train_step)

    async def produce_batch(
        self,
        batch_size: int,
        train_step: int,
        *,
        model_step: int,
    ) -> ProduceBatchResult:
        """共卡训练唯一生产入口。

        中文不变量：
        - 不触碰非共卡 status/update_event。
        - 所有 active task 生产结束后，再统一收尾 pending。
        - 同一 manager 实例不并发调用 produce_batch；strategy pending 是本次调用的局部状态。
        - 返回必须是非空训练 batch。
        """

        task_sizes = (
            {self.task_runners[0].task_name: batch_size}
            if len(self.task_runners) == 1
            else self.get_task_batch_sizes(batch_size, train_step)
        )
        validate_task_batch_sizes(self.task_runners, task_sizes, batch_size)
        progress = ProduceProgress.build(
            task_names=self.task_names,
            target_samples=task_sizes,
        )
        active_contexts = [
            (
                task,
                ProduceContext(
                    task_name=task.task_name,
                    agent_loop=task.agent_loop,
                    sampler=task.sampler,
                    replay_buffer=self.replay_buffer,
                    task_batch_size=task_sizes[task.task_name],
                    train_step=train_step,
                    model_step=model_step,
                    progress=progress,
                    is_valid_sample_fn=getattr(task.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn),
                    stale_threshold=task.stale_threshold,
                ),
            )
            for task in self.task_runners
            if task_sizes[task.task_name] > 0
        ]

        await self.rollout_controller.continue_generation()
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            train_step=train_step,
        )
        await asyncio.gather(*[task.produce_strategy.produce_batch(ctx) for task, ctx in active_contexts])
        # 共卡 multi-task 的关键顺序：所有 task 正常完成生产后，才统一 pause/drain pending。
        # 如果上面的生产抛异常，异常直接冒泡中断训练，不在 manager 内做 best-effort cleanup。
        for task, ctx in active_contexts:
            await task.produce_strategy.pause_produce(ctx)
        result = await take_train_batch(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            task_sizes=task_sizes,
            progress=progress,
        )
        await self.rollout_controller.pause_generation()

        assert result.rollout_states, "共卡 produce_batch 必须返回非空训练 batch。"
        return result

    async def save(self, checkpoint_path: Path, model_step: int) -> None:
        # 共卡 checkpoint 不保存 DisaggProduceProgress。
        for task in self.task_runners:
            task.sampler.save(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.save(checkpoint_path)

    async def resume(self, checkpoint_path: Path) -> int:
        for task in self.task_runners:
            task.sampler.resume(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.resume(checkpoint_path)
        return 0


class DisaggAgentLoopManager:
    def __init__(
        self,
        task_runners: list[TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger: Any,
    ) -> None:
        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self.rollout_controller = rollout_controller
        self.logger = logger
        self.task_names = [task.task_name for task in task_runners]
        self.status = DisaggManagerStatus.NORMAL
        self.update_event = asyncio.Event()
        self.finish_event = asyncio.Event()
        self.model_step = 0
        self.pause_time_s = 0.0
        self.progress = DisaggProduceProgress.build(self.task_names)

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        return allocate_task_batch_sizes(self.task_runners, global_batch_size, train_step)

    async def produce_loop(self, batch_size: int) -> None:
        """非共卡 Background Producer。"""

        while not self.finish_event.is_set():
            if self.status == DisaggManagerStatus.FINISH:
                break
            if self.status in (
                DisaggManagerStatus.UPDATE_WEIGHT_AND_ABORT,
                DisaggManagerStatus.EXPIRED_BATCH,
            ):
                await self._wait_for_status_exit(self.status)
                continue

            task_sizes = self.progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=self.progress.producer_future_step,
                allocate_batch_sizes=self.get_task_batch_sizes,
            )
            validate_task_batch_sizes(self.task_runners, task_sizes, batch_size)
            statuses = await asyncio.gather(
                *[
                    task.produce_strategy.produce_batch(
                        DisaggProduceContext(
                            task_name=task.task_name,
                            agent_loop=task.agent_loop,
                            sampler=task.sampler,
                            replay_buffer=self.replay_buffer,
                            task_batch_size=task_sizes[task.task_name],
                            train_step=self.progress.producer_future_step,
                            model_step=self.model_step,
                            progress=self.progress,
                            is_valid_sample_fn=getattr(
                                task.produce_strategy,
                                "is_valid_sample_fn",
                                default_is_valid_sample_fn,
                            ),
                            stale_threshold=task.stale_threshold,
                            update_event=self.update_event,
                        )
                    )
                    for task in self.task_runners
                    if task_sizes[task.task_name] > 0
                ]
            )

            if ProduceBatchStatus.EXPIRED_BATCH in statuses:
                self.status = DisaggManagerStatus.EXPIRED_BATCH
            elif ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT in statuses:
                self.status = DisaggManagerStatus.UPDATE_WEIGHT_AND_ABORT
            else:
                self.progress.advance_future_step()

            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        """非共卡 Training Consumer。"""

        self.progress.begin_consume(train_step)
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            train_step=train_step,
        )
        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        validate_task_batch_sizes(self.task_runners, task_sizes, batch_size)
        current_model_step = train_step - 1

        while not self.finish_event.is_set():
            if self.status == DisaggManagerStatus.EXPIRED_BATCH:
                if current_model_step > self.model_step:
                    return ProduceBatchResult([], status=ProduceBatchStatus.EXPIRED_BATCH)
                if not await self.replay_buffer.is_ready(task_sizes):
                    raise RuntimeError("Expired Produce Batch 不能跳过，且当前训练 batch 未 ready。")

            if await self.replay_buffer.is_ready(task_sizes):
                result = await take_train_batch(
                    task_runners=self.task_runners,
                    replay_buffer=self.replay_buffer,
                    task_sizes=task_sizes,
                    progress=self.progress,
                )
                if self.status == DisaggManagerStatus.EXPIRED_BATCH:
                    result.status = ProduceBatchStatus.EXPIRED_BATCH
                if result.rollout_states:
                    self.progress.finish_consume(train_step)
                    await refresh_for_all_tasks(
                        task_runners=self.task_runners,
                        replay_buffer=self.replay_buffer,
                        train_step=train_step + 1,
                    )
                    return result

            await asyncio.sleep(1.0)

        return ProduceBatchResult([])

    async def pause_produce(self) -> float:
        """非共卡权重同步前的显式暂停入口。"""

        self.update_event.set()
        self.status = DisaggManagerStatus.UPDATE_WEIGHT_AND_ABORT
        await self.rollout_controller.pause_generation()

        pause_time_s = 0.0
        for task in self.task_runners:
            ctx = DisaggProduceContext(
                task_name=task.task_name,
                agent_loop=task.agent_loop,
                sampler=task.sampler,
                replay_buffer=self.replay_buffer,
                task_batch_size=0,
                train_step=self.progress.producer_future_step,
                model_step=self.model_step,
                progress=self.progress,
                is_valid_sample_fn=getattr(task.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn),
                stale_threshold=task.stale_threshold,
                update_event=self.update_event,
            )
            pause_time_s += await task.produce_strategy.pause_produce(ctx)
        self.pause_time_s = pause_time_s
        return pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        self.model_step = model_step
        await self.rollout_controller.continue_generation()
        self.status = DisaggManagerStatus.NORMAL
        self.update_event.clear()

    def shutdown(self) -> None:
        self.status = DisaggManagerStatus.FINISH
        self.update_event.set()
        self.finish_event.set()

    async def save(self, checkpoint_path: Path, model_step: int) -> None:
        pending = {
            task.task_name: task.produce_strategy.pending_task_count()
            for task in self.task_runners
            if task.produce_strategy.pending_task_count() > 0
        }
        if pending:
            raise RuntimeError(f"保存 checkpoint 前必须先 pause producer: {pending}")

        for task in self.task_runners:
            task.sampler.save(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.save(checkpoint_path)
        self._save_manager_state(checkpoint_path, model_step)

    async def resume(self, checkpoint_path: Path) -> int:
        for task in self.task_runners:
            task.sampler.resume(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.resume(checkpoint_path)
        saved_model_step = self._load_manager_state(checkpoint_path)

        self.update_event = asyncio.Event()
        self.finish_event = asyncio.Event()
        self.update_event.set()
        self.status = DisaggManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self.model_step = saved_model_step
        return saved_model_step

    async def _wait_for_status_exit(self, blocked_status: DisaggManagerStatus) -> None:
        while not self.finish_event.is_set() and self.status == blocked_status:
            await asyncio.sleep(1.0)

    def _save_manager_state(self, checkpoint_path: Path, model_step: int) -> None:
        ...

    def _load_manager_state(self, checkpoint_path: Path) -> int:
        ...


class RLTrainer:
    def __init__(self, cfg: Any) -> None:
        cfg.agent_loop_manager_cfg.mode = "colocate"
        self.agent_loop_manager = cfg.agent_loop_manager_cfg.build(
            rollout_controller=cfg.rollout_controller,
            tokenizer=cfg.tokenizer,
            replay_buffer=cfg.replay_buffer_config.build(),
            logger=cfg.logger,
            sync_weights_interval=cfg.sync_weights_interval,
        )
        if cfg.eval_agent_loop_manager_cfg is not None:
            cfg.eval_agent_loop_manager_cfg.mode = "colocate"
            self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build(...)

    def fit(self) -> None:
        for train_step in range(1, self.total_train_steps + 1):
            produce_result = asyncio.run(
                self.agent_loop_manager.produce_batch(
                    self.train_batch_size,
                    train_step=train_step,
                    model_step=self._current_rollout_model_step(train_step),
                )
            )
            self._train_one_batch(produce_result.rollout_states, train_step)


class RLDisaggTrainer:
    def __init__(self, cfg: Any) -> None:
        train_replay_buffer = cfg.replay_buffer_config.build()
        cfg.agent_loop_manager_cfg.mode = "disaggregated"
        self.agent_loop_manager = cfg.agent_loop_manager_cfg.build(
            rollout_controller=cfg.rollout_controller,
            tokenizer=cfg.tokenizer,
            replay_buffer=train_replay_buffer,
            logger=cfg.logger,
            sync_weights_interval=cfg.sync_weights_interval,
        )
        # eval 是一次性同步 produce_batch，不应构建成后台 manager。
        if cfg.eval_agent_loop_manager_cfg is not None:
            cfg.eval_agent_loop_manager_cfg.mode = "colocate"
            self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build(
                rollout_controller=cfg.rollout_controller,
                tokenizer=cfg.tokenizer,
                replay_buffer=train_replay_buffer,
                logger=cfg.logger,
                sync_weights_interval=cfg.sync_weights_interval,
            )

    async def _fit(self) -> None:
        producer_task = asyncio.create_task(
            self.agent_loop_manager.produce_loop(batch_size=self.train_batch_size)
        )
        train_step = self.cur_step + 1
        while train_step <= self.total_train_steps:
            get_batch_task = asyncio.create_task(
                self.agent_loop_manager.get_batch(
                    self.train_batch_size,
                    train_step=train_step,
                )
            )
            # 非共卡 fail-fast：consumer 等 batch 时必须同时观察后台 producer。
            # producer 异常不是业务 status，直接通过 result() 暴露原始异常栈。
            done, _ = await asyncio.wait(
                {producer_task, get_batch_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if producer_task in done:
                producer_task.result()
                raise RuntimeError("非共卡后台 producer 在训练结束前退出。")
            produce_result = get_batch_task.result()

            empty_expired = (
                produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
                and not produce_result.rollout_states
            )
            if not empty_expired:
                self._train_one_batch(produce_result.rollout_states, train_step)
                sync_model_step = train_step
            else:
                sync_model_step = train_step - 1

            if self._need_sync(sync_model_step, produce_result):
                await self.agent_loop_manager.pause_produce()
                await self._sync_weights_and_save(sync_model_step)
                await self.agent_loop_manager.continue_produce(model_step=sync_model_step)

            if empty_expired:
                continue
            self.cur_step = train_step
            train_step += 1

        self.agent_loop_manager.shutdown()
        await producer_task
