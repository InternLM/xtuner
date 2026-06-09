import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


if TYPE_CHECKING:
    from xtuner.v1.rl.rollout.controller import RolloutControllerProxy

import ray
import tqdm
from mmengine.dist import get_rank
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    get_group_status,
    reset_rollout_response,
)
from xtuner.v1.rl.agent_loop import AgentLoopSpec
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.trace import (
    TRACE_EXTRA_MODEL_STEP,
    TRACE_EXTRA_PRODUCE_BATCH_ID,
    TRACE_EXTRA_PRODUCER_FUTURE_STEP,
    TRACE_EXTRA_TRAIN_STEP,
    build_produce_batch_id,
    trace_function,
)
from xtuner.v1.rl.utils import (
    AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S,
    PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
    calculate_seq_staleness,
    cancel_and_drain,
    create_task,
)
from xtuner.v1.utils import get_logger

from .sampler import Sampler


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"


class _ProgressDisplayer:
    def __init__(self, progress_bar: Any | None) -> None:
        self._tqdm = progress_bar

    @classmethod
    def create(cls, *, strategy_name: str, task_name: str, total: int, initial: int) -> "_ProgressDisplayer":
        total = max(0, total)
        initial = min(total, max(0, initial))
        if total <= 0 or get_rank() != 0:
            return cls(None)
        return cls(
            tqdm.tqdm(
                total=total,
                initial=initial,
                desc=f"{strategy_name} {task_name}",
                unit="sample",
                dynamic_ncols=True,
                mininterval=30,
                leave=False,
            )
        )

    def update(self, value: int) -> None:
        if self._tqdm is None:
            return
        total = max(0, int(self._tqdm.total or 0))
        value = min(total, max(0, value))
        delta = value - self._tqdm.n
        if delta > 0:
            self._tqdm.update(delta)
            self._tqdm.n = value

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


@dataclass
class ProduceProgress:
    """生产者和消费者共享的 live 进度对象。

    设计目标：
    - Manager / 调用方负责初始化并原地更新这个对象，strategy 只接收引用并读取最新进度。
    - target / consumed 使用全局绝对累计口径，避免 consumer 取走 buffer 中的 completed 后，
      producer 把已消费样本误判成缺口并重复补发。
    - 同一套语义同时服务非共卡全局 progress 和共卡 produce_batch 的局部 progress。

    使用注意：
    - 不要在 strategy 中补 key 或用 dict.get(..., 0) 兜底；缺少 task key 应 fail fast。
    - 除非语义明确要求冻结本轮 produce_batch 的 target / scheduled_target，
      否则不要把字段值复制成局部快照后跨 await 使用；需要字段值时直接读 progress.xxx，
      让并发更新后的 next_consumer_step / consumed_samples 能尽早生效。
    - 运行中不要整体替换 ProduceProgress 对象；resume 时也应原地更新字段，避免旧引用失效。

    字段含义：
    - next_consumer_step：producer 写入新样本时应面向的训练 step。get_batch(i) 入口设为 i，
      成功取出非空 batch 后设为 i + 1。
    - producer_future_step：producer 当前准备生产的 future step。
    - consumed_samples：各 task 已被 consumer 从 replay buffer 取走的 group 绝对累计数。
    - target_samples：各 task 截至 target_upto_future_step 应生产出的 group 绝对累计目标。
    - target_upto_future_step：target_samples 已覆盖到的最大 future step。
    - raw_rewards_sum / raw_rewards_count：各 task 自上次 consumer 取 batch 后，producer 实际生成出的
      completed group reward 统计。filtered group 在过滤前仍按 completed 生成结果计入。
    - produced_samples / produced_tokens：各 task 自上次 consumer 取 batch 后，producer 实际返回的样本数和
      response token 数，包含 filtered / aborted / 未被训练消费的 completed 样本。
    - produce_time_s：自上次 consumer 取 batch 后，producer 实际执行 produce_batch 的累计 wall time。
    """

    next_consumer_step: int = 1
    producer_future_step: int = 1
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    target_upto_future_step: int = 0
    raw_rewards_sum: dict[str, float] = field(default_factory=dict)
    raw_rewards_count: dict[str, int] = field(default_factory=dict)
    produced_samples: dict[str, int] = field(default_factory=dict)
    produced_tokens: dict[str, int] = field(default_factory=dict)
    produce_time_s: float = 0.0

    @classmethod
    def build(cls, task_names: list[str]) -> "ProduceProgress":
        return cls(
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples={task_name: 0 for task_name in task_names},
            raw_rewards_sum={task_name: 0.0 for task_name in task_names},
            raw_rewards_count={task_name: 0 for task_name in task_names},
            produced_samples={task_name: 0 for task_name in task_names},
            produced_tokens={task_name: 0 for task_name in task_names},
        )

    @classmethod
    def build_local(
        cls,
        task_names: list[str],
        task_batch_sizes: dict[str, int],
        train_step: int,
    ) -> "ProduceProgress":
        # 共卡路径使用局部 progress，只表达本次 produce_batch 的目标，不污染非共卡累计窗口。
        return cls(
            next_consumer_step=train_step,
            producer_future_step=train_step,
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples=dict(task_batch_sizes),
            target_upto_future_step=train_step,
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

        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                task_batch_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, task_batch_size in task_batch_sizes.items():
                    self.target_samples[task_name] += task_batch_size
            self.target_upto_future_step = future_step

        return allocate_batch_sizes(batch_size, future_step)

    def begin_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step

    def mark_consumed(self, consumed_counts: dict[str, int]) -> None:
        # consumer 真实取出多少就累计多少，target 不回退，避免 producer 把已消费样本当成缺口。
        for task_name, count in consumed_counts.items():
            self.consumed_samples[task_name] += count

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

    def finish_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step + 1

    def advance_future_step(self) -> None:
        self.producer_future_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_consumer_step": self.next_consumer_step,
            "producer_future_step": self.producer_future_step,
            "consumed_samples": dict(self.consumed_samples),
            "target_samples": dict(self.target_samples),
            "target_upto_future_step": self.target_upto_future_step,
            "raw_rewards_sum": dict(self.raw_rewards_sum),
            "raw_rewards_count": dict(self.raw_rewards_count),
            "produced_samples": dict(self.produced_samples),
            "produced_tokens": dict(self.produced_tokens),
            "produce_time_s": self.produce_time_s,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # 原地更新 dict，避免 strategy / context 持有旧引用。
        self.next_consumer_step = state["next_consumer_step"]
        self.producer_future_step = state["producer_future_step"]
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


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return True


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs) -> bool:
    return completed_count < batch_size


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be non-negative, got {max_staleness}.")
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")

    # max_staleness 按同步周期计数；+1 表示训练天然必须接受的当前同步周期滞后。
    return (max_staleness + 1) * sync_weights_interval


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs) -> bool: ...


@dataclass
class ProduceContext:
    """单 task 生产上下文。

    这里集中维护 AsyncProduceStrategy 最容易传错的运行时契约：
    - strategy 只接受 ProduceContext，不再兼容散装参数入口；
    - target / consumed 都按绝对累计口径读取；
    - 暂停只读 manager 传入的 update_event；
    - rollout generate 的 ray/local 差异和 timing 字段写入；
    - 生成结果先按业务有效性过滤，再统一交给 ReplayBuffer 写版本、刷新 staleness、执行过期。
    """

    agent_loop: AgentLoopSpec
    sampler: Sampler
    replay_buffer: ReplayBuffer
    task_batch_size: int
    task_name: str
    train_step: int
    update_event: asyncio.Event
    model_step: int
    progress: ProduceProgress
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    stale_threshold: int | None = None

    @property
    def consumer_step(self) -> int:
        return self.progress.next_consumer_step

    @property
    def target_abs(self) -> int:
        return self.progress.target_samples[self.task_name]

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    def trace_kwargs(self) -> dict[str, Any]:
        produce_batch_id = build_produce_batch_id(
            self.train_step,
            self.model_step,
            self.progress.producer_future_step,
        )
        return {
            "task_name": self.task_name,
            "train_step": self.train_step,
            "model_step": self.model_step,
            "producer_future_step": self.progress.producer_future_step,
            "produce_batch_id": produce_batch_id,
        }

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(task_name=self.task_name, group_status=Status.EXPIRED)

    async def available_count(self) -> int:
        completed_count = await self.replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed_count

    @trace_function(
        "xtuner.producer.sample_group",
        trace_kwargs_getter=lambda self, *args, **kwargs: self.trace_kwargs(),
    )
    async def sample_group(self, *, from_expired_pool: bool) -> list[RolloutState]:
        group_status = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.sampler.sample(task_name=self.task_name, group_status=group_status)

    @trace_function("xtuner.producer.generate_group", trace_kwargs_getter=lambda self, *args, **kwargs: self.trace_kwargs())
    async def generate_group(
        self,
        rollout_state: list[RolloutState],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[RolloutState]:
        # strategy 只表达“要生成”，不关心 agent_loop 是 ray actor 还是本地对象。
        trace_kwargs = self.trace_kwargs()
        for state in rollout_state:
            extra_fields = dict(state.extra_fields or {})
            extra_fields[TRACE_EXTRA_TRAIN_STEP] = trace_kwargs["train_step"]
            extra_fields[TRACE_EXTRA_MODEL_STEP] = trace_kwargs["model_step"]
            extra_fields[TRACE_EXTRA_PRODUCER_FUTURE_STEP] = trace_kwargs["producer_future_step"]
            extra_fields[TRACE_EXTRA_PRODUCE_BATCH_ID] = trace_kwargs["produce_batch_id"]
            state.extra_fields = extra_fields
        start = time.perf_counter()
        if isinstance(self.agent_loop, ray.actor.ActorHandle):
            result = await self.agent_loop.generate_group.remote(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        else:
            result = await self.agent_loop.generate_group(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        elapsed = time.perf_counter() - start
        for item in result:
            extra_fields = getattr(item, "extra_fields", None)
            if extra_fields is None:
                extra_fields = {}
                setattr(item, "extra_fields", extra_fields)
            extra_fields[GROUP_GENERATE_TIME_KEY] = elapsed
        return result

    @trace_function(
        "xtuner.producer.put_generated_group",
        target="group",
        trace_kwargs_getter=lambda self, *args, **kwargs: self.trace_kwargs(),
    )
    async def put_generated_group(self, group: list[RolloutState]) -> bool:
        # 只有完整生成的 group 才需要业务有效性过滤；ABORTED / EXPIRED 保留原状态供重试或统计。
        is_completed = get_group_status(group) == Status.COMPLETED
        produced_tokens = sum(len(item.response_ids) for item in group if item.response_ids is not None)
        if is_completed:
            rewards_sum = 0.0
            rewards_count = 0
            for item in group:
                if item.reward is None or "score" not in item.reward:
                    logger.warning(
                        f"Missing reward score in item (uid: {item.uid}) of completed group for task {self.task_name}. This item will be skipped in reward statistics."
                    )
                    continue
                rewards_sum += float(item.reward["score"])  # type: ignore[index]
                rewards_count += 1
            self.progress.add_raw_rewards(self.task_name, rewards_sum, rewards_count)
            is_valid = self.is_valid_sample_fn(group)
            if not is_valid:
                for item in group:
                    item.status = Status.FILTERED
                    reset_rollout_response(item)
        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.consumer_step,
            stale_threshold=self.stale_threshold,
        )
        self.progress.add_produced(self.task_name, samples=len(group), tokens=produced_tokens)
        # replay_buffer.put 可能把 stale group 转为 EXPIRED，返回前重新判断是否仍可训练。
        is_completed = get_group_status(group) == Status.COMPLETED
        return is_completed


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
    """Configuration for asynchronous rollout production.

    The asynchronous strategy keeps producing rollout samples in the background
    and stores them in the replay buffer. It can oversample, allow partial
    rollout continuation, and discard samples that are too stale relative to the
    current training step.

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
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        # 默认同步策略没有跨权重版本的后台样本，只有异步策略需要判定模型过期。
        return False

    def pending_task_count(self) -> int:
        return 0


class _PendingTasks:
    """AsyncProduceStrategy 的并发 pending task 集合。

    这里只封装 pending set 的并发协议，不理解 sampler / rollout / replay buffer：
    - wait 使用快照，随后必须二次 claim，避免 produce 和 pause 重复处理同一个 done task。
    - cancel 前先原子 claim 并清空集合，避免 cancel 后又被其他路径 claim。
    - schedule one 在锁内同时检查 abort 和 pending 数，避免 pause 已触发后继续新增 task。
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def count(self) -> int:
        # 只暴露已经纳入 pending 集合的 task 数量。
        return len(self._tasks)

    async def claim_ready(self) -> set[asyncio.Task]:
        async with self._lock:
            ready = {task for task in self._tasks if task.done()}
            self._tasks.difference_update(ready)
            return ready

    async def wait_and_claim(self, *, timeout_s: float) -> set[asyncio.Task]:
        async with self._lock:
            snapshot = set(self._tasks)
        if not snapshot:
            return set()

        done, _ = await asyncio.wait(snapshot, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        async with self._lock:
            claimed = done & self._tasks
            self._tasks.difference_update(claimed)
            return claimed

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
            # 保持“检查 abort / pending 数 / 新增 task”这一组操作原子化。
            self._tasks.add(await spawn_one())
            return True

    async def _claim_all(self) -> set[asyncio.Task]:
        async with self._lock:
            claimed = set(self._tasks)
            self._tasks.clear()
            return claimed

    async def cancel_all(self) -> int:
        tasks = await self._claim_all()
        if not tasks:
            return 0
        logger.warning(f"Cancelling {len(tasks)} pending rollout tasks.")
        await cancel_and_drain(list(tasks))
        return len(tasks)


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        pending_tasks = set()
        completed_sample_count = await ctx.replay_buffer.count(task_name=ctx.task_name, group_status=Status.COMPLETED)
        # TODO: 是否支持 SyncProduceStrategy 在非共卡时使用？如果支持，下面这行注释掉？
        # assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(ctx.task_batch_size):
            rollout_state = await ctx.sampler.sample(task_name=ctx.task_name)
            task = create_task(ctx.generate_group(rollout_state))
            pending_tasks.add(task)

        logger.info(f"[SyncProduceStrategy] Started {len(pending_tasks)} initial tasks.")

        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.target_abs,
            initial=completed_sample_count,
        )
        try:
            while self.should_continue_fn(completed_sample_count, ctx.task_batch_size):
                if not pending_tasks:
                    logger.warning("[SyncProduceStrategy] All tasks are done but not enough samples collected.")
                    break
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                # 如果要过滤，在这个地方处理，然后加入到 replay buffer
                # 如果被过滤的数据就放到 put_to_filtered pool 中
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
        finally:
            progress_displayer.close()

        return ProduceBatchStatus.NORMAL


class AsyncProduceStrategy(ProduceStrategy):
    # Local retry interval for re-sending pause/abort while pending tasks drain.
    PERIODIC_ABORT_INTERVAL_S = 5.0

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
        self._pending_tasks = _PendingTasks()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        staleness = calculate_seq_staleness(model_step, train_step)
        return staleness >= self.stale_threshold

    def pending_task_count(self) -> int:
        return self._pending_tasks.count()

    async def _put_claimed(
        self,
        claimed_tasks: set[asyncio.Task],
        ctx: ProduceContext,
        available_base: int | None = None,
        progress_displayer: _ProgressDisplayer | None = None,
    ) -> None:
        completed_count = 0
        for task in claimed_tasks:
            items = task.result()
            is_completed = await ctx.put_generated_group(items)
            if is_completed:
                completed_count += 1
            if is_completed and available_base is not None and progress_displayer is not None:
                progress_displayer.update(available_base + completed_count)

    async def _pause_agent_loop(self, ctx: ProduceContext) -> None:
        pause_request_start = time.perf_counter()
        if isinstance(ctx.agent_loop, ray.actor.ActorHandle):
            pause_future = ctx.agent_loop.pause.remote()
        else:
            pause_future = ctx.agent_loop.pause()
        try:
            await asyncio.wait_for(pause_future, timeout=AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                f"Agent loop pause timed out: task={ctx.task_name}, timeout_s={AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S}, "
                f"elapsed={time.perf_counter() - pause_request_start:.2f}s, "
                f"pending={self._pending_tasks.count()}"
            )
        except Exception:
            logger.exception(
                f"Agent loop pause failed: task={ctx.task_name}, "
                f"elapsed={time.perf_counter() - pause_request_start:.2f}s, "
                f"pending={self._pending_tasks.count()}"
            )

    async def pause_produce(self, ctx: ProduceContext) -> float:
        pause_start = time.perf_counter()
        if self._pending_tasks.count() == 0:
            return 0.0

        pending_pause_tasks = {create_task(self._pause_agent_loop(ctx))}
        initial_pending_count = self._pending_tasks.count()

        logger.info(
            f"Pause signal loop started for task {ctx.task_name}. "
            f"Waiting for {initial_pending_count} pending tasks to complete. "
            f"periodic_abort_interval_s={self.PERIODIC_ABORT_INTERVAL_S}, "
            f"producer_pause_pending_task_timeout_s={PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S}"
        )
        cleanup_start_time = time.perf_counter()
        next_periodic_abort_time = cleanup_start_time + self.PERIODIC_ABORT_INTERVAL_S
        while True:
            elapsed_time = time.perf_counter() - cleanup_start_time
            if elapsed_time > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
                # 超时强制取消所有pending的任务
                cancelled_count = await self._pending_tasks.cancel_all()
                logger.warning(
                    f"Cleanup timeout of {PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S}s reached. "
                    f"Forcefully cancelling {cancelled_count} remaining tasks. "
                    f"task={ctx.task_name}"
                )
                break

            if self._pending_tasks.count() == 0:
                break
            current_time = time.perf_counter()
            pending_pause_tasks = {task for task in pending_pause_tasks if not task.done()}

            # 定时发送 pause 信号
            if self.PERIODIC_ABORT_INTERVAL_S > 0 and current_time >= next_periodic_abort_time:
                pending_pause_tasks.add(create_task(self._pause_agent_loop(ctx)))
                next_periodic_abort_time += self.PERIODIC_ABORT_INTERVAL_S

            claimed_done = await self._pending_tasks.wait_and_claim(timeout_s=1)
            for task in claimed_done:
                paused_items = task.result()
                await ctx.put_generated_group(paused_items)
        await cancel_and_drain(list(pending_pause_tasks))
        pause_time = time.perf_counter() - pause_start
        logger.info(f"pause_produce completed for task {ctx.task_name} within {pause_time}s.")
        return pause_time

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        if ctx.task_name not in ctx.progress.consumed_samples:
            raise KeyError(f"ProduceProgress.consumed_samples missing task_name={ctx.task_name!r}")
        if ctx.task_name not in ctx.progress.target_samples:
            raise KeyError(f"ProduceProgress.target_samples missing task_name={ctx.task_name!r}")

        if ctx.target_abs <= 0:
            return ProduceBatchStatus.NORMAL

        # TODO: place this check just before while loop
        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.train_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        # 先回收跨 produce_batch 调用遗留的已完成任务，避免 done task 长期留在 pending 集合里。
        claimed_done = await self._pending_tasks.claim_ready()
        await self._put_claimed(claimed_done, ctx)

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

        # 本轮 produce_batch 的必要累计目标固定；normal 模式只按当前 task batch 追加固定超发预算。
        # tail-batch 模式只补必要缺口，新增任务固定从 EXPIRED pool 取，不再扩大超发窗口。
        target_abs = ctx.target_abs
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = target_abs + oversample_budget
        logger.info(
            f"Starting produce_batch for task {ctx.task_name} with target_abs={target_abs}, "
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
            total=ctx.target_abs,
            initial=initial_available,
        )
        try:
            while True:
                if ctx.should_abort():
                    return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                if self.is_model_expired(ctx.train_step, ctx.model_step):
                    return ProduceBatchStatus.EXPIRED_BATCH

                available = await ctx.available_count()
                progress_displayer.update(available)
                if not self.should_continue_fn(available, target_abs):
                    return ProduceBatchStatus.NORMAL

                pending_count = self._pending_tasks.count()
                desired_pending = max(0, scheduled_target - available)
                if available + pending_count < scheduled_target:
                    while await self._pending_tasks.schedule_one(
                        max_pending=desired_pending,
                        should_abort=ctx.should_abort,
                        spawn_one=spawn_one,
                    ):
                        pass
                    # TODO: remove this check, because will check it when exit if statement, it's redundant
                    if ctx.should_abort():
                        return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT

                if ctx.should_abort():
                    return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                if self._pending_tasks.count() == 0:
                    logger.warning("All tasks are done but not enough samples collected.")
                    return ProduceBatchStatus.NORMAL

                claimed_done = await self._pending_tasks.wait_and_claim(timeout_s=1)
                await self._put_claimed(
                    claimed_done,
                    ctx,
                    available_base=available,
                    progress_displayer=progress_displayer,
                )
        finally:
            progress_displayer.close()
