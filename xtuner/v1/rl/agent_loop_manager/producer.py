import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import ray
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    refresh_seq_staleness,
    update_group_status,
    update_sample_version,
)
from xtuner.v1.rl.agent_loop import AgentLoopSpec, get_agent_loop_rollout_ctl
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.utils import pause_generation
from xtuner.v1.rl.utils import calculate_seq_staleness, create_task
from xtuner.v1.utils import get_logger

from .sampler import Sampler


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"


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
    """

    next_consumer_step: int = 1
    producer_future_step: int = 1
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    target_upto_future_step: int = 0


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()


async def _timed_generate_group(
    agent_loop: AgentLoopSpec,
    rollout_state: list[RolloutState],
    enable_partial_rollout: bool = False,
) -> list[RolloutState]:
    start = time.perf_counter()
    if isinstance(agent_loop, ray.actor.ActorHandle):
        result = await agent_loop.generate_group.remote(
            rollout_state,
            enable_partial_rollout=enable_partial_rollout,
        )
    else:
        result = await agent_loop.generate_group(
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


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return all(sample.status == Status.COMPLETED for sample in samples)


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs) -> bool:
    return completed_count < batch_size


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be non-negative, got {max_staleness}.")
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")

    # max_staleness 按同步周期计数；+1 表示训练天然必须接受的当前同步周期滞后。
    return (max_staleness + 1) * sync_weights_interval


def expire_group_if_needed(group: list[RolloutState], stale_threshold: int) -> list[RolloutState]:
    if stale_threshold <= 0:
        raise ValueError(f"stale_threshold must be positive, got {stale_threshold}.")

    group_status = update_group_status(group)
    if group_status not in (Status.COMPLETED, Status.ABORTED):
        return group
    if any(getattr(sample, "seq_staleness", 0) >= stale_threshold for sample in group):
        # completed / aborted 只要组内任一样本过期，就整组转为 EXPIRED。
        group_stalenss = [getattr(sample, "seq_staleness", 0) for sample in group]
        logger.info(
            f"Expiring group of {len(group)} samples due to sample staleness {group_stalenss} exceeding threshold {stale_threshold}. "
        )
        for sample in group:
            sample.status = Status.EXPIRED
    return group


def _validate_progress_for_task(
    progress: ProduceProgress,
    task_name: str,
    target_cumulative: int | None,
) -> None:
    if task_name not in progress.consumed_samples:
        raise KeyError(f"ProduceProgress.consumed_samples missing task_name={task_name!r}")
    if task_name not in progress.target_samples:
        raise KeyError(f"ProduceProgress.target_samples missing task_name={task_name!r}")

    if target_cumulative is not None and target_cumulative != progress.target_samples[task_name]:
        raise ValueError(
            "target_cumulative must match progress.target_samples when progress is provided, "
            f"got target_cumulative={target_cumulative}, "
            f"progress.target_samples[{task_name!r}]={progress.target_samples[task_name]}"
        )


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs) -> bool: ...


class ProduceStrategyConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(self, *, sync_weights_interval: int = 1) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    def build(self, *, sync_weights_interval: int = 1) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = Field(default=0, ge=0)
    tail_batch_trigger_size: int = 0

    def build(self, *, sync_weights_interval: int = 1) -> "AsyncProduceStrategy":
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
    async def produce_batch(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        train_step: int = 0,
        update_event: asyncio.Event | None = None,
        *,
        model_step: int,
        progress: ProduceProgress,
        target_cumulative: int | None = None,
    ) -> ProduceBatchStatus: ...

    async def pause_produce(
        self,
        agent_loop: AgentLoopSpec,
        replay_buffer: ReplayBuffer,
        task_name: str,
        *,
        progress: ProduceProgress,
    ) -> float:
        return 0.0


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        train_step: int = 0,
        update_event: asyncio.Event | None = None,
        *,
        model_step: int,
        progress: ProduceProgress,
        target_cumulative: int | None = None,
    ) -> ProduceBatchStatus:
        pending_tasks = set()
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        # TODO: 是否支持 SyncProduceStrategy 在非共卡时使用？如果支持，下面这行注释掉？
        # assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(batch_size):
            rollout_state = await sampler.sample(task_name=task_name)
            task = create_task(
                _timed_generate_group(
                    agent_loop,
                    rollout_state,
                )
            )
            pending_tasks.add(task)

        logger.info(f"[SyncProduceStrategy] Started {len(pending_tasks)} initial tasks.")

        while self.should_continue_fn(completed_sample_count, batch_size):
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

                for item in items:
                    update_sample_version(item, model_step)
                refresh_seq_staleness(items, train_step)

                is_valid = self.is_valid_sample_fn(items)
                if not is_valid:
                    for item in items:
                        item.status = Status.FILTERED

                await replay_buffer.put(items, task_name)
                if not is_valid:
                    continue

                completed_sample_count += 1
                if progress.target_samples[task_name] > 0:
                    logger.info(
                        f"[{self.__class__.__name__}] Collected "
                        f"{min(progress.target_samples[task_name], max(0, completed_sample_count))}/"
                        f"{progress.target_samples[task_name]} "
                        f"valid samples for task {task_name}."
                    )

            while len(pending_tasks) + completed_sample_count < batch_size and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = create_task(
                    _timed_generate_group(
                        agent_loop,
                        rollout_state,
                    )
                )
                pending_tasks.add(task)

        return ProduceBatchStatus.NORMAL


class AsyncProduceStrategy(ProduceStrategy):
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
        self._pending_tasks: set[asyncio.Task] = set()
        self._pending_task_model_steps: dict[asyncio.Task, int] = {}
        self._pending_lock = asyncio.Lock()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        staleness = calculate_seq_staleness(model_step, train_step)
        return staleness >= self.stale_threshold

    def _is_model_expired(self, train_step: int, model_step: int) -> bool:
        return self.is_model_expired(train_step, model_step)

    async def _snapshot_pending(self) -> set[asyncio.Task]:
        async with self._pending_lock:
            return set(self._pending_tasks)

    async def _pending_count(self) -> int:
        async with self._pending_lock:
            return len(self._pending_tasks)

    async def _claim_done(self, done: set[asyncio.Task]) -> set[asyncio.Task]:
        async with self._pending_lock:
            claimed = done & self._pending_tasks
            self._pending_tasks.difference_update(claimed)
            return claimed

    async def _claim_already_done(self) -> set[asyncio.Task]:
        async with self._pending_lock:
            done = {task for task in self._pending_tasks if task.done()}
            self._pending_tasks.difference_update(done)
            return done

    async def _put_generated_group(
        self,
        items: list[RolloutState],
        replay_buffer: ReplayBuffer,
        task_name: str,
        current_train_step: int,
        model_step: int,
    ) -> bool:
        for item in items:
            update_sample_version(item, model_step)
        refresh_seq_staleness(items, current_train_step)
        items = expire_group_if_needed(items, self.stale_threshold)

        is_valid = self.is_valid_sample_fn(items)
        if not is_valid:
            for item in items:
                item.status = Status.FILTERED

        await replay_buffer.put(items, task_name)
        return is_valid

    async def _put_claimed_tasks(
        self,
        claimed_tasks: set[asyncio.Task],
        replay_buffer: ReplayBuffer,
        task_name: str,
        progress: ProduceProgress,
        available_base: int | None = None,
    ) -> None:
        valid_completed_count = 0
        for task in claimed_tasks:
            # 每个 pending task 必须绑定调度时的模型版本；缺失说明调度状态已损坏，直接暴露。
            task_model_step = self._pending_task_model_steps.pop(task)
            is_valid = await self._put_generated_group(
                task.result(),
                replay_buffer,
                task_name,
                current_train_step=progress.next_consumer_step,
                model_step=task_model_step,
            )
            if is_valid:
                valid_completed_count += 1
            if is_valid and available_base is not None:
                if progress.target_samples[task_name] > 0:
                    logger.info(
                        f"[{self.__class__.__name__}] Collected "
                        f"{min(progress.target_samples[task_name], max(0, available_base + valid_completed_count))}/"
                        f"{progress.target_samples[task_name]} "
                        f"valid samples for task {task_name}."
                    )

    async def _schedule_one(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        desired_pending: int,
        sample_from_expired: bool,
        task_name: str,
        model_step: int,
        update_event: asyncio.Event | None,
    ) -> bool:
        async with self._pending_lock:
            # update_event 是 manager 级暂停信号；在调度临界区内检查，避免 pause 已触发后继续新增任务。
            if update_event is not None and update_event.is_set():
                return False
            if len(self._pending_tasks) >= desired_pending:
                return False
            group_status = [Status.EXPIRED, Status.ABORTED] if sample_from_expired else [Status.ABORTED]
            rollout_state = await sampler.sample(task_name=task_name, group_status=group_status)
            task = create_task(
                _timed_generate_group(
                    agent_loop,
                    rollout_state,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )
            self._pending_tasks.add(task)
            self._pending_task_model_steps[task] = model_step
            return True

    async def _schedule_tasks_until(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        task_name: str,
        desired_pending: int,
        sample_from_expired: bool,
        model_step: int,
        update_event: asyncio.Event | None,
    ) -> None:
        while await self._schedule_one(
            agent_loop=agent_loop,
            sampler=sampler,
            desired_pending=desired_pending,
            sample_from_expired=sample_from_expired,
            task_name=task_name,
            model_step=model_step,
            update_event=update_event,
        ):
            pass

    async def pause_produce(
        self,
        agent_loop: AgentLoopSpec,
        replay_buffer: ReplayBuffer,
        task_name: str,
        *,
        progress: ProduceProgress,
    ) -> float:
        pause_start = time.perf_counter()
        if await self._pending_count() == 0:
            return 0.0

        rollout_ctl = await get_agent_loop_rollout_ctl(agent_loop)
        await pause_generation(rollout_ctl)
        while True:
            pending_snapshot = await self._snapshot_pending()
            if not pending_snapshot:
                break

            done_tasks, _ = await asyncio.wait(
                pending_snapshot,
                timeout=1,
                return_when=asyncio.FIRST_COMPLETED,
            )
            claimed_done = await self._claim_done(done_tasks)
            for task in claimed_done:
                paused_items = task.result()
                # pause 可能发生在权重同步之后，但这里仍要使用 task 发起时绑定的模型版本。
                task_model_step = self._pending_task_model_steps.pop(task)
                for item in paused_items:
                    logger.debug(
                        f"[{self.__class__.__name__}] Task {task_name} | "
                        f"Collecting paused sample (uid: {item.uid}, status: {item.status}, "
                        f"length: {len(item.response_ids or [])}) after pausing generation."
                    )
                await self._put_generated_group(
                    paused_items,
                    replay_buffer,
                    task_name,
                    current_train_step=progress.next_consumer_step,
                    model_step=task_model_step,
                )
            if await self._pending_count() > 0:
                await pause_generation(rollout_ctl)
                await asyncio.sleep(1)
        return time.perf_counter() - pause_start

    async def produce_batch(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        train_step: int = 0,
        update_event: asyncio.Event | None = None,
        *,
        model_step: int,
        progress: ProduceProgress,
        target_cumulative: int | None = None,
    ) -> ProduceBatchStatus:
        if update_event is None:
            update_event = asyncio.Event()
        _validate_progress_for_task(progress, task_name, target_cumulative)

        if progress.target_samples[task_name] <= 0:
            return ProduceBatchStatus.NORMAL

        if update_event.is_set():
            return ProduceBatchStatus.UPDATE_ABORT
        if self.is_model_expired(train_step, model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        # 先回收跨 produce_batch 调用遗留的已完成任务，避免 done task 长期留在 pending 集合里。
        claimed_done = await self._claim_already_done()
        await self._put_claimed_tasks(
            claimed_done,
            replay_buffer,
            task_name,
            progress,
        )

        if update_event.is_set():
            return ProduceBatchStatus.UPDATE_ABORT
        if self.is_model_expired(train_step, model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        expired_count = await replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size
        if sample_from_expired:
            logger.info(
                f"Tail batch trigger condition met: {expired_count} expired samples "
                f"(threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )

        # 本轮 produce_batch 的必要累计目标固定；normal 模式只按当前 task batch 追加固定超发预算。
        # tail-batch 模式只补必要缺口，新增任务固定从 EXPIRED pool 取，不再扩大超发窗口。
        target_abs = progress.target_samples[task_name]
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * batch_size)
        scheduled_target = target_abs + oversample_budget
        logger.info(
            f"Starting produce_batch for task {task_name} with target_abs={target_abs}, "
            f"oversample_budget={oversample_budget}, scheduled_target={scheduled_target}."
        )
        while True:
            if update_event.is_set():
                return ProduceBatchStatus.UPDATE_ABORT
            if self.is_model_expired(train_step, model_step):
                return ProduceBatchStatus.EXPIRED_BATCH

            fresh = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
            available = progress.consumed_samples[task_name] + fresh
            # if available >= target_abs:
            if not self.should_continue_fn(available, target_abs):
                return ProduceBatchStatus.NORMAL

            pending_count = await self._pending_count()
            desired_pending = max(0, scheduled_target - available)
            if available + pending_count < scheduled_target:
                await self._schedule_tasks_until(
                    agent_loop=agent_loop,
                    sampler=sampler,
                    task_name=task_name,
                    desired_pending=desired_pending,
                    sample_from_expired=sample_from_expired,
                    model_step=model_step,
                    update_event=update_event,
                )
                if update_event.is_set():
                    return ProduceBatchStatus.UPDATE_ABORT

            pending_snapshot = await self._snapshot_pending()
            if update_event.is_set():
                return ProduceBatchStatus.UPDATE_ABORT
            if not pending_snapshot:
                logger.warning("All tasks are done but not enough samples collected.")
                return ProduceBatchStatus.NORMAL

            done_tasks, _ = await asyncio.wait(
                pending_snapshot,
                timeout=1,
                return_when=asyncio.FIRST_COMPLETED,
            )
            claimed_done = await self._claim_done(done_tasks)
            await self._put_claimed_tasks(
                claimed_done,
                replay_buffer,
                task_name,
                progress,
                available_base=available,
            )
