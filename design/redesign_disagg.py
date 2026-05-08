"""RL 共卡/非共卡生产流程重设计伪代码。

设计要点：
- 把进度窗口行为扩展到现有 ProduceProgress。
- 把通用 batch 能力下沉到 ReplayBuffer。
- 简单状态机字段仍保留在 AgentLoopManager。
- 保留 ProduceContext，收窄 AsyncProduceStrategy 对 manager / progress / replay buffer 的依赖面。
- 保留 AgentLoopManager.get_task_batch_sizes(global_batch_size, step) 作为 step 动态分配扩展点。

说明：
- 这是设计伪代码，只表达接口边界和核心流程，不是可直接运行实现。
- 中文注释重点写不变量和设计动机，避免隐藏知识散落到多个调用点。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Protocol


class Status(Enum):
    INIT = auto()
    COMPLETED = auto()
    ABORTED = auto()
    EXPIRED = auto()
    FAILED = auto()
    FILTERED = auto()


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


class ManagerStatus(Enum):
    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


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
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


@dataclass(frozen=True)
class _TaskRunner:
    """单个 task 的运行时依赖。

    真实实现沿用现有 _TaskRunner，避免为重设计引入一套平行命名。
    stale_threshold 可以先作为 property 从 produce_strategy 暴露，避免 manager 到处 getattr。
    """

    task_name: str
    agent_loop: Any
    sampler: Any
    produce_strategy: "ProduceStrategy"
    weight: float
    order: int

    @property
    def stale_threshold(self) -> int | None:
        return getattr(self.produce_strategy, "stale_threshold", None)


class AllocateBatchSizes(Protocol):
    def __call__(self, global_batch_size: int, step: int) -> dict[str, int]: ...


@dataclass
class ProduceProgress:
    """生产/消费共享进度。

    中文不变量：
    - target_samples / consumed_samples 都是绝对累计数。
    - consumer 从 buffer 取走样本后，只增加 consumed，不回退 target。
    - producer_future_step 只在后台 producer 正常完成一个 future step 后推进。
    - batch size 分配策略不放在这里，仍由 AgentLoopManager.get_task_batch_sizes 提供。
    """

    next_consumer_step: int = 1
    producer_future_step: int = 1
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    target_upto_future_step: int = 0

    @classmethod
    def build(cls, task_names: list[str]) -> "ProduceProgress":
        return cls(
            consumed_samples={name: 0 for name in task_names},
            target_samples={name: 0 for name in task_names},
        )

    @classmethod
    def build_local(
        cls,
        task_names: list[str],
        task_sizes: dict[str, int],
        train_step: int,
    ) -> "ProduceProgress":
        # 共卡 produce_batch 使用局部 progress，不污染非共卡全局累计进度。
        return cls(
            next_consumer_step=train_step,
            producer_future_step=train_step,
            consumed_samples={name: 0 for name in task_names},
            target_samples=dict(task_sizes),
            target_upto_future_step=train_step,
        )

    def ensure_target_upto(
        self,
        *,
        batch_size: int,
        future_step: int,
        allocate_batch_sizes: AllocateBatchSizes,
    ) -> dict[str, int]:
        """把 target_samples 累计到 future_step，并返回当前 step 的 task sizes。

        allocate_batch_sizes 由 AgentLoopManager 注入，保留原有 get_task_batch_sizes 扩展点。
        """

        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                task_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, size in task_sizes.items():
                    self.target_samples[task_name] += size
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_consumer_step": self.next_consumer_step,
            "producer_future_step": self.producer_future_step,
            "consumed_samples": self.consumed_samples,
            "target_samples": self.target_samples,
            "target_upto_future_step": self.target_upto_future_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # 原地更新，避免 strategy / context 持有旧引用。
        self.next_consumer_step = state["next_consumer_step"]
        self.producer_future_step = state["producer_future_step"]
        self.target_upto_future_step = state["target_upto_future_step"]
        self.consumed_samples.clear()
        self.consumed_samples.update(state["consumed_samples"])
        self.target_samples.clear()
        self.target_samples.update(state["target_samples"])


class ReplayBuffer(Protocol):
    """ReplayBuffer 需要新增的通用 batch 能力。

    这些方法只表达通用存储语义，不依赖 AgentLoopManager / ProduceBatchResult。
    """

    async def count(self, task_name: str, group_status: Status) -> int: ...

    async def get(self, batch_size: int, task_name: str, group_status: Status) -> list[list[Any]]: ...

    async def put(
        self,
        items: list[Any],
        task_name: str,
        *,
        model_step: int | None = None,
        current_train_step: int | None = None,
        stale_threshold: int | None = None,
    ) -> None:
        # 默认行为保持兼容：不传 model_step/current_train_step 时，
        # 只按 items 当前 status/staleness 入库，支持测试和 sampler 手工注入样本。
        #
        # 生成结果入库时显式传入 model_step/current_train_step/stale_threshold：
        # 1. update_sample_version(items, model_step)
        # 2. refresh_seq_staleness(items, current_train_step)
        # 3. stale_threshold 非空时执行 expire_group_if_needed(items, stale_threshold)
        #
        # is_valid_sample_fn 已在 caller 侧先执行；新的约束是它不依赖 version/staleness/expired 状态。
        ...

    async def refresh_staleness(
        self,
        *,
        task_stale_thresholds: dict[str, int],
        current_train_step: int,
        statuses: list[Status] | None = None,
    ) -> dict[str, int]:
        # 单 task 也用 {task_name: stale_threshold} 表达，避免再维护一套 many 接口。
        ...

    async def is_ready(
        self,
        task_batch_sizes: dict[str, int],
        *,
        group_status: Status = Status.COMPLETED,
    ) -> bool:
        # 判断每个 task 是否都有足够指定状态的 group。
        ...

    async def take_batch(
        self,
        task_batch_sizes: dict[str, int],
        *,
        group_status: Status = Status.COMPLETED,
    ) -> tuple[dict[str, list[list[Any]]], dict[str, int]]:
        # 按 task size 取 batch，并返回真实 consumed_counts。
        ...

    async def count_statuses(
        self,
        task_names: list[str],
        statuses: list[Status],
    ) -> dict[str, dict[Status, int]]:
        # 批量统计 leftover，用于 manager 组装返回字段和日志。
        ...


@dataclass
class ProduceContext:
    """ProduceStrategy 的 task-level 操作界面，重点收窄 AsyncProduceStrategy。

    它不是参数袋；它把容易传错的运行时契约封装成语义方法：
    - target / consumed 的绝对累计口径。
    - update_event 的暂停判断。
    - future_step 和 model_step 的过期判断输入。
    - replay buffer 的 available / expired 计数。
    - put 生成结果前的 producer 专属后处理入口。
    """

    task: _TaskRunner
    task_batch_size: int
    progress: ProduceProgress
    replay_buffer: ReplayBuffer
    update_event: asyncio.Event
    model_step: int

    @property
    def task_name(self) -> str:
        return self.task.task_name

    @property
    def future_step(self) -> int:
        return self.progress.producer_future_step

    @property
    def consumer_step(self) -> int:
        return self.progress.next_consumer_step

    @property
    def target_abs(self) -> int:
        return self.progress.target_samples[self.task_name]

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    def model_expired(self) -> bool:
        stale_threshold = self.task.stale_threshold
        if stale_threshold is None:
            return False
        # model_step 表示哪个 train_step 训练后的模型；完全同步时 current step 天然领先 1。
        return self.future_step - self.model_step - 1 >= stale_threshold

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(self.task_name, Status.EXPIRED)

    async def available_count(self) -> int:
        completed_count = await self.replay_buffer.count(self.task_name, Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed_count

    async def sample_group(self, *, from_expired_pool: bool) -> list[Any]:
        # 采样重试策略仍属于 sampler，不放入 ReplayBuffer。
        group_status = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.task.sampler.sample(task_name=self.task_name, group_status=group_status)

    async def put_generated_group(self, group: list[Any]) -> bool:
        # 新约束：is_valid_sample_fn 只判断生成结果本身是否可训练，
        # 不依赖 response_model_steps / seq_staleness / expired 状态。
        # 因此先过滤，再由 replay_buffer.put 统一做 version/staleness/expire。
        is_valid = self.task.produce_strategy.is_valid_sample_fn(group)
        if not is_valid:
            for item in group:
                item.status = Status.FILTERED
        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.consumer_step,
            stale_threshold=self.task.stale_threshold,
        )
        return is_valid


class ProduceStrategy(Protocol):
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext) -> float: ...


class _PendingTasks:
    """AsyncProduceStrategy 的并发 pending task 集合。

    这个 helper 只封装并发集合语义，不理解 sampler / rollout / replay buffer：
    - wait 使用快照，随后必须 claim，避免 pause 和 produce 重复处理同一个 task。
    - cancel 前先原子取出并清空集合，避免 cancel 后又被其他路径 claim。
    - schedule 时在锁内检查 abort 和 pending 数，避免 pause 已触发后继续新增 task。
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def count(self) -> int:
        async with self._lock:
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
            # spawn_one 内部可以采样并创建 rollout task；放在锁内是为了让 abort 检查和新增 task 原子化。
            self._tasks.add(await spawn_one())
            return True

    async def claim_all(self) -> set[asyncio.Task]:
        async with self._lock:
            claimed = set(self._tasks)
            self._tasks.clear()
            return claimed

    async def cancel_all(self) -> None:
        tasks = await self.claim_all()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class AsyncProduceStrategy:
    """异步生产策略只保留策略决策。

    strategy 不直接读 progress dict、不直接拼 replay_buffer 参数、不直接管理 manager event。
    pending task 的并发集合语义交给 _PendingTasks。
    """

    over_sample_threshold: float
    tail_batch_trigger_size: int
    _pending_tasks: _PendingTasks

    def __init__(self) -> None:
        # 真实实现继续保留现有配置参数；这里强调 pending helper 由 strategy 持有。
        self._pending_tasks = _PendingTasks()

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_ABORT
        if ctx.model_expired():
            return ProduceBatchStatus.EXPIRED_BATCH

        # 先回收跨调用遗留的 done pending；真实实现中所有结果都通过 ctx.put_generated_group 落库。
        await self._put_claimed(await self._pending_tasks.claim_ready(), ctx)

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_ABORT
        if ctx.model_expired():
            return ProduceBatchStatus.EXPIRED_BATCH

        expired_count = await ctx.expired_count()
        from_expired_pool = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size

        target_abs = ctx.target_abs
        oversample_budget = 0 if from_expired_pool else int(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = target_abs + oversample_budget

        while True:
            if ctx.should_abort():
                return ProduceBatchStatus.UPDATE_ABORT
            if ctx.model_expired():
                return ProduceBatchStatus.EXPIRED_BATCH

            available = await ctx.available_count()
            if available >= target_abs:
                return ProduceBatchStatus.NORMAL

            while available + await self._pending_tasks.count() < scheduled_target:
                scheduled = await self._pending_tasks.schedule_one(
                    max_pending=scheduled_target - available,
                    should_abort=ctx.should_abort,
                    spawn_one=lambda: self._spawn_one(ctx, from_expired_pool=from_expired_pool),
                )
                if not scheduled:
                    break

            claimed = await self._pending_tasks.wait_and_claim(timeout_s=1.0)
            if not claimed:
                return ProduceBatchStatus.NORMAL
            await self._put_claimed(claimed, ctx, available_base=available)

    async def pause_produce(self, ctx: ProduceContext) -> float:
        # 真实实现：
        # 1. pause rollout generation。
        # 2. 通过 _pending_tasks.wait_and_claim(...) claim done task。
        # 3. 通过 ctx.put_generated_group(...) 按当前 consumer_step 刷新 staleness 后落库。
        # 4. 超时后取消未完成 task。
        ...

    async def _spawn_one(self, ctx: ProduceContext, *, from_expired_pool: bool) -> asyncio.Task:
        # rollout_state = await ctx.sample_group(from_expired_pool=from_expired_pool)
        # return create_task(generate_group(ctx.task.agent_loop, rollout_state))
        ...

    async def _put_claimed(
        self,
        claimed: set[asyncio.Task],
        ctx: ProduceContext,
        *,
        available_base: int | None = None,
    ) -> int:
        # for task in claimed:
        #     is_valid = await ctx.put_generated_group(task.result())
        #     ...
        # return valid_completed_count
        ...


class AgentLoopManager:
    """训练侧 facade。

    manager 只负责编排：
    - ProduceProgress 负责 progress 不变量和累计 target。
    - ReplayBuffer 负责通用 batch 存储能力。
    - manager 自己保留简单 producer 状态机字段。
    - strategy 通过 ProduceContext 访问 task-level 操作。
    """
    _STATUS_POLL_INTERVAL_S = 1.0

    def __init__(self, task_runners: list[_TaskRunner], replay_buffer: ReplayBuffer):
        self.task_runners = task_runners
        self.task_names = [task.task_name for task in task_runners]
        self.replay_buffer = replay_buffer

        self._status = ManagerStatus.NORMAL
        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()
        self._model_step = 0
        self._pause_time_s = 0.0
        self._produce_progress = ProduceProgress.build(self.task_names)

    async def produce_batch(self, batch_size: int, train_step: int, *, model_step: int) -> ProduceBatchResult:
        """共卡同步入口：生产 -> 显式收尾 -> 取数。"""

        self.continue_produce(model_step)
        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        local_progress = ProduceProgress.build_local(self.task_names, task_sizes, train_step)

        await self._refresh_before_consume(train_step)
        status = await self._produce_to_buffer(task_sizes, local_progress)
        await self.pause_produce(use_global_progress=False, progress=local_progress)

        batch_by_task, _ = await self.replay_buffer.take_batch(task_sizes)
        result = await self._build_result(batch_by_task, status=status)
        result.task_batch_sizes = task_sizes
        assert result.rollout_states, "共卡 produce_batch 必须返回非空训练 batch。"
        return result

    async def produce_loop(self, batch_size: int) -> None:
        """非共卡后台入口：持续生产到 replay buffer。"""

        progress = self._produce_progress
        while not self._finish_event.is_set():
            if self._status == ManagerStatus.FINISH:
                break
            if self._status in (ManagerStatus.UPDATE_ABORT, ManagerStatus.EXPIRED_BATCH):
                await self._wait_until_resumed_or_finished()
                continue

            task_sizes = progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=progress.producer_future_step,
                allocate_batch_sizes=self.get_task_batch_sizes,
            )
            status = await self._produce_to_buffer(task_sizes, progress)

            if status == ProduceBatchStatus.EXPIRED_BATCH:
                self._status = ManagerStatus.EXPIRED_BATCH
            elif status == ProduceBatchStatus.NORMAL:
                progress.advance_future_step()

            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        """非共卡消费入口：等待 replay buffer 准备好后取训练 batch。"""

        progress = self._produce_progress
        progress.begin_consume(train_step)
        await self._refresh_before_consume(train_step)

        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        while not self._finish_event.is_set():
            if self._status == ManagerStatus.EXPIRED_BATCH:
                return ProduceBatchResult([], status=ProduceBatchStatus.EXPIRED_BATCH)

            if await self.replay_buffer.is_ready(task_sizes):
                batch_by_task, consumed_counts = await self.replay_buffer.take_batch(task_sizes)
                progress.mark_consumed(consumed_counts)

                progress.finish_consume(train_step)
                await self._refresh_before_consume(train_step + 1)

                result = await self._build_result(batch_by_task)
                result.task_batch_sizes = task_sizes
                return result

            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

        return ProduceBatchResult([], status=ProduceBatchStatus.FINISH)

    async def pause_produce(
        self,
        *,
        use_global_progress: bool,
        progress: ProduceProgress | None = None,
    ) -> float:
        if use_global_progress:
            if progress is not None:
                raise ValueError("use_global_progress=True 时不应传入局部 progress。")
            pause_progress = self._produce_progress
        else:
            if progress is None:
                raise ValueError("use_global_progress=False 时必须传入本轮局部 progress。")
            pause_progress = progress

        self._update_event.set()
        self._status = ManagerStatus.UPDATE_ABORT
        task_sizes = {task.task_name: 0 for task in self.task_runners}
        self._pause_time_s = await self._pause_with_progress(task_sizes, pause_progress)
        return self._pause_time_s

    def continue_produce(self, model_step: int) -> None:
        self._model_step = model_step
        self._status = ManagerStatus.NORMAL
        self._update_event.clear()

    def shutdown(self) -> None:
        self._status = ManagerStatus.FINISH
        self._update_event.set()
        self._finish_event.set()

    def get_task_batch_sizes(self, global_batch_size: int, step: int) -> dict[str, int]:
        # 维持原扩展点：默认按 weight 分配；自定义动态分配仍可覆盖这个方法。
        ...

    async def _refresh_before_consume(self, train_step: int) -> None:
        task_stale_thresholds = {
            task.task_name: task.stale_threshold
            for task in self.task_runners
            if task.stale_threshold is not None
        }
        if not task_stale_thresholds:
            return

        await self.replay_buffer.refresh_staleness(
            task_stale_thresholds=task_stale_thresholds,
            current_train_step=train_step,
            statuses=[Status.COMPLETED, Status.ABORTED],
        )

    async def _produce_to_buffer(self, task_sizes: dict[str, int], progress: ProduceProgress) -> ProduceBatchStatus:
        if any(self._build_context(task, task_sizes, progress).model_expired() for task in self.task_runners):
            return ProduceBatchStatus.EXPIRED_BATCH

        statuses = await asyncio.gather(
            *[
                task.produce_strategy.produce_batch(self._build_context(task, task_sizes, progress))
                for task in self.task_runners
                if progress.target_samples[task.task_name] > 0
            ]
        )
        return self._aggregate_status(statuses)

    async def _pause_with_progress(self, task_sizes: dict[str, int], progress: ProduceProgress) -> float:
        pause_times = await asyncio.gather(
            *[
                task.produce_strategy.pause_produce(self._build_context(task, task_sizes, progress))
                for task in self.task_runners
            ]
        )
        return sum(pause_times)

    def _build_context(
        self,
        task: _TaskRunner,
        task_sizes: dict[str, int],
        progress: ProduceProgress,
    ) -> ProduceContext:
        return ProduceContext(
            task=task,
            task_batch_size=task_sizes.get(task.task_name, 0),
            progress=progress,
            replay_buffer=self.replay_buffer,
            update_event=self._update_event,
            model_step=self._model_step,
        )

    async def _build_result(
        self,
        batch_by_task: dict[str, list[list[Any]]],
        *,
        status: ProduceBatchStatus = ProduceBatchStatus.NORMAL,
    ) -> ProduceBatchResult:
        # ProduceBatchResult 是 manager 的返回协议，不放入 ReplayBuffer。
        # 真实实现里从 replay_buffer.count_statuses(...) 补充 leftover，并从 sample extra_fields 汇总 timing。
        leftover = await self.replay_buffer.count_statuses(
            self.task_names,
            [Status.INIT, Status.COMPLETED, Status.ABORTED, Status.EXPIRED, Status.FAILED, Status.FILTERED],
        )
        rollout_states = [group for groups in batch_by_task.values() for group in groups]
        pause_time_s = self._pop_pause_time()

        def count_leftover(status: Status) -> int:
            return sum(task_counts.get(status, 0) for task_counts in leftover.values())

        return ProduceBatchResult(
            rollout_states=rollout_states,
            status=status,
            group_gen_pause_time_s=pause_time_s if pause_time_s > 0 else None,
            leftover_init=count_leftover(Status.INIT),
            leftover_completed=count_leftover(Status.COMPLETED),
            leftover_aborted=count_leftover(Status.ABORTED),
            leftover_expired=count_leftover(Status.EXPIRED),
            leftover_failed=count_leftover(Status.FAILED),
            leftover_filtered=count_leftover(Status.FILTERED),
        )

    async def _wait_until_resumed_or_finished(self) -> None:
        while self._status in (ManagerStatus.UPDATE_ABORT, ManagerStatus.EXPIRED_BATCH):
            if self._finish_event.is_set():
                return
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    def _pop_pause_time(self) -> float:
        pause_time_s = self._pause_time_s
        self._pause_time_s = 0.0
        return pause_time_s

    @staticmethod
    def _aggregate_status(statuses: list[ProduceBatchStatus]) -> ProduceBatchStatus:
        if any(status == ProduceBatchStatus.UPDATE_ABORT for status in statuses):
            return ProduceBatchStatus.UPDATE_ABORT
        if any(status == ProduceBatchStatus.EXPIRED_BATCH for status in statuses):
            return ProduceBatchStatus.EXPIRED_BATCH
        if any(status == ProduceBatchStatus.FINISH for status in statuses):
            return ProduceBatchStatus.FINISH
        return ProduceBatchStatus.NORMAL


async def colocate_train_loop(
    manager: AgentLoopManager,
    trainer: Any,
    *,
    train_batch_size: int,
    start_step: int,
    total_train_steps: int,
    sync_weights_interval: int,
) -> None:
    """共卡 trainer 的典型使用流程。"""

    model_step = start_step - 1
    for train_step in range(start_step, total_train_steps + 1):
        produce_result = await manager.produce_batch(
            train_batch_size,
            train_step=train_step,
            model_step=model_step,
        )
        if produce_result.status == ProduceBatchStatus.FINISH:
            break

        await trainer.train_one_batch(produce_result.rollout_states, train_step)

        if train_step % sync_weights_interval == 0:
            await trainer.sync_weights_and_save(train_step)
            model_step = train_step
            if trainer.should_eval(train_step):
                await trainer.run_eval(train_step)

        await trainer.log_step(train_step, produce_result)


async def disagg_train_loop(
    manager: AgentLoopManager,
    trainer: Any,
    *,
    train_batch_size: int,
    start_step: int,
    total_train_steps: int,
    sync_weights_interval: int,
) -> None:
    """非共卡 trainer 的典型使用流程。"""

    producer_task = asyncio.create_task(manager.produce_loop(train_batch_size))
    try:
        for train_step in range(start_step, total_train_steps + 1):
            produce_result = await manager.get_batch(train_batch_size, train_step=train_step)
            if produce_result.status == ProduceBatchStatus.FINISH:
                break

            expired = produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
            if not expired:
                assert produce_result.rollout_states, "非共卡 get_batch 除 EXPIRED_BATCH 外必须返回非空训练 batch。"
                await trainer.train_one_batch(produce_result.rollout_states, train_step)

            should_sync = expired or train_step % sync_weights_interval == 0 or train_step == total_train_steps
            if should_sync:
                await manager.pause_produce(use_global_progress=True)
                await trainer.sync_weights_and_save(train_step)

                if trainer.should_eval(train_step):
                    await trainer.run_eval(train_step)

                manager.continue_produce(model_step=train_step)

            await trainer.log_step(train_step, produce_result)
    finally:
        manager.shutdown()
        await producer_task
