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
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


class ManagerStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


def get_group_status(group: list[Any]) -> Status:
    # 只聚合状态，不在这里修改样本；状态翻转必须发生在显式过滤或过期逻辑中。
    ...


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
        # 3. stale_threshold 非空时执行 maybe_expire_group(items, stale_threshold)
        # 4. 用 get_group_status(items) 按最终状态入库
        #
        # is_valid_sample_fn 只在 caller 侧对 completed group 执行；非完整样本保留原状态。
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
    - replay buffer 的 available / expired 计数。
    - put 生成结果前的 producer 专属后处理入口。

    模型过期判断由 ProduceStrategy.is_model_expired(...) 提供，避免 context 和 strategy
    各维护一份相同逻辑。
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

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(self.task_name, Status.EXPIRED)

    async def available_count(self) -> int:
        completed_count = await self.replay_buffer.count(self.task_name, Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed_count

    async def sample_group(self, *, from_expired_pool: bool) -> list[Any]:
        # 采样重试策略仍属于 sampler，不放入 ReplayBuffer。
        group_status = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.task.sampler.sample(task_name=self.task_name, group_status=group_status)

    async def generate_group(
        self,
        rollout_state: list[Any],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[Any]:
        # 隐藏 agent_loop 本地/Ray 调用差异，并统一写入 generate timing 字段。
        ...

    async def put_generated_group(self, group: list[Any]) -> bool:
        # 只有完整生成的 group 才做业务有效性过滤；ABORTED / EXPIRED 保留原状态供重试或统计。
        is_completed = get_group_status(group) == Status.COMPLETED
        if is_completed:
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
        # put 内部可能因为 staleness 把 completed group 转为 EXPIRED，返回前重新聚合最终状态。
        return get_group_status(group) == Status.COMPLETED


class ProduceStrategy(Protocol):
    is_valid_sample_fn: Callable[[list[Any]], bool]

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext) -> float: ...

    def is_model_expired(self, train_step: int, model_step: int) -> bool: ...

    def pending_task_count(self) -> int: ...


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

    def count(self) -> int:
        # count 是轻量观测值；需要互斥的 claim / schedule / cancel 仍在各自方法里加锁。
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

    async def _claim_all(self) -> set[asyncio.Task]:
        async with self._lock:
            claimed = set(self._tasks)
            self._tasks.clear()
            return claimed

    async def cancel_all(self) -> int:
        tasks = await self._claim_all()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)


class AsyncProduceStrategy:
    """异步生产策略只保留策略决策。

    strategy 不直接读 progress dict、不直接拼 replay_buffer 参数、不直接管理 manager event。
    pending task 的并发集合语义交给 _PendingTasks。
    """

    over_sample_threshold: float
    tail_batch_trigger_size: int
    enable_partial_rollout: bool
    stale_threshold: int
    _pending_tasks: _PendingTasks

    def __init__(self) -> None:
        # 真实实现继续保留现有配置参数；这里强调 pending helper 由 strategy 持有。
        self._pending_tasks = _PendingTasks()
        self.enable_partial_rollout = False
        self.stale_threshold = 1

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        # model_step 表示哪个 train_step 训练后的模型；完全同步时 current step 天然领先 1。
        return train_step - model_step - 1 >= self.stale_threshold

    def pending_task_count(self) -> int:
        return self._pending_tasks.count()

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.future_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        # 先回收跨调用遗留的 done pending；真实实现中所有结果都通过 ctx.put_generated_group 落库。
        await self._put_claimed(await self._pending_tasks.claim_ready(), ctx)

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.future_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        expired_count = await ctx.expired_count()
        from_expired_pool = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size

        target_abs = ctx.target_abs
        oversample_budget = 0 if from_expired_pool else int(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = target_abs + oversample_budget

        async def spawn_one() -> asyncio.Task:
            rollout_state = await ctx.sample_group(from_expired_pool=from_expired_pool)
            return asyncio.create_task(
                ctx.generate_group(
                    rollout_state,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )

        while True:
            if ctx.should_abort():
                return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
            if self.is_model_expired(ctx.future_step, ctx.model_step):
                return ProduceBatchStatus.EXPIRED_BATCH

            available = await ctx.available_count()
            if available >= target_abs:
                return ProduceBatchStatus.NORMAL

            while available + self._pending_tasks.count() < scheduled_target:
                scheduled = await self._pending_tasks.schedule_one(
                    max_pending=scheduled_target - available,
                    should_abort=ctx.should_abort,
                    spawn_one=spawn_one,
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

    async def _put_claimed(
        self,
        claimed: set[asyncio.Task],
        ctx: ProduceContext,
        *,
        available_base: int | None = None,
    ) -> None:
        # for task in claimed:
        #     is_valid = await ctx.put_generated_group(task.result())
        #     根据 is_valid 更新日志计数；函数本身不再返回累计值。
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

        await self.continue_produce(model_step)
        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        local_progress = ProduceProgress.build_local(self.task_names, task_sizes, train_step)

        await self._refresh_for_all_tasks(train_step, [Status.COMPLETED, Status.ABORTED])
        status = await self._produce_batch_to_buffer(task_sizes, local_progress)
        await self.pause_produce(use_global_progress=False, progress=local_progress)

        result = await self._get_batch_from_buffer(
            batch_size=batch_size,
            task_batch_sizes=task_sizes,
            consume_progress=local_progress,
        )
        result.status = status
        result.task_batch_sizes = task_sizes
        assert result.rollout_states, "共卡 produce_batch 必须返回非空训练 batch。"
        return result

    async def produce_loop(self, batch_size: int) -> None:
        """非共卡后台入口：持续生产到 replay buffer。"""

        progress = self._produce_progress
        while not self._finish_event.is_set():
            if self._status == ManagerStatus.FINISH:
                break
            if self._status in (ManagerStatus.UPDATE_WEIGHT_AND_ABORT, ManagerStatus.EXPIRED_BATCH):
                await self._wait_until_resumed_or_finished()
                continue

            task_sizes = progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=progress.producer_future_step,
                allocate_batch_sizes=self.get_task_batch_sizes,
            )
            status = await self._produce_batch_to_buffer(task_sizes, progress)

            if status == ProduceBatchStatus.EXPIRED_BATCH:
                self._status = ManagerStatus.EXPIRED_BATCH
            elif status == ProduceBatchStatus.NORMAL:
                progress.advance_future_step()

            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        """非共卡消费入口：等待 replay buffer 准备好后取训练 batch。"""

        progress = self._produce_progress
        progress.begin_consume(train_step)
        await self._refresh_for_all_tasks(train_step, [Status.COMPLETED, Status.ABORTED])

        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        while not self._finish_event.is_set():
            current_model_step = train_step - 1
            if self._status == ManagerStatus.EXPIRED_BATCH and current_model_step > self._model_step:
                return ProduceBatchResult([], status=ProduceBatchStatus.EXPIRED_BATCH)

            ready = await self.replay_buffer.is_ready(task_sizes)
            if self._status == ManagerStatus.EXPIRED_BATCH and not ready:
                leftover_counts = await self.replay_buffer.count_statuses(self.task_names, _LEFTOVER_STATUSES)
                raise RuntimeError(
                    "EXPIRED_BATCH cannot be skipped and current batch is not ready. "
                    f"train_step={train_step}, current_model_step={current_model_step}, "
                    f"rollout_model_step={self._model_step}, status={self._status}, "
                    f"producer_future_step={progress.producer_future_step}, "
                    f"next_consumer_step={progress.next_consumer_step}, "
                    f"target_upto_future_step={progress.target_upto_future_step}, "
                    f"target_samples={progress.target_samples}, consumed_samples={progress.consumed_samples}, "
                    f"task_sizes={task_sizes}, leftover_counts={leftover_counts}"
                )

            if ready:
                result = await self._get_batch_from_buffer(
                    batch_size=batch_size,
                    task_batch_sizes=task_sizes,
                    consume_progress=progress,
                )
                if self._status == ManagerStatus.EXPIRED_BATCH:
                    result.status = ProduceBatchStatus.EXPIRED_BATCH
                progress.finish_consume(train_step)
                await self._refresh_for_all_tasks(train_step + 1, [Status.COMPLETED, Status.ABORTED])
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
        self._status = ManagerStatus.UPDATE_WEIGHT_AND_ABORT
        # 真实实现这里先调用 rollout_ctl.pause_generation.remote()，再 drain pending。
        task_sizes = {task.task_name: 0 for task in self.task_runners}
        self._pause_time_s = await self._pause_with_progress(task_sizes, pause_progress)
        return self._pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        # 真实实现先更新 model_step，再调用 rollout_ctl.continue_generation.remote()。
        # 状态切回 NORMAL 必须放在 continue_generation 完成之后。
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

    def _validate_task_batch_sizes(self, task_batch_sizes: dict[str, int], global_batch_size: int) -> None:
        # get_task_batch_sizes 是扩展点，入口集中校验 task 名和总 batch size。
        ...

    async def _refresh_for_all_tasks(self, train_step: int, statuses: list[Status]) -> None:
        task_stale_thresholds = {
            task.task_name: task.stale_threshold if task.stale_threshold is not None else 1
            for task in self.task_runners
        }

        await self.replay_buffer.refresh_staleness(
            task_stale_thresholds=task_stale_thresholds,
            current_train_step=train_step,
            statuses=statuses,
        )

    async def _produce_batch_to_buffer(
        self,
        task_batch_sizes: dict[str, int],
        progress: ProduceProgress,
    ) -> ProduceBatchStatus:
        current_future_step = progress.producer_future_step
        model_step = self._model_step
        expired_tasks = [
            task
            for task in self.task_runners
            if task.produce_strategy.is_model_expired(current_future_step, model_step)
        ]
        if expired_tasks:
            return ProduceBatchStatus.EXPIRED_BATCH

        statuses = await asyncio.gather(
            *[
                task.produce_strategy.produce_batch(self._build_context(task, task_batch_sizes, progress))
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

    async def _get_batch_from_buffer(
        self,
        *,
        batch_size: int,
        task_batch_sizes: dict[str, int],
        consume_progress: ProduceProgress,
    ) -> ProduceBatchResult:
        # 所有参数都必填，避免隐藏默认路径绕过 batch size 校验或 progress 更新。
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        batch_by_task, consumed_counts = await self.replay_buffer.take_batch(task_batch_sizes)
        consume_progress.mark_consumed(consumed_counts)
        result = await self._build_result(batch_by_task)
        result.task_batch_sizes = task_batch_sizes
        return result

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
        while self._status in (ManagerStatus.UPDATE_WEIGHT_AND_ABORT, ManagerStatus.EXPIRED_BATCH):
            if self._finish_event.is_set():
                return
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    def _pop_pause_time(self) -> float:
        pause_time_s = self._pause_time_s
        self._pause_time_s = 0.0
        return pause_time_s

    @staticmethod
    def _aggregate_status(statuses: list[ProduceBatchStatus]) -> ProduceBatchStatus:
        if any(status == ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT for status in statuses):
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
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
        train_step = start_step
        while train_step <= total_train_steps:
            produce_result = await manager.get_batch(train_batch_size, train_step=train_step)
            if produce_result.status == ProduceBatchStatus.FINISH:
                break

            expired = produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
            trained = False
            if produce_result.rollout_states:
                assert produce_result.rollout_states, "非共卡 get_batch 除 EXPIRED_BATCH 外必须返回非空训练 batch。"
                await trainer.train_one_batch(produce_result.rollout_states, train_step)
                trained = True

            sync_model_step = train_step if trained else train_step - 1
            should_sync = expired or sync_model_step % sync_weights_interval == 0 or sync_model_step == total_train_steps
            if should_sync:
                await manager.pause_produce(use_global_progress=True)
                await trainer.sync_weights_and_save(sync_model_step)

                if trainer.should_eval(sync_model_step):
                    await trainer.run_eval(sync_model_step)

                await manager.continue_produce(model_step=sync_model_step)

            if trained:
                await trainer.log_step(train_step, produce_result)
                train_step += 1
    finally:
        manager.shutdown()
        await producer_task
