import asyncio
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, runtime_checkable

import ray
import tqdm
from mmengine.dist import get_rank

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    discard_rollout_state,
    get_group_status,
)
from xtuner.v1.rl.agent_loop import AgentLoopSpec
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.utils import (
    AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S,
    PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
    cancel_and_drain,
    create_task,
)
from xtuner.v1.utils import get_logger

from .sampler import Sampler


if TYPE_CHECKING:
    from .disagg_producer import DisaggProduceProgress
    from .producer import ProduceProgress


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"
PERIODIC_ABORT_INTERVAL_S = 5.0


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


@dataclass(kw_only=True)
class BaseProduceContext:
    """共卡/非共卡共享的 sample、generate、put 能力。"""

    agent_loop: AgentLoopSpec
    sampler: Sampler
    replay_buffer: ReplayBuffer
    task_batch_size: int
    task_name: str
    train_step: int
    model_step: int
    progress: "ProduceProgress | DisaggProduceProgress"
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    stale_threshold: int | None = None

    @property
    def consumer_step(self) -> int:
        return self.train_step

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(task_name=self.task_name, group_status=Status.EXPIRED)

    async def sample_group(self, *, from_expired_pool: bool) -> list[RolloutState]:
        group_status = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.sampler.sample(task_name=self.task_name, group_status=group_status)

    async def generate_group(
        self,
        rollout_state: list[RolloutState],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[RolloutState]:
        # strategy 不关心 agent_loop 是 ray actor 还是本地对象。
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

    async def put_generated_group(self, group: list[RolloutState]) -> bool:
        produced_tokens = sum(len(item.response_ids) for item in group if item.response_ids is not None)
        initial_status = get_group_status(group)
        discard_status: Status | None = None

        if initial_status == Status.COMPLETED:
            rewards_sum = 0.0
            rewards_count = 0
            for item in group:
                if item.reward is None or "score" not in item.reward:
                    logger.warning(
                        f"Missing reward score in item (rollout_id: {item.rollout_id}) of completed group for task {self.task_name}. This item will be skipped in reward statistics."
                    )
                    continue
                rewards_sum += float(item.reward["score"])  # type: ignore[index]
                rewards_count += 1
            self.progress.add_raw_rewards(self.task_name, rewards_sum, rewards_count)

            if not self.is_valid_sample_fn(group):
                discard_status = Status.FILTERED
        elif initial_status == Status.FAILED:
            discard_status = Status.FAILED

        if discard_status is not None:
            # 失败样本和业务过滤样本都不进入 replay buffer。
            self.progress.add_produced(self.task_name, samples=len(group), tokens=produced_tokens)
            self.progress.add_discarded(self.task_name, discard_status, samples=len(group))
            for item in group:
                discard_rollout_state(item)
            return False

        # ABORTED / EXPIRED 保持可重试状态；COMPLETED 也可能在 put 时因 staleness 转成 EXPIRED。
        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.consumer_step,
            stale_threshold=self.stale_threshold,
        )
        self.progress.add_produced(self.task_name, samples=len(group), tokens=produced_tokens)
        final_status = get_group_status(group)
        return final_status == Status.COMPLETED


@dataclass
class ProduceBatchResult:
    """Result of a single ``produce_batch`` call.

    Args:
        rollout_states (list[list[RolloutState]]): Completed rollout groups retrieved from the replay buffer for training.
        group_gen_count (int | None): Number of generate-group calls finished in this batch (None if no generations ran).
        group_gen_mean_s (float | None): Mean wall-clock time per generate-group call, in seconds.
        group_gen_p50_s (float | None): Median (p50) generate-group time, in seconds.
        group_gen_p99_s (float | None): 99th percentile generate-group time, in seconds.
        group_gen_p99_p50_ratio (float | None): Ratio of p99 to p50, indicating tail-latency skew.
        group_gen_pause_time_s (float | None): Time spent in pause/cleanup phase (async strategy only), in seconds.
        leftover_init (int): Number of init groups remaining in the replay buffer after this batch.
        leftover_completed (int): Number of completed groups remaining in the replay buffer after this batch.
        leftover_aborted (int): Number of aborted groups remaining in the replay buffer.
        leftover_expired (int): Number of expired groups remaining in the replay buffer.
        failed_samples (int): Number of failed samples observed in the current produce window, including replay-buffer leftovers and samples discarded before insertion.
        filtered_samples (int): Number of filtered samples observed in the current produce window, including replay-buffer leftovers and samples discarded before insertion.
        raw_rewards_sum (float): Sum of rewards produced before replay-buffer insertion for the current window.
        raw_rewards_count (int): Number of reward-bearing samples included in ``raw_rewards_sum``.
        produced_samples (int): Number of rollout samples produced in the current produce window.
        produced_tokens (int): Number of response tokens produced in the current produce window.
        produce_time_s (float): Wall-clock production time consumed by the current produce window.
    """

    rollout_states: list[list[RolloutState]]
    status: ProduceBatchStatus = ProduceBatchStatus.NORMAL
    # per-group generation timing stats (all None if no generations occurred)
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    # leftover buffer counts after batch retrieval
    leftover_init: int = 0
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0
    # failed / filtered are produce-window sample counts, not pure buffer leftover snapshots
    failed_samples: int = 0
    filtered_samples: int = 0
    # rewards produced during the current produce window, including completed and filtered groups.
    raw_rewards_sum: float = 0.0
    raw_rewards_count: int = 0
    produced_samples: int = 0
    produced_tokens: int = 0
    produce_time_s: float = 0.0
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


@dataclass(frozen=True)
class _TaskRunner:
    task_name: str
    agent_loop: AgentLoopSpec
    produce_strategy: Any
    sampler: Sampler
    weight: float = 1.0
    order: int = 0

    @property
    def is_valid_sample_fn(self) -> IsValidSampleFn:
        return getattr(self.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn)

    @property
    def stale_threshold(self) -> int | None:
        return getattr(self.produce_strategy, "stale_threshold", None)


class _TaskSamplerView:
    def __init__(self, samplers: list[Sampler]):
        self._samplers = samplers

    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self._samplers)


def _fill_produce_timing_stats(
    result: ProduceBatchResult, generate_times_s: list[float], pause_time_s: float = 0.0
) -> None:
    if not generate_times_s:
        if pause_time_s > 0:
            result.group_gen_pause_time_s = pause_time_s
        return
    sorted_times = sorted(generate_times_s)
    n = len(sorted_times)
    mean_s = sum(sorted_times) / n
    p50_s = sorted_times[n // 2]
    p99_s = sorted_times[int(n * 0.99)]
    ratio = p99_s / p50_s if p50_s > 0 else float("inf")
    result.group_gen_count = n
    result.group_gen_mean_s = mean_s
    result.group_gen_p50_s = p50_s
    result.group_gen_p99_s = p99_s
    result.group_gen_p99_p50_ratio = ratio
    result.group_gen_pause_time_s = pause_time_s


def _fill_group_timing_stats(
    result: ProduceBatchResult, rollout_states: list[list[RolloutState]], pause_time_s: float = 0.0
) -> None:
    generate_times: list[float] = []
    for group in rollout_states:
        if not group:
            continue
        group_time = getattr(group[0], "extra_fields", {}).get(GROUP_GENERATE_TIME_KEY)
        if group_time is not None:
            generate_times.append(group_time)

    _fill_produce_timing_stats(result, generate_times, pause_time_s=pause_time_s)


_LEFTOVER_STATUSES = [
    Status.INIT,
    Status.COMPLETED,
    Status.ABORTED,
    Status.EXPIRED,
    Status.FAILED,
    Status.FILTERED,
]
_TASK_CHECKPOINT_DIR = "tasks"
_MANAGER_STATE_PATH = "agent_loop_manager_state.json"
_STATUS_POLL_INTERVAL_S = 1.0


def _fill_leftover_counts(result: ProduceBatchResult, status_counts: dict[Status, int]) -> None:
    result.leftover_init = status_counts.get(Status.INIT, 0)
    result.leftover_completed = status_counts.get(Status.COMPLETED, 0)
    result.leftover_aborted = status_counts.get(Status.ABORTED, 0)
    result.leftover_expired = status_counts.get(Status.EXPIRED, 0)
    result.failed_samples = status_counts.get(Status.FAILED, 0)
    result.filtered_samples = status_counts.get(Status.FILTERED, 0)


def _merge_discarded_counts(
    result: ProduceBatchResult,
    progress: "ProduceProgress | DisaggProduceProgress",
    task_name: str,
) -> None:
    discarded_failed, discarded_filtered = progress.consume_discarded(task_name)
    result.failed_samples += discarded_failed
    result.filtered_samples += discarded_filtered


def allocate_task_batch_sizes(
    task_runners: list[_TaskRunner],
    global_batch_size: int,
    train_step: int,
) -> dict[str, int]:
    # train_step 只为后台 progress 回调保留同一形状；当前按静态 weight 分配。
    if global_batch_size < 0:
        raise ValueError(f"global_batch_size must be non-negative, got {global_batch_size}")

    total_weight = sum(task.weight for task in task_runners)
    if total_weight <= 0:
        raise ValueError("Sum of task weights must be positive.")
    if global_batch_size == 0:
        task_batch_sizes = {task.task_name: 0 for task in task_runners}
    else:
        raw_allocations = [global_batch_size * task.weight / total_weight for task in task_runners]
        floor_allocations = [math.floor(raw) for raw in raw_allocations]
        remaining = global_batch_size - sum(floor_allocations)

        task_batch_sizes = {task.task_name: floor_allocations[idx] for idx, task in enumerate(task_runners)}
        ranked_tasks = sorted(
            enumerate(task_runners),
            key=lambda item: (
                -(raw_allocations[item[0]] - floor_allocations[item[0]]),
                item[1].order,
            ),
        )
        for idx, task in ranked_tasks[:remaining]:
            task_batch_sizes[task.task_name] += 1

    expected_task_names = {task.task_name for task in task_runners}
    actual_task_names = set(task_batch_sizes.keys())
    if actual_task_names != expected_task_names:
        missing_task_names = expected_task_names - actual_task_names
        extra_task_names = actual_task_names - expected_task_names
        raise ValueError(
            "Invalid task batch sizes allocated: "
            f"missing={sorted(missing_task_names)}, extra={sorted(extra_task_names)}"
        )

    negative_batch_sizes = {
        task_name: task_batch_size for task_name, task_batch_size in task_batch_sizes.items() if task_batch_size < 0
    }
    if negative_batch_sizes:
        raise ValueError(f"Task batch sizes must be non-negative, got {negative_batch_sizes}")

    total_batch_size = sum(task_batch_sizes.values())
    if total_batch_size != global_batch_size:
        raise ValueError(
            "Task batch sizes must sum to the requested global batch size, "
            f"got total={total_batch_size}, expected={global_batch_size}"
        )
    return task_batch_sizes


async def refresh_for_all_tasks(
    *,
    task_runners: list[_TaskRunner],
    replay_buffer: ReplayBuffer,
    logger,
    manager_name: str,
    train_step: int,
    statuses: list[Status],
) -> None:
    task_stale_thresholds: dict[str, int] = {}
    for task in task_runners:
        # 没有 stale_threshold 的同步策略按 1 处理。
        task_stale_thresholds[task.task_name] = task.stale_threshold if task.stale_threshold is not None else 1

    expired_counts = await replay_buffer.refresh_staleness(
        task_stale_thresholds=task_stale_thresholds,
        current_train_step=train_step,
        statuses=statuses,
    )
    for task_name, expired_count in expired_counts.items():
        logger.info(
            f"[AgentLoopManager][{manager_name}] Refresh staleness for task {task_name}: expired_count={expired_count}"
        )


def aggregate_task_results(
    ordered_tasks: list[_TaskRunner], task_results: dict[str, ProduceBatchResult]
) -> ProduceBatchResult:
    rollout_states: list[list[RolloutState]] = []
    leftover_init = 0
    leftover_completed = 0
    leftover_aborted = 0
    leftover_expired = 0
    failed_samples = 0
    filtered_samples = 0
    total_group_count = 0
    weighted_group_mean_sum = 0.0
    weighted_group_p50_sum = 0.0
    weighted_group_p99_sum = 0.0
    weighted_group_ratio_sum = 0.0
    total_pause_time_s = 0.0
    raw_rewards_sum = 0.0
    raw_rewards_count = 0
    produced_samples = 0
    produced_tokens = 0
    produce_time_s = 0.0

    for task in ordered_tasks:
        result = task_results[task.task_name]
        rollout_states.extend(result.rollout_states)
        leftover_init += result.leftover_init
        leftover_completed += result.leftover_completed
        leftover_aborted += result.leftover_aborted
        leftover_expired += result.leftover_expired
        failed_samples += result.failed_samples
        filtered_samples += result.filtered_samples
        raw_rewards_sum += result.raw_rewards_sum
        raw_rewards_count += result.raw_rewards_count
        produced_samples += result.produced_samples
        produced_tokens += result.produced_tokens
        produce_time_s += result.produce_time_s
        if result.group_gen_count is not None and result.group_gen_mean_s is not None:
            total_group_count += result.group_gen_count
            weighted_group_mean_sum += result.group_gen_count * result.group_gen_mean_s
            weighted_group_p50_sum += result.group_gen_count * (result.group_gen_p50_s or 0.0)
            weighted_group_p99_sum += result.group_gen_count * (result.group_gen_p99_s or 0.0)
            weighted_group_ratio_sum += result.group_gen_count * (result.group_gen_p99_p50_ratio or 0.0)
            total_pause_time_s += result.group_gen_pause_time_s or 0.0

    aggregated = ProduceBatchResult(
        rollout_states=rollout_states,
        leftover_init=leftover_init,
        leftover_completed=leftover_completed,
        leftover_aborted=leftover_aborted,
        leftover_expired=leftover_expired,
        failed_samples=failed_samples,
        filtered_samples=filtered_samples,
        raw_rewards_sum=raw_rewards_sum,
        raw_rewards_count=raw_rewards_count,
        produced_samples=produced_samples,
        produced_tokens=produced_tokens,
        produce_time_s=produce_time_s,
        task_results={task.task_name: task_results[task.task_name] for task in ordered_tasks},
    )
    if total_group_count > 0:
        aggregated.group_gen_count = total_group_count
        aggregated.group_gen_mean_s = weighted_group_mean_sum / total_group_count
        aggregated.group_gen_p50_s = weighted_group_p50_sum / total_group_count
        aggregated.group_gen_p99_s = weighted_group_p99_sum / total_group_count
        aggregated.group_gen_p99_p50_ratio = weighted_group_ratio_sum / total_group_count
        aggregated.group_gen_pause_time_s = total_pause_time_s
    return aggregated


def log_buffer_counts(
    logger,
    *,
    manager_name: str,
    task_runners: list[_TaskRunner],
    task_batch_sizes: dict[str, int],
    batch_by_task: dict[str, list[list[RolloutState]]],
    leftover_counts: dict[str, dict[Status, int]],
) -> None:
    for task in task_runners:
        task_name = task.task_name
        task_counts = leftover_counts.get(task_name, {})
        logger.info(
            f"[AgentLoopManager][{manager_name}] get_batch from buffer for task {task_name}: "
            f"requested={task_batch_sizes[task_name]}, retrieved={len(batch_by_task.get(task_name, []))}, "
            f"leftover_init={task_counts.get(Status.INIT, 0)}, "
            f"leftover_completed={task_counts.get(Status.COMPLETED, 0)}, "
            f"leftover_aborted={task_counts.get(Status.ABORTED, 0)}, "
            f"leftover_expired={task_counts.get(Status.EXPIRED, 0)}, "
            f"buffer_failed_groups={task_counts.get(Status.FAILED, 0)}, "
            f"buffer_filtered_groups={task_counts.get(Status.FILTERED, 0)}"
        )


def build_produce_batch_result(
    *,
    task_runners: list[_TaskRunner],
    task_batch_sizes: dict[str, int],
    batch_by_task: dict[str, list[list[RolloutState]]],
    leftover_counts: dict[str, dict[Status, int]],
    progress: "ProduceProgress | DisaggProduceProgress",
    pause_time_s: float,
) -> ProduceBatchResult:
    if len(task_runners) == 1:
        task = task_runners[0]
        raw_rewards_sum, raw_rewards_count = progress.consume_raw_rewards(task.task_name)
        produced_samples, produced_tokens = progress.consume_produced(task.task_name)
        produce_time_s = progress.consume_produce_time()
        result = ProduceBatchResult(
            rollout_states=batch_by_task.get(task.task_name, []),
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
            produced_samples=produced_samples,
            produced_tokens=produced_tokens,
            produce_time_s=produce_time_s,
        )
        _fill_leftover_counts(result, leftover_counts.get(task.task_name, {}))
        _merge_discarded_counts(result, progress, task.task_name)
        _fill_group_timing_stats(result, result.rollout_states, pause_time_s=pause_time_s)
        return result

    task_results: dict[str, ProduceBatchResult] = {}
    produce_time_s = progress.consume_produce_time()
    for task in task_runners:
        raw_rewards_sum, raw_rewards_count = progress.consume_raw_rewards(task.task_name)
        produced_samples, produced_tokens = progress.consume_produced(task.task_name)
        result = ProduceBatchResult(
            rollout_states=batch_by_task.get(task.task_name, []),
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
            produced_samples=produced_samples,
            produced_tokens=produced_tokens,
        )
        _fill_leftover_counts(result, leftover_counts.get(task.task_name, {}))
        _merge_discarded_counts(result, progress, task.task_name)
        task_results[task.task_name] = result

    ordered_tasks = sorted(task_runners, key=lambda task: (task.task_name, task.order))
    aggregated = aggregate_task_results(ordered_tasks, task_results)
    aggregated.produce_time_s = produce_time_s
    aggregated.task_batch_sizes = {task.task_name: task_batch_sizes[task.task_name] for task in ordered_tasks}
    _fill_group_timing_stats(aggregated, aggregated.rollout_states, pause_time_s=pause_time_s)
    return aggregated


async def take_train_batch(
    *,
    task_runners: list[_TaskRunner],
    replay_buffer: ReplayBuffer,
    logger,
    manager_name: str,
    task_batch_sizes: dict[str, int],
    progress: "ProduceProgress | DisaggProduceProgress",
    pause_time_s: float = 0.0,
) -> ProduceBatchResult:
    batch_by_task, consumed_counts = await replay_buffer.take_batch(task_batch_sizes)
    if hasattr(progress, "mark_consumed"):
        progress.mark_consumed(consumed_counts)
    task_names = [task.task_name for task in task_runners]
    leftover_counts = await replay_buffer.count_statuses(task_names, _LEFTOVER_STATUSES)
    log_buffer_counts(
        logger,
        manager_name=manager_name,
        task_runners=task_runners,
        task_batch_sizes=task_batch_sizes,
        batch_by_task=batch_by_task,
        leftover_counts=leftover_counts,
    )
    return build_produce_batch_result(
        task_runners=task_runners,
        task_batch_sizes=task_batch_sizes,
        batch_by_task=batch_by_task,
        leftover_counts=leftover_counts,
        progress=progress,
        pause_time_s=pause_time_s,
    )


def task_checkpoint_path(checkpoint_path: Path | str, task_name: str) -> Path:
    return Path(checkpoint_path) / _TASK_CHECKPOINT_DIR / task_name


def manager_state_path(checkpoint_path: Path | str) -> Path:
    return Path(checkpoint_path) / _MANAGER_STATE_PATH


def get_pending_task_counts(task_runners: list[_TaskRunner]) -> dict[str, int]:
    pending_task_counts: dict[str, int] = {}
    for task in task_runners:
        pending_count = task.produce_strategy.pending_task_count()
        if pending_count > 0:
            pending_task_counts[task.task_name] = pending_count
    return pending_task_counts


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


class _LocalPendingTasks:
    """把共卡本次调用的局部 pending set 适配成统一 drain 协议。

    共卡 pending 不跨 produce_batch 调用；这里原地修改传入的 set，让 pending_task_count() 在 pause 过程中仍能反映剩余本地任务数量。
    """

    def __init__(self, tasks: set[asyncio.Task]) -> None:
        self._tasks = tasks

    def count(self) -> int:
        return len(self._tasks)

    async def wait_and_claim(self, *, timeout_s: float) -> set[asyncio.Task]:
        if not self._tasks:
            return set()
        done, _ = await asyncio.wait(set(self._tasks), timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        self._tasks.difference_update(done)
        return done

    async def cancel_all(self) -> int:
        tasks = set(self._tasks)
        self._tasks.clear()
        if not tasks:
            return 0
        logger.warning(f"Cancelling {len(tasks)} pending rollout tasks.")
        await cancel_and_drain(list(tasks))
        return len(tasks)


async def request_agent_loop_pause(ctx: BaseProduceContext, *, pending_count: int) -> None:
    """发送一次 agent loop pause 请求。"""

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
            f"elapsed={time.perf_counter() - pause_request_start:.2f}s, pending={pending_count}"
        )
    except Exception:
        logger.exception(
            f"Agent loop pause failed: task={ctx.task_name}, "
            f"elapsed={time.perf_counter() - pause_request_start:.2f}s, pending={pending_count}"
        )


async def pause_pending_tasks(
    *,
    pending_tasks: set[asyncio.Task] | _PendingTasks,
    ctx: BaseProduceContext,
    put_claimed_task: Callable[[asyncio.Task], Awaitable[Any]],
) -> float:
    """Pause/drain pending；超时后 cancel 剩余任务。"""

    pending = _LocalPendingTasks(pending_tasks) if isinstance(pending_tasks, set) else pending_tasks
    pause_start = time.perf_counter()
    if pending.count() == 0:
        return 0.0

    initial_pending_count = pending.count()
    logger.info(
        f"Pause signal loop started for task {ctx.task_name}. "
        f"Waiting for {initial_pending_count} pending tasks to complete. "
        f"periodic_abort_interval_s={PERIODIC_ABORT_INTERVAL_S}, "
        f"producer_pause_pending_task_timeout_s={PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S}"
    )

    pending_pause_tasks = {create_task(request_agent_loop_pause(ctx, pending_count=initial_pending_count))}
    cleanup_start_time = time.perf_counter()
    next_periodic_abort_time = cleanup_start_time + PERIODIC_ABORT_INTERVAL_S
    while True:
        elapsed_time = time.perf_counter() - cleanup_start_time
        if elapsed_time > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
            cancelled_count = await pending.cancel_all()
            logger.warning(
                f"Cleanup timeout of {PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S}s reached. "
                f"Forcefully cancelling {cancelled_count} remaining tasks. task={ctx.task_name}"
            )
            break

        if pending.count() == 0:
            break
        current_time = time.perf_counter()
        pending_pause_tasks = {task for task in pending_pause_tasks if not task.done()}

        # 定时发送 pause 信号，避免后端漏掉第一次 pause 后 pending 长时间不结束。
        if PERIODIC_ABORT_INTERVAL_S > 0 and current_time >= next_periodic_abort_time:
            pending_pause_tasks.add(create_task(request_agent_loop_pause(ctx, pending_count=pending.count())))
            next_periodic_abort_time += PERIODIC_ABORT_INTERVAL_S

        claimed_done = await pending.wait_and_claim(timeout_s=1)
        for task in claimed_done:
            await put_claimed_task(task)

    await cancel_and_drain(list(pending_pause_tasks))
    pause_time = time.perf_counter() - pause_start
    logger.info(f"pause_produce completed for task {ctx.task_name} within {pause_time}s.")
    return pause_time


async def _put_claimed_tasks(
    claimed_tasks: set[asyncio.Task],
    ctx: BaseProduceContext,
    *,
    available_base: int | None = None,
    progress_displayer: _ProgressDisplayer | None = None,
) -> None:
    completed_count = 0
    for task in claimed_tasks:
        is_completed = await ctx.put_generated_group(task.result())
        if is_completed:
            completed_count += 1
        if is_completed and available_base is not None and progress_displayer is not None:
            progress_displayer.update(available_base + completed_count)
