"""Multi-task disaggregated agent loop manager.

This module owns the replay-buffer window orchestration used by disaggregated
training. Colocated batch production stays in colocated_agent_loop_manager.py.
"""

import asyncio
import time

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto import Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController, continue_generation, pause_generation

from .manager_base import (
    BaseAgentLoopManager,
    ProduceBatchResult,
    TaskSpecConfig,
    _produce_single_task_window_to_replay_buffer,
    build_task_runners,
)


class DisaggregatedMultiTaskAgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[TaskSpecConfig]

    def build(
        self,
        rollout_controller: RolloutController,
        judger: Judger,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
    ) -> "DisaggregatedMultiTaskAgentLoopManager":
        task_runners = build_task_runners(
            self.tasks,
            rollout_controller=rollout_controller,
            judger=judger,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
        )
        return DisaggregatedMultiTaskAgentLoopManager(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            logger=logger,
        )


class DisaggregatedMultiTaskAgentLoopManager(BaseAgentLoopManager):
    def get_window_task_batch_sizes(
        self, train_batch_size: int, start_rollout_step: int, train_steps: int
    ) -> dict[str, int]:
        if train_steps <= 0:
            raise ValueError(f"train_steps must be positive, got {train_steps}")

        # window 里的目标是“多个训练步所需 batch 的总和”。
        # 这里逐 step 累加，是为了保留多 task 权重分配在每个 step 上的整数性质。
        window_task_batch_sizes = {task.task_name: 0 for task in self.task_runners}
        for offset in range(train_steps):
            step_task_batch_sizes = self.get_task_batch_sizes(train_batch_size, start_rollout_step + offset)
            self._validate_task_batch_sizes(step_task_batch_sizes, train_batch_size)
            for task_name, task_batch_size in step_task_batch_sizes.items():
                window_task_batch_sizes[task_name] += task_batch_size
        return window_task_batch_sizes

    async def produce_window_to_replay_buffer(
        self,
        required_task_batch_sizes: dict[str, int] | None = None,
        target_task_batch_sizes: dict[str, int] | None = None,
        rollout_step: int = 0,
        enable_partial_rollout: bool = False,
    ) -> ProduceBatchResult:
        start = time.perf_counter()
        self.logger.info(
            f"[DisaggregatedMultiTaskAgentLoopManager][{self.name}] produce_window_to_replay_buffer start"
        )

        if required_task_batch_sizes is None:
            raise ValueError("required_task_batch_sizes must be provided for produce_window_to_replay_buffer.")
        # required 表示“训练至少要拿到多少”，target 表示“这轮总共希望生产到多少”。
        # 在 stale / partial rollout 模式下，两者可以不同。
        self._validate_task_counts(required_task_batch_sizes)
        if target_task_batch_sizes is None:
            target_task_batch_sizes = required_task_batch_sizes
        self._validate_task_counts(target_task_batch_sizes)
        self._validate_target_task_counts(required_task_batch_sizes, target_task_batch_sizes)
        active_tasks = [
            task
            for task in self.task_runners
            if (
                required_task_batch_sizes[task.task_name] > 0
                or target_task_batch_sizes[task.task_name] > 0
            )
        ]

        results: list[ProduceBatchResult] = []
        if active_tasks:
            rollout_ctl = self._get_shared_rollout_ctl(active_tasks)
            await continue_generation(rollout_ctl)
            try:
                # 这里并发做的是“把样本打进 replay buffer”，不是立刻取训练 batch。
                results = await self._gather_fail_fast(
                    *[
                        _produce_single_task_window_to_replay_buffer(
                            task_runner=task,
                            replay_buffer=self.replay_buffer,
                            required_batch_size=required_task_batch_sizes[task.task_name],
                            target_batch_size=target_task_batch_sizes[task.task_name],
                            rollout_step=rollout_step,
                            enable_partial_rollout=enable_partial_rollout,
                            logger=self.logger,
                            manager_name="DisaggregatedMultiTaskAgentLoopManager",
                        )
                        for task in active_tasks
                    ]
                )
            finally:
                # produce_window 内部可能已经在 cleanup 阶段 pause 过一次；
                # 外层这里保留 safety pause，但静默成功日志，避免重复噪音。
                await pause_generation(rollout_ctl, log_success=False)

        task_results = {task.task_name: result for task, result in zip(active_tasks, results)}
        for task in self.task_runners:
            if task.task_name not in task_results:
                task_results[task.task_name] = ProduceBatchResult(rollout_states=[])

        ordered_tasks = list(self.task_runners)
        aggregated = self._aggregate_task_results(ordered_tasks, task_results)
        aggregated.task_batch_sizes = {task.task_name: target_task_batch_sizes[task.task_name] for task in ordered_tasks}
        aggregated.required_task_batch_sizes = {
            task.task_name: required_task_batch_sizes[task.task_name] for task in ordered_tasks
        }

        self.logger.info(
            f"[DisaggregatedMultiTaskAgentLoopManager][{self.name}] produce_window_to_replay_buffer done "
            f"elapsed={time.perf_counter() - start:.3f}, completed_leftover={aggregated.leftover_completed}"
        )
        return aggregated

    async def get_completed_batch(
        self,
        batch_size: int,
        rollout_step: int = 0,
        task_batch_sizes: dict[str, int] | None = None,
        poll_interval_s: float = 1.0,
        max_wait_s: float | None = 1800.0,
    ) -> ProduceBatchResult:
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        if task_batch_sizes is None:
            task_batch_sizes = self.get_task_batch_sizes(batch_size, rollout_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        if batch_size == 0:
            result = ProduceBatchResult(rollout_states=[])
            result.task_batch_sizes = dict(task_batch_sizes)
            result.required_task_batch_sizes = dict(task_batch_sizes)
            return result

        start = time.perf_counter()
        while True:
            # 这里故意用轮询而不是条件变量/queue：
            # 当前实现里 replay buffer 还是 trainer 进程内对象，轮询实现更简单，语义也足够清楚。
            completed_counts = await asyncio.gather(
                *[
                    self.replay_buffer.count(task_name=task.task_name, group_status=Status.COMPLETED)
                    for task in self.task_runners
                ]
            )
            completed_count_by_task = {
                task.task_name: completed_count for task, completed_count in zip(self.task_runners, completed_counts)
            }
            if all(
                completed_count_by_task[task_name] >= task_batch_size
                for task_name, task_batch_size in task_batch_sizes.items()
            ):
                break
            if max_wait_s is not None and time.perf_counter() - start >= max_wait_s:
                raise TimeoutError(
                    "Timed out waiting for completed samples in replay buffer. "
                    f"waited={max_wait_s}s, task_batch_sizes={task_batch_sizes}, "
                    f"completed_count_by_task={completed_count_by_task}"
                )
            await asyncio.sleep(poll_interval_s)

        ordered_tasks = list(self.task_runners)
        task_results: dict[str, ProduceBatchResult] = {}
        for task in ordered_tasks:
            task_batch_size = task_batch_sizes[task.task_name]
            if task_batch_size == 0:
                task_results[task.task_name] = ProduceBatchResult(rollout_states=[])
                continue
            # 这里真正把本步训练所需的 COMPLETED 样本从 replay buffer 中取走。
            batch_rollout_states = await self.replay_buffer.get(task_batch_size, task.task_name, Status.COMPLETED)
            result = ProduceBatchResult(rollout_states=batch_rollout_states)
            completed_sample_count, aborted_sample_count, expired_sample_count = await asyncio.gather(
                self.replay_buffer.count(task_name=task.task_name, group_status=Status.COMPLETED),
                self.replay_buffer.count(task_name=task.task_name, group_status=Status.ABORTED),
                self.replay_buffer.count(task_name=task.task_name, group_status=Status.EXPIRED),
            )
            result.leftover_completed = completed_sample_count
            result.leftover_aborted = aborted_sample_count
            result.leftover_expired = expired_sample_count
            task_results[task.task_name] = result

        aggregated = self._aggregate_task_results(ordered_tasks, task_results)
        aggregated.task_batch_sizes = {task.task_name: task_batch_sizes[task.task_name] for task in ordered_tasks}
        aggregated.required_task_batch_sizes = dict(aggregated.task_batch_sizes)
        return aggregated
