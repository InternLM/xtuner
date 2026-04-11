"""Single-task disaggregated agent loop manager.

这个版本把单 task 场景从多 task dict/聚合逻辑里解耦出来，保留最直接的
window produce / completed batch 获取接口。
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
    build_task_runner,
)


class DisaggregatedSingleTaskAgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task: TaskSpecConfig

    def build(
        self,
        rollout_controller: RolloutController,
        judger: Judger,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
    ) -> "DisaggregatedSingleTaskAgentLoopManager":
        task_runner = build_task_runner(
            self.task,
            rollout_controller=rollout_controller,
            judger=judger,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
        )
        return DisaggregatedSingleTaskAgentLoopManager(
            task_runner=task_runner,
            replay_buffer=replay_buffer,
            logger=logger,
        )


class DisaggregatedSingleTaskAgentLoopManager(BaseAgentLoopManager):
    def __init__(self, task_runner, replay_buffer: ReplayBuffer, logger=None):
        super().__init__([task_runner], replay_buffer=replay_buffer, logger=logger)
        self.task_runner = task_runner

    def get_window_batch_size(self, train_batch_size: int, start_rollout_step: int, train_steps: int) -> int:
        if train_batch_size < 0:
            raise ValueError(f"train_batch_size must be non-negative, got {train_batch_size}")
        if train_steps <= 0:
            raise ValueError(f"train_steps must be positive, got {train_steps}")
        # 保留 start_rollout_step 这个参数只是为了和 multi-task manager 共享同一组接口；
        # 单 task 的 required window size 仅由 train_batch_size * train_steps 决定。
        return train_batch_size * train_steps

    def get_window_task_batch_sizes(
        self, train_batch_size: int, start_rollout_step: int, train_steps: int
    ) -> dict[str, int]:
        # 兼容旧的 trainer / manager 调用形状；单 task 主路径应优先走标量接口。
        return {self.task_runner.task_name: self.get_window_batch_size(train_batch_size, start_rollout_step, train_steps)}

    async def produce_window(
        self,
        required_batch_size: int,
        target_batch_size: int | None = None,
        rollout_step: int = 0,
        enable_partial_rollout: bool = False,
    ) -> ProduceBatchResult:
        task_name = self.task_runner.task_name
        target_batch_size = required_batch_size if target_batch_size is None else target_batch_size
        start = time.perf_counter()
        self.logger.info(
            f"[DisaggregatedSingleTaskAgentLoopManager][{task_name}] "
            "produce_window_to_replay_buffer start"
        )

        rollout_ctl = self.task_runner.agent_loop.rollout_ctl
        await continue_generation(rollout_ctl)
        try:
            result = await _produce_single_task_window_to_replay_buffer(
                task_runner=self.task_runner,
                replay_buffer=self.replay_buffer,
                required_batch_size=required_batch_size,
                target_batch_size=target_batch_size,
                rollout_step=rollout_step,
                enable_partial_rollout=enable_partial_rollout,
                logger=self.logger,
                manager_name="DisaggregatedSingleTaskAgentLoopManager",
            )
        finally:
            # produce_window 内部可能已经在 cleanup 阶段 pause 过一次；
            # 外层这里保留 safety pause，但静默成功日志，避免重复噪音。
            await pause_generation(rollout_ctl, log_success=False)

        result.task_batch_sizes = {task_name: target_batch_size}
        result.required_task_batch_sizes = {task_name: required_batch_size}
        result.task_results = {task_name: self._copy_task_result(result)}
        self.logger.info(
            f"[DisaggregatedSingleTaskAgentLoopManager][{task_name}] "
            f"produce_window_to_replay_buffer done elapsed={time.perf_counter() - start:.3f}, "
            f"completed_leftover={result.leftover_completed}"
        )
        return result

    async def produce_window_to_replay_buffer(
        self,
        required_task_batch_sizes: dict[str, int] | None = None,
        target_task_batch_sizes: dict[str, int] | None = None,
        rollout_step: int = 0,
        enable_partial_rollout: bool = False,
    ) -> ProduceBatchResult:
        if required_task_batch_sizes is None:
            raise ValueError("required_task_batch_sizes must be provided for produce_window_to_replay_buffer.")
        self._validate_task_counts(required_task_batch_sizes)
        if target_task_batch_sizes is not None:
            self._validate_task_counts(target_task_batch_sizes)
            self._validate_target_task_counts(required_task_batch_sizes, target_task_batch_sizes)
        required_batch_size = required_task_batch_sizes[self.task_runner.task_name]
        target_batch_size = (
            required_batch_size
            if target_task_batch_sizes is None
            else target_task_batch_sizes[self.task_runner.task_name]
        )
        return await self.produce_window(
            required_batch_size=required_batch_size,
            target_batch_size=target_batch_size,
            rollout_step=rollout_step,
            enable_partial_rollout=enable_partial_rollout,
        )

    async def get_completed_batch_single(
        self,
        batch_size: int,
        poll_interval_s: float = 1.0,
        max_wait_s: float | None = 1800.0,
    ) -> ProduceBatchResult:
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        task_name = self.task_runner.task_name

        if batch_size == 0:
            result = ProduceBatchResult(rollout_states=[])
            result.task_batch_sizes = {task_name: 0}
            result.required_task_batch_sizes = {task_name: 0}
            result.task_results = {task_name: self._copy_task_result(result)}
            return result

        start = time.perf_counter()
        while True:
            completed_count = await self.replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
            if completed_count >= batch_size:
                break
            if max_wait_s is not None and time.perf_counter() - start >= max_wait_s:
                raise TimeoutError(
                    "Timed out waiting for completed samples in replay buffer. "
                    f"waited={max_wait_s}s, task_name={task_name}, "
                    f"task_batch_size={batch_size}, completed_count={completed_count}"
                )
            await asyncio.sleep(poll_interval_s)

        batch_rollout_states = await self.replay_buffer.get(batch_size, task_name, Status.COMPLETED)
        result = ProduceBatchResult(rollout_states=batch_rollout_states)
        completed_sample_count = await self.replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        aborted_sample_count = await self.replay_buffer.count(task_name=task_name, group_status=Status.ABORTED)
        expired_sample_count = await self.replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
        result.leftover_completed = completed_sample_count
        result.leftover_aborted = aborted_sample_count
        result.leftover_expired = expired_sample_count
        result.task_batch_sizes = {task_name: batch_size}
        result.required_task_batch_sizes = {task_name: batch_size}
        result.task_results = {task_name: self._copy_task_result(result)}
        return result

    async def get_completed_batch(
        self,
        batch_size: int,
        rollout_step: int = 0,
        task_batch_sizes: dict[str, int] | None = None,
        poll_interval_s: float = 1.0,
        max_wait_s: float | None = 1800.0,
    ) -> ProduceBatchResult:
        task_name = self.task_runner.task_name
        if task_batch_sizes is not None:
            self._validate_task_counts(task_batch_sizes)
            batch_size = task_batch_sizes[task_name]
        # 保留 rollout_step 是为了和 multi-task manager 的公共接口对齐；
        # 单 task completed batch 获取不依赖当前 rollout_step。
        return await self.get_completed_batch_single(
            batch_size=batch_size,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
        )
