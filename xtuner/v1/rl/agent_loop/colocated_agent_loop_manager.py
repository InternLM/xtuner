"""Colocated agent loop manager.

This module only keeps the produce_batch orchestration path that is used by
shared-card training/evaluation. Disaggregated replay-buffer windowing lives in
disaggregated_agent_loop_manager.py.
"""

import time

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController, continue_generation, pause_generation

from .manager_base import (
    BaseAgentLoopManager,
    ProduceBatchResult,
    TaskSpecConfig,
    _produce_single_task_batch,
    build_task_runners,
)


class ColocatedAgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[TaskSpecConfig] | TaskSpecConfig

    def build(
        self,
        rollout_controller: RolloutController,
        judger: Judger,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
    ) -> "ColocatedAgentLoopManager":
        task_runners = build_task_runners(
            self.tasks,
            rollout_controller=rollout_controller,
            judger=judger,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
        )
        return ColocatedAgentLoopManager(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            logger=logger,
        )


class ColocatedAgentLoopManager(BaseAgentLoopManager):
    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        start = time.perf_counter()
        self.logger.info(f"[ColocatedAgentLoopManager][{self.name}] produce_batch start batch={batch_size}")
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")

        if len(self.task_runners) == 1:
            task = self.task_runners[0]
            if batch_size == 0:
                result = ProduceBatchResult(rollout_states=[])
                result.task_batch_sizes = {task.task_name: 0}
                result.required_task_batch_sizes = {task.task_name: 0}
                result.task_results = {task.task_name: self._copy_task_result(result)}
                return result
            rollout_ctl = task.agent_loop.rollout_ctl
            await continue_generation(rollout_ctl)
            try:
                result = await _produce_single_task_batch(
                    task_runner=task,
                    replay_buffer=self.replay_buffer,
                    batch_size=batch_size,
                    rollout_step=rollout_step,
                    logger=self.logger,
                    manager_name="ColocatedAgentLoopManager",
                )
                result.required_task_batch_sizes = {task.task_name: batch_size}
                result.task_results = {task.task_name: self._copy_task_result(result)}
                return result
            finally:
                await pause_generation(rollout_ctl)

        task_batch_sizes = self.get_task_batch_sizes(batch_size, rollout_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        active_tasks = [task for task in self.task_runners if task_batch_sizes[task.task_name] > 0]

        results: list[ProduceBatchResult] = []
        if active_tasks:
            rollout_ctl = self._get_shared_rollout_ctl(active_tasks)
            await continue_generation(rollout_ctl)
            try:
                results = await self._gather_fail_fast(
                    *[
                        _produce_single_task_batch(
                            task_runner=task,
                            replay_buffer=self.replay_buffer,
                            batch_size=task_batch_sizes[task.task_name],
                            rollout_step=rollout_step,
                            logger=self.logger,
                            manager_name="ColocatedAgentLoopManager",
                        )
                        for task in active_tasks
                    ]
                )
            finally:
                await pause_generation(rollout_ctl)

        task_results = {task.task_name: result for task, result in zip(active_tasks, results)}
        for task in self.task_runners:
            if task.task_name not in task_results:
                task_results[task.task_name] = ProduceBatchResult(rollout_states=[])

        ordered_tasks = list(self.task_runners)
        aggregated = self._aggregate_task_results(ordered_tasks, task_results)
        aggregated.task_batch_sizes = {task.task_name: task_batch_sizes[task.task_name] for task in ordered_tasks}
        aggregated.required_task_batch_sizes = dict(aggregated.task_batch_sizes)

        self.logger.info(
            f"[ColocatedAgentLoopManager][{self.name}] produce_batch done elapsed={time.perf_counter() - start:.3f}, completed_groups={len(aggregated.rollout_states)}"
        )
        return aggregated
