import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import asyncio_run
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoop, AgentLoopConfig
from .producer import ProducerTimings, ProduceStrategy, ProduceStrategyConfig, SyncProduceStrategyConfig
from .sampler import Sampler, SamplerConfig


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
        leftover_completed (int): Number of completed groups remaining in the replay buffer after this batch.
        leftover_aborted (int): Number of aborted groups remaining in the replay buffer.
        leftover_expired (int): Number of expired groups remaining in the replay buffer.
    """

    rollout_states: list[list[RolloutState]]
    # per-group generation timing stats (all None if no generations occurred)
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    # leftover samples remaining in replay buffer after batch retrieval
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0


class AgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    agent_loop_config: AgentLoopConfig
    produce_strategy_config: ProduceStrategyConfig = SyncProduceStrategyConfig()
    sampler_config: SamplerConfig

    def build(
        self,
        rollout_controller: RolloutController,
        judger: Judger,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
    ) -> "AgentLoopManager":
        agent_loop = self.agent_loop_config.build(rollout_controller=rollout_controller, judger=judger, logger=logger)
        produce_strategy = self.produce_strategy_config.build()
        sampler = self.sampler_config.build(tokenizer=tokenizer, replay_buffer=replay_buffer)
        return AgentLoopManager(
            agent_loop=agent_loop,
            produce_strategy=produce_strategy,
            sampler=sampler,
            replay_buffer=replay_buffer,
            task_name=self.task_name,
            logger=logger,
        )


class AgentLoopManager:
    def __init__(
        self,
        agent_loop: AgentLoop,
        produce_strategy: ProduceStrategy,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        task_name: str,
        logger=None,
    ):
        self._agent_loop: AgentLoop = agent_loop  # 负责一条或者一组样本生成
        # TODO(@duanyanhui): ProduceStrategy是用config来build还是直接使用实例，后续再统一改，目前先使用实例
        self._scheduler: ProduceStrategy = produce_strategy
        self._replay_buffer: ReplayBuffer = replay_buffer
        self._data_sampler: Sampler = sampler  # 负责采样数据，提供给 ProduceStrategy 来生成样本
        self.task_name = task_name
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    # 共卡
    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        start = time.perf_counter()
        self.logger.info(f"[AgentLoopManager][{self.task_name}] produce_batch start batch={batch_size}")
        stats: ProducerTimings = await self._scheduler.produce_batch(
            self._agent_loop, self._data_sampler, self._replay_buffer, batch_size, self.task_name, rollout_step
        )
        self.logger.info(
            f"[AgentLoopManager][{self.task_name}] produce scheduler done elapsed={time.perf_counter() - start:.3f}, and start replay_buffer.get"
        )
        result = ProduceBatchResult(rollout_states=[])
        if stats.generate_times_s:
            sorted_times = sorted(stats.generate_times_s)
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
            result.group_gen_pause_time_s = stats.pause_time_s

        start = time.perf_counter()
        batch_rollout_states: list[list[RolloutState]] = await self._replay_buffer.get(
            batch_size, self.task_name, Status.COMPLETED
        )
        self.logger.info(
            f"[AgentLoopManager][{self.task_name}] replay_buffer.get done completed_groups={len(batch_rollout_states)} elapsed={time.perf_counter() - start:.3f}"
        )
        result.rollout_states = batch_rollout_states
        completed_sample_count, aborted_sample_count, expired_sample_count = await asyncio.gather(
            self._replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED),
            self._replay_buffer.count(task_name=self.task_name, group_status=Status.ABORTED),
            self._replay_buffer.count(task_name=self.task_name, group_status=Status.EXPIRED),
        )
        result.leftover_completed = completed_sample_count
        result.leftover_aborted = aborted_sample_count
        result.leftover_expired = expired_sample_count
        return result

    def save(self, checkpoint_path: Path | str) -> None:
        """Save the sampler's dataloader state to checkpoint."""
        self._data_sampler.save(checkpoint_path)
        asyncio_run(self._replay_buffer.save(checkpoint_path))

    def resume(self, checkpoint_path: Path | str) -> None:
        """Resume the sampler's dataloader state from checkpoint."""
        self._data_sampler.resume(checkpoint_path)
        asyncio_run(self._replay_buffer.resume(checkpoint_path))

    # # 非共卡
    # async def disaggregate_produce_batch(self, batch_size: int):
    #     self._scheduler.produce_batch(batch_size, self._data_sampler, ...)

    # async def disaggregate_get_batch(self, task_name: str, batch_size: int):
    #     # 从不同的 replay_buffer 中采样，然后训练
    #     return self._replay_buffer.get(batch_size, task_name, Status.COMPLETED)
