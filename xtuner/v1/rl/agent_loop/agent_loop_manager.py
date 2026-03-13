import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoop, AgentLoopConfig
from .producer import ProduceBatchStats, ProduceStrategy, ProduceStrategyConfig, SyncProduceStrategyConfig
from .sampler import Sampler, SamplerConfig


@dataclass
class ProduceBatchResult:
    rollout_states: list[list[RolloutState]]
    # generate timing (all None if no generations occurred)
    timing_n: int | None = None
    timing_mean_s: float | None = None
    timing_p50_s: float | None = None
    timing_p99_s: float | None = None
    timing_p99_p50_ratio: float | None = None
    timing_pause_time_s: float | None = None
    # replay buffer state after get
    completed_samples: int = 0
    aborted_samples: int = 0
    expired_samples: int = 0


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
        stats: ProduceBatchStats = await self._scheduler.produce_batch(
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
            result.timing_n = n
            result.timing_mean_s = mean_s
            result.timing_p50_s = p50_s
            result.timing_p99_s = p99_s
            result.timing_p99_p50_ratio = ratio
            result.timing_pause_time_s = stats.pause_time_s

        start = time.perf_counter()
        batch_rollout_states: list[list[RolloutState]] = await self._replay_buffer.get(
            batch_size, self.task_name, Status.COMPLETED
        )
        self.logger.info(
            f"[AgentLoopManager][{self.task_name}] replay_buffer.get done completed_groups={len(batch_rollout_states)} elapsed={time.perf_counter() - start:.3f}"
        )
        result.rollout_states = batch_rollout_states
        completed_sample_count = await self._replay_buffer.count(
            task_name=self.task_name, group_status=Status.COMPLETED
        )
        aborted_sample_count = await self._replay_buffer.count(task_name=self.task_name, group_status=Status.ABORTED)
        expired_sample_count = await self._replay_buffer.count(task_name=self.task_name, group_status=Status.EXPIRED)

        result.completed_samples = completed_sample_count
        result.aborted_samples = aborted_sample_count
        result.expired_samples = expired_sample_count
        return result

    # # 非共卡
    # async def disaggregate_produce_batch(self, batch_size: int):
    #     self._scheduler.produce_batch(batch_size, self._data_sampler, ...)

    # async def disaggregate_get_batch(self, task_name: str, batch_size: int):
    #     # 从不同的 replay_buffer 中采样，然后训练
    #     return self._replay_buffer.get(batch_size, task_name, Status.COMPLETED)
