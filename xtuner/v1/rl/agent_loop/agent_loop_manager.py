import asyncio
import math
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController, continue_generation, pause_generation
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


@dataclass
class MultiEnvProduceBatchResult(ProduceBatchResult):
    """Aggregated result for a multi-env produce_batch call."""

    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, ProduceBatchResult] | None = None


@dataclass(frozen=True)
class _EnvRunner:
    env_name: str
    task_name: str
    agent_loop: AgentLoop
    produce_strategy: ProduceStrategy
    sampler: Sampler
    weight: float = 1.0
    order: int = 0


class _MultiEnvSamplerView:
    def __init__(self, samplers: list[Sampler]):
        self._samplers = samplers

    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self._samplers)


def _fill_produce_timing_stats(result: ProduceBatchResult, stats: ProducerTimings) -> None:
    if not stats.generate_times_s:
        return
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


async def _produce_single_env_batch(
    env_runner: _EnvRunner,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    rollout_step: int,
    logger,
    manager_name: str,
) -> ProduceBatchResult:
    start = time.perf_counter()
    logger.info(f"[{manager_name}][{env_runner.task_name}] produce_batch start batch={batch_size}")
    stats: ProducerTimings = await env_runner.produce_strategy.produce_batch(
        env_runner.agent_loop,
        env_runner.sampler,
        replay_buffer,
        batch_size,
        env_runner.task_name,
        rollout_step,
    )
    logger.info(
        f"[{manager_name}][{env_runner.task_name}] produce scheduler done elapsed={time.perf_counter() - start:.3f}, and start replay_buffer.get"
    )

    result = ProduceBatchResult(rollout_states=[])
    _fill_produce_timing_stats(result, stats)

    start = time.perf_counter()
    batch_rollout_states: list[list[RolloutState]] = await replay_buffer.get(
        batch_size, env_runner.task_name, Status.COMPLETED
    )
    logger.info(
        f"[{manager_name}][{env_runner.task_name}] replay_buffer.get done completed_groups={len(batch_rollout_states)} elapsed={time.perf_counter() - start:.3f}"
    )
    result.rollout_states = batch_rollout_states
    completed_sample_count, aborted_sample_count, expired_sample_count = await asyncio.gather(
        replay_buffer.count(task_name=env_runner.task_name, group_status=Status.COMPLETED),
        replay_buffer.count(task_name=env_runner.task_name, group_status=Status.ABORTED),
        replay_buffer.count(task_name=env_runner.task_name, group_status=Status.EXPIRED),
    )
    result.leftover_completed = completed_sample_count
    result.leftover_aborted = aborted_sample_count
    result.leftover_expired = expired_sample_count
    return result


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


class MultiEnvSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    env_name: str
    weight: float = Field(ge=0.0)
    agent_loop_config: AgentLoopConfig
    produce_strategy_config: ProduceStrategyConfig = SyncProduceStrategyConfig()
    sampler_config: SamplerConfig
    task_name: str | None = None


class MultiEnvAgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    envs: list[MultiEnvSpecConfig]

    def build(
        self,
        rollout_controller: RolloutController,
        judger: Judger,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
    ) -> "MultiEnvAgentLoopManager":
        if not self.envs:
            raise ValueError("MultiEnvAgentLoopManagerConfig requires at least one env config.")

        seen_env_names: set[str] = set()
        seen_task_names: set[str] = set()
        managed_envs: list[_EnvRunner] = []
        for order, env_cfg in enumerate(self.envs):
            if env_cfg.env_name in seen_env_names:
                raise ValueError(f"Duplicate env_name found in MultiEnvAgentLoopManagerConfig: {env_cfg.env_name}")
            seen_env_names.add(env_cfg.env_name)

            env_task_name = env_cfg.task_name or f"{self.task_name}:{env_cfg.env_name}"
            if env_task_name in seen_task_names:
                raise ValueError(
                    f"Duplicate task_name found in MultiEnvAgentLoopManagerConfig after resolution: {env_task_name}"
                )
            seen_task_names.add(env_task_name)

            agent_loop = env_cfg.agent_loop_config.build(
                rollout_controller=rollout_controller,
                judger=judger,
                logger=logger,
            )
            produce_strategy = env_cfg.produce_strategy_config.build()
            sampler = env_cfg.sampler_config.build(tokenizer=tokenizer, replay_buffer=replay_buffer)
            managed_envs.append(
                _EnvRunner(
                    env_name=env_cfg.env_name,
                    task_name=env_task_name,
                    agent_loop=agent_loop,
                    produce_strategy=produce_strategy,
                    sampler=sampler,
                    weight=env_cfg.weight,
                    order=order,
                )
            )

        return MultiEnvAgentLoopManager(
            managed_envs=managed_envs,
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
        self.env_runner = _EnvRunner(
            env_name=task_name,
            task_name=task_name,
            agent_loop=agent_loop,
            produce_strategy=produce_strategy,
            sampler=sampler,
        )
        self.agent_loop: AgentLoop = agent_loop
        self.scheduler: ProduceStrategy = produce_strategy
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.data_sampler: Sampler = sampler
        self.task_name = task_name
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    # 共卡
    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        rollout_ctl = self.agent_loop.rollout_ctl
        await continue_generation(rollout_ctl)
        try:
            return await _produce_single_env_batch(
                env_runner=self.env_runner,
                replay_buffer=self.replay_buffer,
                batch_size=batch_size,
                rollout_step=rollout_step,
                logger=self.logger,
                manager_name="AgentLoopManager",
            )
        finally:
            await pause_generation(rollout_ctl)

    def save(self, checkpoint_path: Path | str) -> None:
        """Save the sampler's dataloader state to checkpoint."""
        self.data_sampler.save(checkpoint_path)
        asyncio_run(self.replay_buffer.save(checkpoint_path))

    def resume(self, checkpoint_path: Path | str) -> None:
        """Resume the sampler's dataloader state from checkpoint."""
        self.data_sampler.resume(checkpoint_path)
        asyncio_run(self.replay_buffer.resume(checkpoint_path))

    # # 非共卡
    # async def disaggregate_produce_batch(self, batch_size: int):
    #     self.scheduler.produce_batch(batch_size, self.data_sampler, ...)

    # async def disaggregate_get_batch(self, task_name: str, batch_size: int):
    #     # 从不同的 replay_buffer 中采样，然后训练
    #     return self.replay_buffer.get(batch_size, task_name, Status.COMPLETED)


class MultiEnvAgentLoopManager:
    _ENV_CHECKPOINT_DIR = "envs"

    def __init__(
        self,
        managed_envs: list[_EnvRunner],
        replay_buffer: ReplayBuffer,
        task_name: str,
        logger=None,
    ):
        if not managed_envs:
            raise ValueError("MultiEnvAgentLoopManager requires at least one managed env.")
        if sum(env.weight for env in managed_envs) <= 0:
            raise ValueError("At least one env weight must be positive for MultiEnvAgentLoopManager.")

        self.managed_envs = managed_envs
        self.replay_buffer = replay_buffer
        self.data_sampler = _MultiEnvSamplerView([env.sampler for env in managed_envs])
        self.task_name = task_name
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    def get_task_batch_sizes(self, global_batch_size: int, rollout_step: int) -> dict[str, int]:
        """Return the per-task batch sizes for the current rollout step.

        Subclasses may override this method to implement custom dynamic batch allocation policies. Returning 0 for a
        task effectively disables that env for the current produce_batch call.
        """
        if global_batch_size < 0:
            raise ValueError(f"global_batch_size must be non-negative, got {global_batch_size}")

        total_weight = sum(env.weight for env in self.managed_envs)
        if total_weight <= 0:
            raise ValueError("Sum of env weights must be positive.")
        if global_batch_size == 0:
            return {env.task_name: 0 for env in self.managed_envs}

        raw_allocations = [global_batch_size * env.weight / total_weight for env in self.managed_envs]
        floor_allocations = [math.floor(raw) for raw in raw_allocations]
        remaining = global_batch_size - sum(floor_allocations)

        task_batch_sizes = {env.task_name: floor_allocations[idx] for idx, env in enumerate(self.managed_envs)}
        if remaining <= 0:
            return task_batch_sizes

        ranked_envs = sorted(
            enumerate(self.managed_envs),
            key=lambda item: (
                -(raw_allocations[item[0]] - floor_allocations[item[0]]),
                item[1].order,
            ),
        )
        for idx, env in ranked_envs[:remaining]:
            task_batch_sizes[env.task_name] += 1
        return task_batch_sizes

    def _validate_task_batch_sizes(self, task_batch_sizes: dict[str, int], global_batch_size: int) -> None:
        expected_task_names = {env.task_name for env in self.managed_envs}
        actual_task_names = set(task_batch_sizes.keys())
        if actual_task_names != expected_task_names:
            missing_task_names = expected_task_names - actual_task_names
            extra_task_names = actual_task_names - expected_task_names
            raise ValueError(
                "Invalid task batch sizes returned by get_task_batch_sizes: "
                f"missing={sorted(missing_task_names)}, extra={sorted(extra_task_names)}"
            )

        negative_batch_sizes = {
            task_name: task_batch_size
            for task_name, task_batch_size in task_batch_sizes.items()
            if task_batch_size < 0
        }
        if negative_batch_sizes:
            raise ValueError(f"Task batch sizes must be non-negative, got {negative_batch_sizes}")

        total_batch_size = sum(task_batch_sizes.values())
        if total_batch_size != global_batch_size:
            raise ValueError(
                "Task batch sizes must sum to the requested global batch size, "
                f"got total={total_batch_size}, expected={global_batch_size}"
            )

    @staticmethod
    def _aggregate_env_results(
        ordered_envs: list[_EnvRunner], task_results: dict[str, ProduceBatchResult]
    ) -> MultiEnvProduceBatchResult:
        rollout_states: list[list[RolloutState]] = []
        leftover_completed = 0
        leftover_aborted = 0
        leftover_expired = 0
        total_group_count = 0
        weighted_group_mean_sum = 0.0
        weighted_group_p50_sum = 0.0
        weighted_group_p99_sum = 0.0
        weighted_group_ratio_sum = 0.0
        total_pause_time_s = 0.0

        for env in ordered_envs:
            result = task_results[env.task_name]
            rollout_states.extend(result.rollout_states)
            leftover_completed += result.leftover_completed
            leftover_aborted += result.leftover_aborted
            leftover_expired += result.leftover_expired
            if result.group_gen_count is not None and result.group_gen_mean_s is not None:
                total_group_count += result.group_gen_count
                weighted_group_mean_sum += result.group_gen_count * result.group_gen_mean_s
                weighted_group_p50_sum += result.group_gen_count * (result.group_gen_p50_s or 0.0)
                weighted_group_p99_sum += result.group_gen_count * (result.group_gen_p99_s or 0.0)
                weighted_group_ratio_sum += result.group_gen_count * (result.group_gen_p99_p50_ratio or 0.0)
                total_pause_time_s += result.group_gen_pause_time_s or 0.0

        aggregated = MultiEnvProduceBatchResult(
            rollout_states=rollout_states,
            leftover_completed=leftover_completed,
            leftover_aborted=leftover_aborted,
            leftover_expired=leftover_expired,
            task_results={env.task_name: task_results[env.task_name] for env in ordered_envs},
        )
        if total_group_count > 0:
            aggregated.group_gen_count = total_group_count
            aggregated.group_gen_mean_s = weighted_group_mean_sum / total_group_count
            aggregated.group_gen_p50_s = weighted_group_p50_sum / total_group_count
            aggregated.group_gen_p99_s = weighted_group_p99_sum / total_group_count
            aggregated.group_gen_p99_p50_ratio = weighted_group_ratio_sum / total_group_count
            aggregated.group_gen_pause_time_s = total_pause_time_s
        return aggregated

    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> MultiEnvProduceBatchResult:
        start = time.perf_counter()
        self.logger.info(f"[MultiEnvAgentLoopManager][{self.task_name}] produce_batch start batch={batch_size}")

        task_batch_sizes = self.get_task_batch_sizes(batch_size, rollout_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        active_envs = [env for env in self.managed_envs if task_batch_sizes[env.task_name] > 0]

        results: list[ProduceBatchResult] = []
        if active_envs:
            rollout_ctl = active_envs[0].agent_loop.rollout_ctl
            await continue_generation(rollout_ctl)
            try:
                results = await asyncio.gather(
                    *[
                        _produce_single_env_batch(
                            env_runner=env,
                            replay_buffer=self.replay_buffer,
                            batch_size=task_batch_sizes[env.task_name],
                            rollout_step=rollout_step,
                            logger=self.logger,
                            manager_name="MultiEnvAgentLoopManager",
                        )
                        for env in active_envs
                    ]
                )
            finally:
                await pause_generation(rollout_ctl)

        task_results = {env.task_name: result for env, result in zip(active_envs, results)}
        for env in self.managed_envs:
            if env.task_name not in task_results:
                task_results[env.task_name] = ProduceBatchResult(rollout_states=[])

        ordered_envs = sorted(self.managed_envs, key=lambda env: (env.task_name, env.order))
        aggregated = self._aggregate_env_results(ordered_envs, task_results)
        aggregated.task_batch_sizes = {env.task_name: task_batch_sizes[env.task_name] for env in ordered_envs}

        self.logger.info(
            f"[MultiEnvAgentLoopManager][{self.task_name}] produce_batch done elapsed={time.perf_counter() - start:.3f}, completed_groups={len(aggregated.rollout_states)}"
        )
        return aggregated

    def _env_checkpoint_path(self, checkpoint_path: Path | str, env_name: str) -> Path:
        checkpoint_path = Path(checkpoint_path)
        return checkpoint_path / self._ENV_CHECKPOINT_DIR / env_name

    def save(self, checkpoint_path: Path | str) -> None:
        """Save all env sampler states and the shared replay buffer."""
        for env in self.managed_envs:
            env_checkpoint_path = self._env_checkpoint_path(checkpoint_path, env.env_name)
            env_checkpoint_path.mkdir(parents=True, exist_ok=True)
            env.sampler.save(env_checkpoint_path)
        asyncio_run(self.replay_buffer.save(checkpoint_path))

    def resume(self, checkpoint_path: Path | str) -> None:
        """Resume all env sampler states and the shared replay buffer."""
        for env in self.managed_envs:
            env.sampler.resume(self._env_checkpoint_path(checkpoint_path, env.env_name))
        asyncio_run(self.replay_buffer.resume(checkpoint_path))
