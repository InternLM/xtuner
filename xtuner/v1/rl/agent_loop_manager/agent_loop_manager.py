import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop import AgentLoopConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig, build_judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.utils import get_logger

from .produce_utils import (
    _MANAGER_STATE_PATH,
    _STATUS_POLL_INTERVAL_S,
    _TASK_CHECKPOINT_DIR,
    ProduceBatchResult,
    _TaskRunner,
    _TaskSamplerView,
    allocate_task_batch_sizes,
    get_pending_task_counts,
    manager_state_path,
    refresh_for_all_tasks,
    take_train_batch,
    task_checkpoint_path,
)
from .producer import (
    ProduceContext,
    ProduceProgress,
    ProduceStrategy,
    ProduceStrategyConfig,
    SyncProduceStrategyConfig,
)
from .sampler import Sampler, SamplerConfig


class TaskSpecConfig(BaseModel):
    """Configuration for one task managed by ``AgentLoopManager``.

    A task spec binds together the dataset sampler, agent loop, optional judger,
    production strategy, and sampling weight for one RL data source. Multi-task
    training is represented as a list of ``TaskSpecConfig`` objects.

    Args:
        task_name (str): Unique task name used for logging, replay-buffer
            routing, and checkpoint state.
        weight (float): Relative batch allocation weight for this task in
            multi-task training. Defaults to 1.0.
        agent_loop_config (AgentLoopConfig): Agent loop configuration used to
            generate rollout samples for this task.
        judger_config (JudgerConfig | ComposedJudgerConfig | None): Optional
            judger configuration used to score generated samples. Defaults to
            None.
        produce_strategy_config (ProduceStrategyConfig): Strategy used to
            produce rollout samples. Defaults to ``SyncProduceStrategyConfig``.
        sampler_config (SamplerConfig): Dataset sampler configuration for this
            task.

    **Examples:**

    Example configuration for one task::

        task = TaskSpecConfig(
            task_name="gsm8k",
            weight=1.0,
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint="Qwen/Qwen3-8B",
                sample_params=SampleParams(max_tokens=1024),
            ),
            judger_config=GSM8KJudgerConfig(),
            sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=8),
        )
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    weight: float = Field(default=1.0, ge=0.0)
    agent_loop_config: AgentLoopConfig
    judger_config: JudgerConfig | ComposedJudgerConfig | None = None
    produce_strategy_config: ProduceStrategyConfig = SyncProduceStrategyConfig()
    sampler_config: SamplerConfig


class AgentLoopManagerConfig(BaseModel):
    """Configuration for the agent loop manager.

    ``AgentLoopManagerConfig`` defines the rollout-producing side of RL
    training. It may manage a single task or a weighted list of tasks, and each
    task owns its sampler, agent loop, judger, and production strategy.

    Args:
        tasks (list[TaskSpecConfig] | TaskSpecConfig): One task config or a
            list of task configs. Task names must be unique when a list is
            provided.

    **Examples:**

    Example configuration for a single-task manager::

        config = AgentLoopManagerConfig(
            tasks=TaskSpecConfig(
                task_name="gsm8k",
                agent_loop_config=SingleTurnAgentLoopConfig(
                    hf_checkpoint="Qwen/Qwen3-8B",
                    sample_params=SampleParams(max_tokens=1024),
                ),
                judger_config=GSM8KJudgerConfig(),
                sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=8),
            )
        )
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[TaskSpecConfig] | TaskSpecConfig

    def build(
        self,
        rollout_controller: RolloutController,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
        sync_weights_interval: int = 1,
    ) -> "AgentLoopManager":
        tasks = self.tasks if isinstance(self.tasks, list) else [self.tasks]
        if not tasks:
            raise ValueError("AgentLoopManagerConfig requires at least one task config.")

        seen_task_names: set[str] = set()
        task_runners: list[_TaskRunner] = []
        for order, task_cfg in enumerate(tasks):
            if task_cfg.task_name in seen_task_names:
                raise ValueError(f"Duplicate task_name found in AgentLoopManagerConfig: {task_cfg.task_name}")
            seen_task_names.add(task_cfg.task_name)

            agent_loop = task_cfg.agent_loop_config.build(
                rollout_controller=rollout_controller,
                judger=build_judger(task_cfg.judger_config) if task_cfg.judger_config is not None else None,
                logger=logger,
            )
            produce_strategy = task_cfg.produce_strategy_config.build(
                sync_weights_interval=sync_weights_interval,
                rollout_controller=rollout_controller,
            )
            sampler = task_cfg.sampler_config.build(tokenizer=tokenizer, replay_buffer=replay_buffer)
            task_runners.append(
                _TaskRunner(
                    task_name=task_cfg.task_name,
                    agent_loop=agent_loop,
                    produce_strategy=produce_strategy,
                    sampler=sampler,
                    weight=task_cfg.weight,
                    order=order,
                )
            )

        return AgentLoopManager(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            rollout_controller=rollout_controller,
            logger=logger,
        )


class AgentLoopManager:
    _TASK_CHECKPOINT_DIR = _TASK_CHECKPOINT_DIR
    _MANAGER_STATE_PATH = _MANAGER_STATE_PATH
    _STATUS_POLL_INTERVAL_S = _STATUS_POLL_INTERVAL_S
    task_runners: list[_TaskRunner]
    replay_buffer: ReplayBuffer
    _rollout_controller: RolloutController
    data_sampler: Sampler | _TaskSamplerView
    name: str
    logger: Any
    task_names: list[str]

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger=None,
    ):
        if not task_runners:
            raise ValueError("AgentLoopManager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for AgentLoopManager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self._rollout_controller = rollout_controller
        self.data_sampler = (
            task_runners[0].sampler
            if len(task_runners) == 1
            else _TaskSamplerView([task.sampler for task in task_runners])
        )
        self.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
        self.logger = get_logger() if logger is None else logger
        self.task_names = [task.task_name for task in task_runners]

    async def produce_batch(
        self,
        batch_size: int,
        train_step: int,
        *,
        model_step: int,
    ) -> ProduceBatchResult:
        # 共卡同步入口：生产入 buffer -> pause/drain 本轮 pending -> 取非空训练 batch。
        if batch_size <= 0:
            raise ValueError(f"produce_batch expects batch_size > 0, got {batch_size}")
        start = time.perf_counter()
        self.logger.info(
            f"[AgentLoopManager][{self.name}] Start produce_batch: train_step={train_step} model_step={model_step} batch_size={batch_size}"
        )
        current_sizes = allocate_task_batch_sizes(self.task_runners, batch_size, train_step)
        active_tasks = [task for task in self.task_runners if current_sizes[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        await self._rollout_controller.continue_generation.remote()  # type: ignore[attr-defined]
        local_progress = ProduceProgress.build(
            task_names=self.task_names,
            target_samples=current_sizes,
        )
        # 生产前刷新已有 completed / aborted 的 staleness。
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            train_step=train_step,
            statuses=[Status.COMPLETED, Status.ABORTED],
        )
        produce_start = time.perf_counter()
        produce_futures = []
        for task in active_tasks:
            produce_strategy = cast(ProduceStrategy, task.produce_strategy)
            produce_futures.append(
                produce_strategy.produce_batch(
                    ProduceContext(
                        agent_loop=task.agent_loop,
                        sampler=task.sampler,
                        replay_buffer=self.replay_buffer,
                        task_batch_size=current_sizes[task.task_name],
                        task_name=task.task_name,
                        train_step=train_step,
                        model_step=model_step,
                        progress=local_progress,
                        is_valid_sample_fn=task.is_valid_sample_fn,
                        stale_threshold=task.stale_threshold,
                    )
                )
            )
        await asyncio.gather(*produce_futures)
        local_progress.add_produce_time(time.perf_counter() - produce_start)

        # pause 只收尾本轮本地 pending。
        await self._rollout_controller.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in active_tasks:
            produce_strategy = cast(ProduceStrategy, task.produce_strategy)
            pause_time_s += await produce_strategy.pause_produce(
                ProduceContext(
                    agent_loop=task.agent_loop,
                    sampler=task.sampler,
                    replay_buffer=self.replay_buffer,
                    task_batch_size=0,
                    task_name=task.task_name,
                    train_step=train_step,
                    model_step=model_step,
                    progress=local_progress,
                    is_valid_sample_fn=task.is_valid_sample_fn,
                    stale_threshold=task.stale_threshold,
                )
            )
        result = await take_train_batch(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            task_batch_sizes=current_sizes,
            progress=local_progress,
            pause_time_s=pause_time_s,
        )
        assert result.rollout_states, (
            "AgentLoopManager.produce_batch() must return non-empty rollout_states for colocated training. "
            "Use get_batch() for disaggregated empty/expired reads."
        )

        self.logger.info(
            f"[AgentLoopManager][{self.name}] produce_batch done "
            f"elapsed={time.perf_counter() - start:.3f}, completed_groups={len(result.rollout_states)}"
        )
        return result

    async def save(
        self,
        checkpoint_path: Path | str,
        model_step: int,
        *,
        no_save_replay_buffer: bool = False,
    ) -> None:
        """Save all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pending_task_counts = get_pending_task_counts(self.task_runners)
        if pending_task_counts:
            raise RuntimeError(
                "Cannot save AgentLoopManager while pending rollout tasks still exist: "
                f"{pending_task_counts}. Finish the current produce_batch before saving."
            )
        for task in self.task_runners:
            checkpoint_dir = task_checkpoint_path(checkpoint_path, task.task_name)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            task.sampler.save(checkpoint_dir)
        # manager 层保持 async 语义；同步入口只允许在 trainer 边界用 asyncio_run 包起来。
        if no_save_replay_buffer:
            self.logger.info(f"Skip saving replay buffer to {checkpoint_path}")
        else:
            await self.replay_buffer.save(checkpoint_path)
        state_path = manager_state_path(checkpoint_path)
        with state_path.open("w") as f:
            json.dump(
                {
                    "model_step": model_step,
                    "replay_buffer_saved": not no_save_replay_buffer,
                },
                f,
            )

    async def resume(self, checkpoint_path: Path | str) -> int:
        """Resume all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        for task in self.task_runners:
            task.sampler.resume(task_checkpoint_path(checkpoint_path, task.task_name))

        state_path = manager_state_path(checkpoint_path)
        with state_path.open("r") as f:
            manager_state = json.load(f)
        if manager_state.get("replay_buffer_saved", True):
            # replay buffer 恢复是 async I/O，不能在已有 event loop 中再次嵌套 asyncio_run。
            await self.replay_buffer.resume(checkpoint_path)
        elif len(self.replay_buffer) > 0:
            raise RuntimeError("Cannot resume without replay buffer checkpoint into a non-empty buffer")
        else:
            self.logger.info(f"Skip replay buffer resume for checkpoint without replay buffer: {checkpoint_path}")
        return manager_state["model_step"]
