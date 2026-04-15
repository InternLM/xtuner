import json
from pathlib import Path
from typing import Union

import ray
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from transformers import AutoTokenizer
from xtuner.v1._writer import get_writer
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.data_proto import RolloutState, refresh_seq_staleness
from xtuner.v1.rl.agent_loop import (
    AgentLoopManagerConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import JudgerConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer.worker import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run
from xtuner.v1.train.rl_colocate_trainer import (
    RLColocateTrainer,
    TrainInfo,
    check_fa3,
)
from xtuner.v1.train.trainer import LoadCheckpointConfig, XTunerMeta
from xtuner.v1.utils import is_hf_model_path, set_deterministic, timer


class RLDisaggregatedTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train_resources: AcceleratorResourcesConfig
    rollout_resources: AcceleratorResourcesConfig
    train_worker_cfg: WorkerConfig
    rollout_config: RolloutConfig
    judger_config: JudgerConfig
    tokenizer_path: Union[str, Path]
    replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig = SyncReplayBufferConfig()
    agent_loop_manager_cfg: AgentLoopManagerConfig
    eval_agent_loop_manager_cfg: AgentLoopManagerConfig
    evaluator_config: EvaluatorConfig
    load_from: Union[str, Path]
    train_batch_size: int
    total_train_steps: int
    trigger_parameter_sync_step: int = 1

    enable_evaluate: bool = True
    enable_initial_evaluate: bool = False
    evaluate_step: int = 1
    work_dir: Union[Path, str, None] = None
    auto_resume: bool = False
    load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig()
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    hf_interval: int | None = -1
    hf_max_keep: int | None = -1
    checkpoint_no_save_optimizer: bool = False
    log_dir: Union[Path, str, None] = None
    seed: int = 66
    debug_rollout: bool = False
    skip_checkpoint_validation: bool = False
    exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard"

    def model_post_init(self, __context) -> None:
        if self.train_batch_size <= 0:
            raise ValueError(f"train_batch_size must be positive, got {self.train_batch_size}")
        if self.total_train_steps <= 0:
            raise ValueError(f"total_train_steps must be positive, got {self.total_train_steps}")
        if self.trigger_parameter_sync_step <= 0:
            raise ValueError(
                "trigger_parameter_sync_step must be positive, "
                f"got {self.trigger_parameter_sync_step}"
            )

    def build(self) -> "RLDisaggregatedTrainer":
        return RLDisaggregatedTrainer(
            train_resources=self.train_resources,
            rollout_resources=self.rollout_resources,
            train_worker_cfg=self.train_worker_cfg,
            rollout_config=self.rollout_config,
            judger_config=self.judger_config,
            tokenizer_path=self.tokenizer_path,
            replay_buffer_config=self.replay_buffer_config,
            agent_loop_manager_cfg=self.agent_loop_manager_cfg,
            eval_agent_loop_manager_cfg=self.eval_agent_loop_manager_cfg,
            evaluator_config=self.evaluator_config,
            enable_evaluate=self.enable_evaluate,
            enable_initial_evaluate=self.enable_initial_evaluate,
            evaluate_step=self.evaluate_step,
            work_dir=self.work_dir,
            auto_resume=self.auto_resume,
            load_checkpoint_cfg=self.load_checkpoint_cfg,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_maxkeep=self.checkpoint_maxkeep,
            checkpoint_no_save_optimizer=self.checkpoint_no_save_optimizer,
            hf_interval=self.hf_interval,
            hf_max_keep=self.hf_max_keep,
            load_from=self.load_from,
            log_dir=self.log_dir,
            seed=self.seed,
            debug_rollout=self.debug_rollout,
            skip_checkpoint_validation=self.skip_checkpoint_validation,
            train_batch_size=self.train_batch_size,
            total_train_steps=self.total_train_steps,
            trigger_parameter_sync_step=self.trigger_parameter_sync_step,
            exp_tracker=self.exp_tracker,
        )


class RLDisaggregatedTrainer(RLColocateTrainer):
    _META_PATH = ".xtuner_rl_disaggregated_trainer"

    def __init__(
        self,
        *,
        train_resources: AcceleratorResourcesConfig,
        rollout_resources: AcceleratorResourcesConfig,
        train_worker_cfg: WorkerConfig,
        rollout_config: RolloutConfig,
        judger_config: JudgerConfig,
        tokenizer_path: str | Path,
        replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig,
        agent_loop_manager_cfg: AgentLoopManagerConfig,
        eval_agent_loop_manager_cfg: AgentLoopManagerConfig,
        evaluator_config: EvaluatorConfig,
        enable_evaluate: bool = True,
        enable_initial_evaluate: bool = False,
        evaluate_step: int = 1,
        work_dir: Path | str | None = None,
        auto_resume: bool = False,
        load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig(),
        checkpoint_interval: int | None = -1,
        checkpoint_maxkeep: int | None = -1,
        checkpoint_no_save_optimizer: bool = False,
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        load_from: str | Path,
        log_dir: Path | str | None = None,
        seed: int = 66,
        debug_rollout: bool = False,
        skip_checkpoint_validation: bool = False,
        train_batch_size: int,
        total_train_steps: int,
        trigger_parameter_sync_step: int = 1,
        exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard",
    ):
        check_fa3()

        work_dir = Path(work_dir) if work_dir else Path.cwd() / "work_dirs"
        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
        self._meta = XTunerMeta.build(work_dir, self._META_PATH, auto_resume)

        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from
        is_hf_path, error_info = is_hf_model_path(load_from) if load_from is not None else (False, "")
        self._load_from_hf = is_hf_path
        if not self._load_from_hf:
            raise NotImplementedError(error_info)
        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_maxkeep = checkpoint_maxkeep
        self._checkpoint_no_save_optimizer = checkpoint_no_save_optimizer
        self._load_checkpoint_cfg = self._resolve_load_checkpoint_cfg(auto_resume, load_checkpoint_cfg)

        log_dir = self.exp_dir / "logs"
        self.logger = self._build_logger(log_dir)

        # force_set_tokenize_workers(self.logger)
        self.logger.warning(
            "Disaggregated weight sync is currently mocked. The real cross-device weight update path is expected "
            "to be provided by the follow-up weight sync module."
        )

        if skip_checkpoint_validation:
            patch_default_save_plan()

        # 这里的 total_train_steps 指的是训练步，不是 rollout launch 次数。
        self._total_train_steps = total_train_steps
        # 共享的 RLColocateTrainer 日志/保存逻辑仍然读取 _rollout_steps。
        # 对 disaggregated 来说两者语义都是“总训练步数”，这里保留一个兼容别名，
        # 避免在共享逻辑里继续散落一批 if hasattr(...) 分支。
        self._rollout_steps = self._total_train_steps
        self._cur_step = 0
        self._global_train_step = 0
        self._seed = seed
        set_deterministic()
        set_random_seed(seed)
        self.train_batch_size = train_batch_size
        self._trigger_parameter_sync_step = trigger_parameter_sync_step

        pg_name_prefix = f"disaggregated_{self.exp_dir.name}"
        # 训练和 rollout 明确分成两套 placement group，
        # 这是 disaggregated 和 colocated 在资源层面最本质的区别。
        self._train_pg = AutoAcceleratorWorkers.build_placement_group(
            train_resources, name=f"{pg_name_prefix}_train"
        )
        self._rollout_pg = AutoAcceleratorWorkers.build_placement_group(
            rollout_resources, name=f"{pg_name_prefix}_rollout"
        )

        if train_worker_cfg.seed is None:
            self.logger.warning(f"RLDisaggregatedTrainer seed {seed} is used as train worker seed.")
            train_worker_cfg.seed = seed
        train_worker_cfg.load_from = load_from
        train_worker_cfg.log_dir = log_dir
        train_worker_cfg.use_rollout_logprobs_as_old_logprobs = True
        self._train_worker_cfg = train_worker_cfg

        rollout_config.worker_log_dir = log_dir

        self.train_controller = train_worker_cfg.build(self._train_pg)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            ray.get(self.train_controller.resume.remote(self._load_checkpoint_cfg))
            train_state_path = Path(self._load_checkpoint_cfg.checkpoint_path) / self._SAVE_TRAIN_STATE_PATH
            with train_state_path.open("r") as f:
                train_state = json.load(f)
                self._cur_step = train_state["cur_step"]
            self.logger.warning(
                "Disaggregated weight sync is currently mocked. Rollout workers load weights from load_from after "
                "resume and are not updated from the resumed trainer checkpoint in this implementation."
            )

        self.rollout_controller = rollout_config.build(self._rollout_pg)

        judger = judger_config.build()
        replay_buffer = replay_buffer_config.build()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.agent_loop_manager = agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            judger=judger,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
        )
        self.agent_loop_manager.set_sync_interval_context(
            trigger_parameter_sync_step=self._trigger_parameter_sync_step,
            total_train_steps=self._total_train_steps,
        )

        self.eval_agent_loop_manager = eval_agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            judger=judger,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
        )

        self._enable_evaluate = enable_evaluate
        self._enable_initial_evaluate = enable_initial_evaluate
        self._evaluate_step = evaluate_step

        total_eval_samples = len(self.eval_agent_loop_manager.data_sampler)
        self.evaluator = evaluator_config.build(total_eval_samples=total_eval_samples)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            self.logger.info(f"Resume sampler from {self._load_checkpoint_cfg.checkpoint_path}")
            self.agent_loop_manager.resume(self._load_checkpoint_cfg.checkpoint_path)

        if debug_rollout:
            self.logger.warning("Debug rollout mode is enabled, training and weight sync will be skipped.")
        self._debug_rollout = debug_rollout
        self._exp_tracker = get_writer(writer_type=exp_tracker, log_dir=log_dir / self._EXP_TRACKING_PATH)
        self._display_all_workers_log = False

    def _build_logger(self, log_dir: Path):
        from xtuner.v1.utils import get_logger

        return get_logger(log_dir=log_dir, tag="RLDisaggregatedTrainer")

    def fit(self):
        asyncio_run(self._fit_async())

    async def _fit_async(self):
        self.logger.info("Start disaggregated RL training")
        if self._cur_step >= self._total_train_steps:
            self.logger.info(f"Train steps {self._total_train_steps} reached, stop training")
            return

        if self._enable_initial_evaluate and not self._debug_rollout:
            eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
                self.evaluator.eval_batch_size, rollout_step=0
            )
            eval_metrics = self.evaluator.run(eval_produce_result.rollout_states)
            self.logger.info(f"Initial rollout evaluate scores {eval_metrics} and start training")
            self._exp_tracker.add_scalars(
                tag_scalar_dict={f"eval/{k}": v for k, v in eval_metrics.items()},
                global_step=0,
            )

        for train_step in range(self._cur_step + 1, self._total_train_steps + 1):
            self.logger.info(f"Train step {train_step}/{self._total_train_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                if self._should_start_fullasync_interval(train_step):
                    with timer("schedule_rollout", step_timer_dict):
                        await self.agent_loop_manager.fullasync_produce_batch(
                            self.train_batch_size,
                            rollout_step=train_step,
                        )
                with timer("wait_rollout", step_timer_dict):
                    produce_result = await self.agent_loop_manager.get_completed_batch(
                        self.train_batch_size,
                        rollout_step=train_step,
                    )
                train_batch = produce_result.rollout_states
                self._refresh_train_batch_staleness(train_batch, train_step)
                if train_batch:
                    self.logger.info(f"Get {len(train_batch) * len(train_batch[0])} samples for training")
                self._save_rollout_batch_trajectories(
                    train_batch, train_step, stage="train", step_label="Train step"
                )

                if not self._debug_rollout:
                    data_batches, data_info = self._prepare_train_batch(train_batch, step_timer_dict)

                    with timer("training", step_timer_dict):
                        # Ray ObjectRef 在当前版本里可以直接 await，
                        # 这里不需要再用 ray.get + to_thread 桥接。
                        workers_log_item = await self.train_controller.fit.remote(
                            data_batches,
                            pack_max_length=self._train_worker_cfg.pack_max_length,
                            rollout_idx=train_step,
                        )
                    train_log_info: TrainInfo = {
                        "data_info": data_info,
                        "workers_log_item": workers_log_item,
                    }
                    if self._should_sync_after_step(train_step):
                        with timer("sync_and_save", step_timer_dict):
                            self._sync_weights_and_save(train_step, step_timer_dict)

                    eval_log_info = await self._evaluate_step_async(
                        train_step, step_timer_dict, step_label="Train step"
                    )
                else:
                    train_log_info = {}
                    eval_log_info = {}

            self._log_step(train_step, step_timer_dict, produce_result, train_log_info, eval_log_info)
            self._cur_step = train_step

    def _should_sync_after_step(self, train_step: int) -> bool:
        return train_step % self._trigger_parameter_sync_step == 0 or train_step == self._total_train_steps

    def _should_start_fullasync_interval(self, train_step: int) -> bool:
        return (train_step - 1) % self._trigger_parameter_sync_step == 0

    def _refresh_train_batch_staleness(self, train_batch: list[list[RolloutState]], train_step: int) -> None:
        # 训练前再刷新一次，是为了让真正进入训练的 batch 带上“本训练步视角”的 staleness。
        for group in train_batch:
            for item in group:
                refresh_seq_staleness(item, train_step)

    def _sync_weights_and_save(self, rollout_idx: int, step_timer_dict: dict):
        with timer("save_ckpt", step_timer_dict):
            self._maybe_save_checkpoint(rollout_idx)
            self._maybe_save_hf(rollout_idx)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        with timer("sync_weight", step_timer_dict):
            self._mock_disaggregated_weight_sync(rollout_idx)

    def _mock_disaggregated_weight_sync(self, rollout_idx: int) -> None:
        self.logger.info(
            f"Mock disaggregated weight sync for rollout_idx={rollout_idx}. The real cross-device weight update path "
            "is expected to be provided by the follow-up weight sync module."
        )
