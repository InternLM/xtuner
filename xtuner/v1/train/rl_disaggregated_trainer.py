import json
from pathlib import Path
from shutil import rmtree
from typing import cast

import ray
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, model_validator

from transformers import AutoTokenizer
from xtuner.v1._writer import get_writer
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.rl.agent_loop import (
    AgentLoopManagerConfig,
    ProduceBatchStatus,
)
from xtuner.v1.rl.agent_loop.agent_loop_manager import AgentLoopManagerStatus
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer.worker import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run, create_task
from xtuner.v1.train.rl_colocate_trainer import (
    TRAINER_RAY_GET_TIMEOUT,
    RLColocateTrainer,
    bind_train_rollout,
    check_fa3,
    force_set_tokenize_workers,
)
from xtuner.v1.train.trainer import LoadCheckpointConfig, XTunerMeta
from xtuner.v1.utils import get_logger, is_hf_model_path, set_deterministic, timer


def _validate_disagg_sync_schedule(
    sync_weights_interval: int,
    checkpoint_interval: int | None,
    hf_interval: int | None,
) -> None:
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")

    for name, interval in (
        ("checkpoint_interval", checkpoint_interval),
        ("hf_interval", hf_interval),
    ):
        if interval is None or interval == -1:
            continue
        if interval <= 0:
            raise ValueError(f"{name} must be positive or -1/None to disable it, got {interval}.")
        if interval % sync_weights_interval != 0:
            raise ValueError(
                f"{name}={interval} must be a multiple of sync_weights_interval={sync_weights_interval}, "
                "because disaggregated checkpoint/HF saves only run on sync steps."
            )


class RLDisaggregatedTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train_resources: AcceleratorResourcesConfig
    rollout_resources: AcceleratorResourcesConfig
    train_worker_cfg: WorkerConfig
    rollout_config: RolloutConfig
    tokenizer_path: str | Path
    replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig = SyncReplayBufferConfig()
    agent_loop_manager_cfg: AgentLoopManagerConfig
    eval_agent_loop_manager_cfg: AgentLoopManagerConfig
    evaluator_config: EvaluatorConfig
    load_from: str | Path
    total_train_steps: int
    train_batch_size: int
    sync_weights_interval: int = 1

    enable_evaluate: bool = True
    enable_initial_evaluate: bool = False
    evaluate_step: int = 1
    work_dir: Path | str | None = None
    auto_resume: bool = False
    load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig()
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    checkpoint_no_save_optimizer: bool = False
    hf_interval: int | None = -1
    hf_max_keep: int | None = -1
    log_dir: Path | str | None = None
    seed: int = 66
    debug_rollout: bool = False
    skip_checkpoint_validation: bool = False
    exp_tracker: str = "tensorboard"

    @model_validator(mode="after")
    def _validate_sync_intervals(self):
        _validate_disagg_sync_schedule(
            sync_weights_interval=self.sync_weights_interval,
            checkpoint_interval=self.checkpoint_interval,
            hf_interval=self.hf_interval,
        )
        return self

    def build(self) -> "RLDisaggregatedTrainer":
        return RLDisaggregatedTrainer(
            train_resources=self.train_resources,
            rollout_resources=self.rollout_resources,
            train_worker_cfg=self.train_worker_cfg,
            rollout_config=self.rollout_config,
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
            total_train_steps=self.total_train_steps,
            train_batch_size=self.train_batch_size,
            sync_weights_interval=self.sync_weights_interval,
            exp_tracker=cast(str, self.exp_tracker),
        )


class RLDisaggregatedTrainer(RLColocateTrainer):
    _META_PATH = ".xtuner_rl_disaggregated_trainer"

    def _build_disaggregated_placement_groups(
        self,
        train_resources: AcceleratorResourcesConfig,
        rollout_resources: AcceleratorResourcesConfig,
    ):
        pg_name_prefix = f"xtuner_rl_disagg_{self.exp_dir.name}"
        train_pg_name = f"{pg_name_prefix}_train"
        rollout_pg_name = f"{pg_name_prefix}_rollout"

        train_pg = AutoAcceleratorWorkers.build_placement_group(train_resources, name=train_pg_name)
        rollout_pg = AutoAcceleratorWorkers.build_placement_group(rollout_resources, name=rollout_pg_name)
        if train_pg.id == rollout_pg.id:
            raise RuntimeError(
                "RLDisaggregatedTrainer requires distinct placement groups for train and rollout, "
                f"but both resolved to the same placement group id={train_pg.id}. "
                "Please check placement-group naming and stale Ray cluster state."
            )

        self.logger.info(
            "Created disaggregated placement groups: "
            f"train={train_pg_name}(id={train_pg.id}), "
            f"rollout={rollout_pg_name}(id={rollout_pg.id})"
        )
        return train_pg, rollout_pg

    def __init__(
        self,
        *,
        train_resources: AcceleratorResourcesConfig,
        rollout_resources: AcceleratorResourcesConfig,
        train_worker_cfg: WorkerConfig,
        rollout_config: RolloutConfig,
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
        hf_interval: int | None = -1,
        hf_max_keep: int | None = -1,
        load_from: str | Path,
        log_dir: Path | str | None = None,
        seed: int = 66,
        debug_rollout: bool = False,
        skip_checkpoint_validation: bool = False,
        total_train_steps: int,
        train_batch_size: int,
        sync_weights_interval: int,
        exp_tracker: str = "tensorboard",
    ):
        # 设计目标：
        # - 复用 colocate trainer 现有的大部分基础设施，比如 meta/work_dir、
        #   train controller、rollout controller、日志和 checkpoint 目录约定；
        # - 只把“训练主循环”改造成非共卡语义：
        #   后台持续生产，前台按需消费，到同步点显式暂停 producer。
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
        _validate_disagg_sync_schedule(
            sync_weights_interval=sync_weights_interval,
            checkpoint_interval=checkpoint_interval,
            hf_interval=hf_interval,
        )
        self._load_checkpoint_cfg = self._resolve_load_checkpoint_cfg(auto_resume, load_checkpoint_cfg)

        log_dir = self.exp_dir / "logs"
        self.logger = get_logger(log_dir=log_dir, tag="RLDisaggTrainer")
        force_set_tokenize_workers(self.logger)
        if skip_checkpoint_validation:
            patch_default_save_plan()

        self._rollout_steps = total_train_steps
        self._cur_step = 0
        self._global_train_step = 0
        self._seed = seed
        self.train_batch_size = train_batch_size
        self._sync_weights_interval = sync_weights_interval
        set_deterministic()
        set_random_seed(seed)

        # 非共卡的核心前提是 train 和 rollout 不再共用同一套 placement group。
        # 这样 rollout 可以在后台持续生成，而 train 侧前台做自己的优化步骤。
        self._train_pg, self._rollout_pg = self._build_disaggregated_placement_groups(
            train_resources=train_resources,
            rollout_resources=rollout_resources,
        )

        if train_worker_cfg.seed is None:
            self.logger.warning(f"RLTrainer seed {seed} is used as train worker seed.")
            train_worker_cfg.seed = seed
        train_worker_cfg.load_from = load_from
        train_worker_cfg.log_dir = log_dir
        self._train_worker_cfg = train_worker_cfg

        rollout_config.worker_log_dir = log_dir
        if self._load_checkpoint_cfg.checkpoint_path is not None:
            rollout_config.skip_load_weights = True
            self.logger.info(
                f"Skip load rollout weights due to resume from checkpoint {self._load_checkpoint_cfg.checkpoint_path}"
            )

        self.train_controller = train_worker_cfg.build(self._train_pg)
        self.rollout_controller = rollout_config.build(self._rollout_pg)

        replay_buffer = replay_buffer_config.build()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.agent_loop_manager = agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
        )
        self.eval_agent_loop_manager = eval_agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
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
            self._resume_from_checkpoint(self._load_checkpoint_cfg.checkpoint_path)

        if debug_rollout:
            self.logger.warning(
                "Debug rollout mode is enabled. Disaggregated training keeps rollout workers resident."
            )
        self._debug_rollout = debug_rollout
        self._exp_tracker = get_writer(writer_type=exp_tracker, log_dir=log_dir / self._EXP_TRACKING_PATH)
        self._display_all_workers_log = False

    def _resume_from_checkpoint(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)
        ray.get(self.train_controller.resume.remote(self._load_checkpoint_cfg))

        train_state_path = checkpoint_path / self._SAVE_TRAIN_STATE_PATH
        with train_state_path.open("r") as f:
            train_state = json.load(f)
        self._cur_step = train_state["cur_step"]

        self.logger.info(f"Resume sampler from {checkpoint_path}")
        saved_model_rollout_step = self.agent_loop_manager.resume(checkpoint_path)

        bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
        self.logger.info("Rollout workers skip load weights, update weights from train workers.")
        self.fake_update_weights()
        self.agent_loop_manager.continue_product(model_rollout_step=saved_model_rollout_step)

    def fit(self):
        # 对外仍保留同步接口，和现有 CLI / config 调用方式保持一致。
        return asyncio_run(self._fit())

    async def _fit(self):
        self.logger.info("Start RL disaggregated training")
        if self._cur_step >= self._rollout_steps:
            self.logger.info(f"Rollout steps {self._rollout_steps} reached, stop training")
            return

        if self._enable_initial_evaluate:
            # 初始评测仍然走同步 evaluate 逻辑，避免一开始就和 producer 竞争 rollout 资源。
            eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
                self.evaluator.eval_batch_size, rollout_step=0
            )
            eval_metrics = self.evaluator.run(eval_produce_result.rollout_states)
            tb_scores = {f"eval/{k}": v for k, v in eval_metrics.items()}
            self._exp_tracker.add_scalars(tag_scalar_dict=tb_scores, global_step=0)

        # 后台 producer:
        # - 持续调用 agent_loop_manager.produce_loop()
        # - 只负责往 replay buffer 里填数据
        # - 不直接把 batch 返回给 trainer
        producer_task = create_task(
            self.agent_loop_manager.produce_loop(
                batch_size=self.train_batch_size,
                start_rollout_step=self._cur_step,  # rename: latest_consumer_step
            )
        )
        try:
            for rollout_idx in range(self._cur_step + 1, self._rollout_steps + 1):
                self.logger.info(f"Train step {rollout_idx}/{self._rollout_steps} start")
                step_timer_dict: dict[str, float] = {}
                train_log_info = {}
                eval_log_info = {}
                with timer("step", step_timer_dict):
                    # 前台 trainer:
                    # - 只通过 get_batch() 从 replay buffer 取数据
                    # - 不再像 colocate 那样主动触发一整轮 rollout
                    produce_result = await self.agent_loop_manager.get_batch(
                        self.train_batch_size, rollout_step=rollout_idx
                    )
                    need_sync = (
                        produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
                        or rollout_idx % self._sync_weights_interval == 0
                        or rollout_idx == self._rollout_steps
                    )
                    if produce_result.status != ProduceBatchStatus.EXPIRED_BATCH:
                        # 正常路径：拿到有效 batch 后，训练侧只做 prepare + fit。
                        # 这里仍然复用 colocate trainer 的数据整理逻辑，减少重复代码。
                        train_batch = produce_result.rollout_states
                        assert train_batch, (
                            "RLDisaggregatedTrainer expects get_batch() to return non-empty rollout_states "
                            "unless status is EXPIRED_BATCH."
                        )
                        train_trajectory_dir = self.exp_dir / "train_rollout"
                        train_trajectory_dir.mkdir(parents=True, exist_ok=True)
                        train_trajectory_path = train_trajectory_dir / f"train_rollout_{rollout_idx}.jsonl"
                        self._save_trajectories(train_batch, train_trajectory_path)
                        with timer("prepare_data", step_timer_dict):
                            data_batches, data_info = self._prepare_train_data(
                                train_batch, self._train_worker_cfg.pack_max_length
                            )
                        with timer("training", step_timer_dict):
                            workers_log_item = ray.get(
                                self.train_controller.fit.remote(
                                    data_batches,
                                    pack_max_length=self._train_worker_cfg.pack_max_length,
                                    rollout_idx=rollout_idx,
                                )
                            )
                        train_log_info = {
                            "data_info": data_info,
                            "workers_log_item": workers_log_item,
                        }
                    else:
                        # EXPIRED_BATCH 的设计动机：
                        # - 不是“这一轮刚好没数据”
                        # - 而是 rollout 侧当前使用的模型权重已经过旧
                        # 所以这里刻意跳过训练，优先推进同步，
                        # 让 rollout 尽快切到更新后的权重继续工作。
                        self.logger.info(
                            "Skip train step because rollout model is expired; prioritize weight sync first."
                        )

                    if need_sync:
                        # 到权重同步点时，先显式打断 producer。
                        # 这样可以保证：
                        # - 不再继续补发新的 rollout
                        # - pending rollout 都在同步前被收尾
                        # - 后面的 save / sync 发生在相对静止的状态
                        with timer("pause_product", step_timer_dict):
                            await self.agent_loop_manager.pause_product(for_weight_update=True)

                        await self._sync_weights_and_save(rollout_idx, step_timer_dict)

                        if self._enable_evaluate and rollout_idx % self._evaluate_step == 0:
                            # 这里刻意把 eval 放在恢复 producer 前面。
                            # 设计上希望 eval 优先于 background producer，
                            # 避免 continue_product 后 producer 立刻恢复生成，与 eval 竞争 rollout 资源。
                            with timer("evaluation", step_timer_dict):
                                eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
                                    self.evaluator.eval_batch_size, rollout_step=rollout_idx
                                )
                                eval_batch = eval_produce_result.rollout_states
                                eval_metrics = self.evaluator.run(eval_batch)
                                eval_trajectory_dir = self.exp_dir / "eval_rollout"
                                eval_trajectory_dir.mkdir(parents=True, exist_ok=True)
                                eval_trajectory_path = eval_trajectory_dir / f"eval_rollout_{rollout_idx}.jsonl"
                                self._save_trajectories(eval_batch, eval_trajectory_path)
                                eval_log_info.update(eval_metrics)

                        # pause_product(for_weight_update=True) 和 continue_product() 是一对：
                        # - 前者让 producer 停下
                        # - 后者在同步和评测结束后恢复 producer
                        self.agent_loop_manager.continue_product(model_rollout_step=rollout_idx)

                self._log_step(rollout_idx, step_timer_dict, produce_result, train_log_info, eval_log_info)
                self._cur_step = rollout_idx
        finally:
            # 训练退出时必须显式通知后台 producer 正常收尾，
            # 避免遗留悬空协程继续占着事件循环。
            self.agent_loop_manager._status = AgentLoopManagerStatus.FINISH
            self.agent_loop_manager._finish_event.set()
            await producer_task

    async def _sync_weights_and_save(self, rollout_idx: int, step_timer_dict: dict):
        # 非共卡下，这里只负责“静止态”之后的保存和同步动作；
        # producer 的停止动作已经在 _fit() 里提前完成，
        # 这样调用顺序更直观：cleanup -> save -> bind -> fake_update_weights。
        with timer("save_ckpt", step_timer_dict):
            self._maybe_save_checkpoint(rollout_idx)
            self._maybe_save_hf(rollout_idx)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        with timer("sync_weight", step_timer_dict):
            bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
            self.fake_update_weights()

    def fake_update_weights(self):
        # 这里保留 fake 接口，是为了把“trainer 主流程”和“真实跨卡同步实现”解耦。
        # 当前分支先复用现有 controller 的 update_weights 链路把流程打通；
        # 后续如果接入真正的 disaggregated 权重同步模块，只需要替换这个函数即可。
        ray.get(self.train_controller.update_weights.remote(), timeout=TRAINER_RAY_GET_TIMEOUT)
        self.logger.info("Rollout workers updated weights through fake disaggregated sync.")

    def _maybe_save_checkpoint(self, cur_step: int) -> None:
        ckp_interval = self._checkpoint_interval
        if ckp_interval is None or ckp_interval == -1:
            return
        if cur_step % ckp_interval != 0:
            return

        checkpoint_path = self.exp_dir / self._CHECKPOINT_DIR / f"ckpt-step-{cur_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving sampler state to {checkpoint_path}")
        self.agent_loop_manager.save(checkpoint_path, model_rollout_step_override=cur_step)

        self.logger.info(f"Saving DCP checkpoint to {checkpoint_path}")
        ray.get(self.train_controller.save.remote(str(checkpoint_path), self._checkpoint_no_save_optimizer))

        train_state_path = checkpoint_path / self._SAVE_TRAIN_STATE_PATH
        with train_state_path.open("w") as f:
            json.dump({"cur_step": cur_step}, f)

        current_exp = self._meta.latest_exp
        current_exp.checkpoint_list.append(str(checkpoint_path))

        ckp_maxkeep = self._checkpoint_maxkeep
        ckp_list = current_exp.checkpoint_list
        if ckp_maxkeep is not None and ckp_maxkeep > 0 and len(ckp_list) > ckp_maxkeep:
            for deleted in ckp_list[:-ckp_maxkeep]:
                if Path(deleted).exists():
                    rmtree(deleted, ignore_errors=True)
            current_exp.checkpoint_list = ckp_list[-ckp_maxkeep:]

        meta_path = self.exp_dir.parent / self._META_PATH
        with meta_path.open("w") as f:
            f.write(self._meta.model_dump_json(indent=2))
