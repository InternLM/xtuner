import json
import os
import random
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import cast

import numpy as np
import psutil
import ray
import torch
from mmengine import load
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator
from ray.actor import ActorClass
from typing_extensions import Self

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import check_valid_dataflow_item
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.evaluator import Evaluator, EvaluatorConfig
from xtuner.v1.ray.judger import JudgerConfig
from xtuner.v1.rl.base import TrainingController, WorkerConfig
from xtuner.v1.rl.base import TrainingWorker as BaseTrainingWorker
from xtuner.v1.train import ResumeConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger, is_hf_model_path, record_git_info, timer, timer_logger
from xtuner.v1.utils.device import get_device, get_torch_device_module
from xtuner.v1.utils.env_check import get_rollout_engine_version

from .trainer import ExpHistory, ExpInfo, GitInfo, XTunerMeta


# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def bind_train_rollout(
    train_controller,
    env_controller,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return


class RLTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    load_from: str | Path
    resources: AcceleratorResourcesConfig
    rollout_config: RolloutConfig
    dataflow_config: DataFlowConfig
    judger_config: JudgerConfig
    replay_buffer_config: ReplayBufferConfig
    train_worker_config: WorkerConfig
    evaluator_config: EvaluatorConfig | None = None
    tokenizer_path: str | Path
    work_dir: Path | str | None = None
    log_dir: Path | str | None = None
    total_epochs: int
    resume_config: ResumeConfig | None = None
    strict_load: bool = True
    hf_interval: int | None = None
    hf_max_keep: int | None = None
    seed: int = 42
    debug: bool = False

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self

    @field_serializer("replay_buffer_config")
    def serialize_replay_buffer_cfg(self, replay_buffer_config: ReplayBufferConfig) -> str:
        return replay_buffer_config.model_dump(include={"replay_ratio", "replay_weights"})

    @field_serializer("evaluator_config")
    def serialize_evaluator_cfg(self, evaluator_config: EvaluatorConfig) -> str:
        if evaluator_config:
            return evaluator_config.model_dump(exclude={"tokenizer", "dataset_cfg", "compute_metric_func"})
        else:
            return ""

    @field_serializer("judger_config")
    def serialize_judger_config(self, judger_config: JudgerConfig) -> str:
        return judger_config.model_dump(exclude={"tokenizer", "reward_func"})


def get_train_seq_ctx(
    input_ids: torch.LongTensor, multimodal_train_info: dict | None = None, len_response_ids: int = 0
):
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    if multimodal_train_info and len(multimodal_train_info) > 0:
        position_ids = multimodal_train_info.get("position_ids")  # (1,n) or (3,1,n)
        if position_ids is not None and len(position_ids.shape) == 3:
            # qwen3vl 需要特殊处理，其余的不需要额外处理
            max_value = position_ids.max(dim=-1).values  # (3,1)
            response_position_ids = max_value.unsqueeze(-1).expand(-1, -1, len_response_ids) + torch.arange(
                1, len_response_ids + 1, device=max_value.device
            )
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            seq_ctx.position_ids = position_ids  # type: ignore[assignment]
            assert position_ids.size(-1) == input_ids.size(-1)
        seq_ctx.pixel_values = multimodal_train_info.get("pixel_values")
        seq_ctx.image_grid_thw = multimodal_train_info.get("image_grid_thw")
    return seq_ctx


class RLTrainer:
    """Universal Reinforcement Learning Trainer for XTuner.

    A flexible RL training orchestrator that supports multiple RL algorithms
    through pluggable training workers and controllers. Manages the complete
    RL training workflow including rollout generation, policy updates,
    evaluation, and checkpoint management.

    **Training Workflow:**
        1. Initialize distributed workers and rollout environment
        2. Generate experiences using current policy
        3. Update policy using algorithm-specific training logic
        4. Synchronize weights between training and rollout workers
        5. Evaluate model performance and save checkpoints

    Args:
        load_from (str | Path): Path to the base model to load. Should be a HuggingFace
            model path (e.g., "meta-llama/Llama-2-7b-hf") or local model directory.
        resources (AcceleratorResourcesConfig): Configuration for distributed computing
            resources including number of workers, GPU allocation, and placement groups.
        rollout_config (RolloutConfig): Configuration for rollout workers that generate
            experiences by interacting with the environment.
        dataflow_config (DataFlowConfig): Data orchestration configuration controlling
            experience collection, batch formation, and data distribution across workers.
        judger_config (JudgerConfig): Configuration for the reward model or scoring system
            that evaluates generated responses and provides training signals.
        replay_buffer_config (ReplayBufferConfig): Settings for experience replay buffer
            including capacity, sampling strategy, and data retention policies.
        evaluator_config (EvaluatorConfig | None): Evaluation configuration specifying metrics,
            evaluation datasets, and assessment frequency for monitoring training progress. Defaults to None.
        train_worker_cfg (WorkerConfig): Configuration for distributed training workers
            including model architecture, optimizer settings, loss functions, and parallelism.
        tokenizer_path (str | Path): Path to the tokenizer for text preprocessing.
            Should be compatible with the base model specified in load_from.
        work_dir (Path | str | None): Working directory for experiment outputs,
            checkpoints, and logs. Defaults to None.
        log_dir (Path | str | None): Directory for training logs and monitoring outputs.
            Defaults to None.
        total_epochs (int): Total number of training epochs to execute.
        enable_evaluate (bool): Whether to perform periodic evaluation during training.
        resume_config (ResumeConfig | None): Configuration for resuming training from
            a previous checkpoint. Defaults to None.
        strict_load (bool): Whether to strictly enforce checkpoint loading compatibility.
            Defaults to True.
        hf_interval (int | None): Interval (in epochs) for saving HuggingFace format
            checkpoints. Defaults to None.
        hf_max_keep (int | None): Maximum number of HuggingFace checkpoints to retain.
            Defaults to None.
        seed (int): Random seed for reproducible training. Defaults to 42.
        debug (bool): Enable debug mode with additional logging. Defaults to False.

    **Examples:**

    Example configuration for GRPO RL training setup::

        trainer = RLTrainer(
            load_from="Qwen3-8B",
            resources=resources_config,
            rollout_config=rollout_cfg,
            dataflow_config=dataflow_cfg,
            judger_config=judger_cfg,
            replay_buffer_config=buffer_cfg,
            evaluator_config=eval_cfg,
            train_worker_cfg=worker_cfg,
            tokenizer_path="Qwen3-8B",
            total_epochs=10,
            enable_evaluate=True
        )
        trainer.fit()
    """

    META_PATH = ".xtuner_grpo"

    def __init__(
        self,
        *,
        load_from: str | Path,  # Huggingface model path or saved trainer_path
        resources: AcceleratorResourcesConfig,
        rollout_config: RolloutConfig,
        dataflow_config: DataFlowConfig,
        judger_config: JudgerConfig,
        replay_buffer_config: ReplayBufferConfig,
        train_worker_cfg: WorkerConfig,
        evaluator_config: EvaluatorConfig | None = None,
        tokenizer_path: str | Path,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        total_epochs: int,
        resume_config: ResumeConfig | None = None,
        strict_load: bool = True,
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        seed: int = 42,
        debug: bool = False,
        trainer_cfg: RLTrainerConfig | None = None,
    ):
        """Initialize the RL training system."""
        if os.environ.get("XTUNER_USE_FA3", "0") == "1":
            try:
                from xtuner.v1.ops.flash_attn import get_flash_attn_varlen

                get_flash_attn_varlen()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Flash attention v3 runtime error {e}, Please install it first or set XTUNER_USE_FA3=0."
                )
        train_worker_cfg.load_from = load_from

        self._total_epochs = total_epochs
        self._cur_step = 0

        self._rl_trainer_cfg = trainer_cfg
        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from
        self._load_from_hf = load_from is not None and is_hf_model_path(load_from)
        if not self._load_from_hf:
            raise NotImplementedError

        if not self._load_from_hf:
            assert hf_interval is None and hf_max_keep is None, (
                "`hf_interval` and `hf_max_keep` should be None when `load_from` is not a Huggingface model path, "
            )

        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self._debug = debug
        self._seed = seed
        self._set_deterministic()
        self._set_random_seed(seed)

        if work_dir is None:
            work_dir = Path.cwd() / "work_dir"

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        self._work_dir = work_dir
        self._meta = self._init_xtuner_meta(work_dir, resume_config is not None)

        if log_dir is None:
            log_dir = self.exp_dir
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.logger = self._init_logger(log_dir)
        train_worker_cfg.log_dir = log_dir
        dataflow_config.worker_log_dir = log_dir
        rollout_config.worker_log_dir = log_dir
        self._enable_evaluate = False
        self._enable_initial_evaluate = False
        if evaluator_config:
            evaluator_config.worker_log_dir = log_dir
            self._enable_evaluate = evaluator_config.enable_evaluate
            self._enable_initial_evaluate = evaluator_config.enable_initial_evaluate
        self._pg = AutoAcceleratorWorkers.build_placement_group(resources)
        # We need to build train controller first, and then build rollout dataflow to make
        # inference engines know how much memory they can utilize.
        self._train_controller = self._build_train_controller(train_worker_cfg)

        self._rollout_env_controller, self._rollout_dataflow = self._build_rollout_dataflow(
            dataflow_cfg=dataflow_config,
            rollout_cfg=rollout_config,
            judger_cfg=judger_config,
            replay_buffer_config=replay_buffer_config,
        )
        self._dataflow_partial_rollout_step = dataflow_config.partial_rollout_step
        if self._enable_evaluate and evaluator_config:
            self._evaluator = Evaluator.remote(evaluator_config, self._rollout_env_controller)  # type: ignore[attr-defined]
            self._eval_step = evaluator_config.evaluate_step
        else:
            pass

        self._global_batch_size = dataflow_config.global_batch_size
        self._rollout_steps = (
            ray.get(self._rollout_dataflow.get_train_dataset_length.remote())  # type: ignore[attr-defined]
            // dataflow_config.global_batch_size
            * total_epochs
        )
        bind_train_rollout(train_controller=self._train_controller, env_controller=self._rollout_env_controller)
        ray.get(self._train_controller.offload.remote(target="all"))

        self._train_worker_cfg = train_worker_cfg

        if self._rl_trainer_cfg is not None and get_rank() == 0:
            config_path = log_dir / "rl_trainer_config.json"
            with config_path.open("w") as f:
                f.write(self._rl_trainer_cfg.model_dump_json(indent=2))

            env_path = log_dir / "env.json"
            environment_variables = dict(os.environ)
            infer_engine_version = get_rollout_engine_version()
            environment_variables.update(infer_engine_version)
            with env_path.open("w") as f:
                json.dump(environment_variables, f, indent=2)

    @classmethod
    def from_config(cls, config: RLTrainerConfig) -> Self:
        """Create a Trainer instance from a TrainerConfig.

        Args:
            config (TrainerConfig): TrainerConfig instance containing all configuration parameters.

        Returns:
            Self: Trainer instance initialized with the provided config.
        """
        self = cls(
            load_from=config.load_from,
            resources=config.resources,
            rollout_config=config.rollout_config,
            dataflow_config=config.dataflow_config,
            judger_config=config.judger_config,
            replay_buffer_config=config.replay_buffer_config,
            train_worker_cfg=config.train_worker_config,
            evaluator_config=config.evaluator_config,
            tokenizer_path=config.tokenizer_path,
            work_dir=config.work_dir,
            log_dir=config.log_dir,
            total_epochs=config.total_epochs,
            resume_config=config.resume_config,
            strict_load=config.strict_load,
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            seed=config.seed,
            debug=config.debug,
            trainer_cfg=config,
        )
        return self

    def _build_rollout_dataflow(
        self,
        dataflow_cfg: DataFlowConfig,
        rollout_cfg: RolloutConfig,
        judger_cfg: JudgerConfig,
        replay_buffer_config: ReplayBufferConfig,
    ):
        env = cast(ActorClass, SingleTurnEnvironment).remote("grpo", self._pg, rollout_cfg, self._pg, judger_cfg)
        flow = cast(ActorClass, DataFlow).remote("grpo", dataflow_cfg, replay_buffer_config, env)
        return env, flow

    def _build_train_controller(self, train_worker_cfg: WorkerConfig):
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorker, train_worker_cfg, self._pg)
        ray.get([worker.__ray_ready__.remote() for worker in train_workers])
        train_controller = cast(ActorClass, TrainingController).remote(
            workers=train_workers,
        )
        ray.get(train_controller.__ray_ready__.remote())
        return train_controller

    def fit(self):
        """Run the RL training loop.

        This method executes the main rl training loop, iterating generating through the dataset and performing
        training steps. It handles rollout, prepare training data, update policy , synchronize model weights, and
        evaluation.
        """
        self.logger.info("Start RL training")
        if self._enable_initial_evaluate and self._enable_evaluate and self._evaluator:
            ray.get(self._rollout_env_controller.check_active_workers.remote())
            scores, eval_data_groups = ray.get(self._evaluator.run.remote(return_samples=True))
            trajectory_save_path = self.exp_dir / "eval_0_trajectory.jsonl"
            self._save_trajectories(eval_data_groups, trajectory_save_path)
            self.logger.info(f"Initial rollout evaluate scores {scores} and start training")
        for rollout_idx in range(1, self._rollout_steps + 1):
            timer_log_str = f"Rollout {rollout_idx} start \n"
            step_timer_dict = {}
            memory_status_dict = {}
            # 1. Rollout
            with timer("generation", step_timer_dict):
                ray.get(self._rollout_env_controller.check_active_workers.remote())
                data_groups, multimodal_train_infos = ray.get(self._rollout_dataflow.run.remote())
            # self._log_memory_usage(rollout_idx, stage="generation")
            memory_status_dict["generation"] = self._get_memory_stats()
            # 2. Offload rollout models and save trajectories
            with timer("offload_and_dump", step_timer_dict):
                ray.get(self._rollout_env_controller.offload.remote())
                trajectory_save_path = self.exp_dir / f"rollout_idx_{rollout_idx}_trajectory.jsonl"
                self._save_trajectories(data_groups, trajectory_save_path)
                self.logger.info(f"Rollout_idx {rollout_idx} finished, saved trajectories to {trajectory_save_path}")
            # self._log_memory_usage(rollout_idx, stage="offload_and_dump")
            memory_status_dict["offload_and_dump"] = self._get_memory_stats()
            # 3. Onload training models and prepare data
            with timer("onload_and_prepare_data", step_timer_dict):
                ray.get(self._train_controller.onload.remote(target="all"))
                self.logger.info("Training controller loaded")
                data_batches, data_info = self._prepare_train_data(
                    data_groups, self._train_worker_cfg.pack_max_length, multimodal_train_infos
                )
                self.logger.info(f"Prepared {len(data_batches)} training data batches")
                self._log_data_info(rollout_idx, data_info)
            memory_status_dict["onload_and_prepare_data"] = self._get_memory_stats()
            # 4. Training Step
            with timer("training", step_timer_dict):
                ray.get(
                    self._train_controller.fit.remote(
                        data_batches, pack_max_length=self._train_worker_cfg.pack_max_length, rollout_idx=rollout_idx
                    )
                )
            memory_status_dict["training"] = self._get_memory_stats()
            # 5. Saving and sync weights
            with timer("saving_and_sync_weight", step_timer_dict):
                ray.get(self._train_controller.offload.remote(target="optimizer"))
                self._maybe_save_hf()
                bind_train_rollout(
                    train_controller=self._train_controller, env_controller=self._rollout_env_controller
                )
                ray.get(self._rollout_env_controller.onload_weights.remote())
                ray.get(self._train_controller.update_weights.remote())
                self.logger.info("Model weights synchronized successfully.")
                ray.get(self._train_controller.offload.remote(target="model"))
                ray.get(self._rollout_env_controller.onload_kvcache.remote())
            memory_status_dict["saving_and_sync_weight"] = self._get_memory_stats()
            timer_log_str = f"Rollout {rollout_idx} training finished and timing listed: \n"
            timer_log_str += timer_logger(step_timer_dict)

            self.logger.info(timer_log_str)

            # evaluate
            if self._enable_evaluate and self._evaluator and rollout_idx % self._eval_step == 0:
                scores, eval_data_groups = ray.get(self._evaluator.run.remote(return_samples=True))
                trajectory_save_path = self.exp_dir / f"eval_{rollout_idx}_trajectory.jsonl"
                self._save_trajectories(eval_data_groups, trajectory_save_path)
                self.logger.info(f"Evaluate idx {rollout_idx} scores {scores}")
            memory_status_dict["saving_and_sync_weight"] = self._get_memory_stats()
            self._log_memory_usage(rollout_idx, memory_status_dict)
            self._cur_step += 1

    def _get_memory_stats(self) -> dict:
        """Gets the current CPU and GPU memory usage stats."""
        stats = {}
        # CPU Memory
        cpu_mem = psutil.virtual_memory()
        stats["cpu"] = {
            "used_gb": cpu_mem.used / (1024**3),
            "total_gb": cpu_mem.total / (1024**3),
            "percent": cpu_mem.percent,
        }

        # GPU Memory
        if torch.cuda.is_available():
            stats["gpu"] = {}
            for i in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                used_mem = total_mem - free_mem
                stats["gpu"][str(i)] = {
                    "used_gb": used_mem / (1024**3),
                    "total_gb": total_mem / (1024**3),
                    "percent": (used_mem / total_mem) * 100 if total_mem > 0 else 0,
                }
        return stats

    def _log_memory_usage(self, rollout_idx: int, memory_stats: dict):
        """Logs CPU and GPU memory usage across all stages for a rollout."""
        if get_rank() != 0:
            return

        log_msg = f"Rollout {rollout_idx} Memory Usage Summary: "

        rl_training_stats = memory_stats.get("RL_training")
        if rl_training_stats:
            cpu_usage = rl_training_stats.get("cpu", {}).get("used_gb", 0.0)
            avg_gpu_usage = 0.0
            gpu_stats = rl_training_stats.get("gpu")
            if gpu_stats and len(gpu_stats) > 0:
                total_gpu_usage = sum(stat.get("used_gb", 0.0) for stat in gpu_stats.values())
                avg_gpu_usage = total_gpu_usage / len(gpu_stats)

            log_msg += f"Avg CPU Usage: {cpu_usage:.2f} GB, Avg GPU Usage: {avg_gpu_usage:.2f} GB"

        for stage, stats in memory_stats.items():
            log_msg += f"\n[Stage: {stage}]"
            if "cpu" in stats:
                cpu_stat = stats["cpu"]
                log_msg += (
                    f"\n  - CPU Memory: Used {cpu_stat['used_gb']:.2f} GB / "
                    f"Total {cpu_stat['total_gb']:.2f} GB ({cpu_stat['percent']}%)"
                )

            if "gpu" in stats:
                log_msg += "\n  - GPU Memory:"
                for i, gpu_stat in stats["gpu"].items():
                    log_msg += (
                        f"\n    - Rank {i}: Used {gpu_stat['used_gb']:.2f} GB / "
                        f"Total {gpu_stat['total_gb']:.2f} GB ({gpu_stat['percent']:.2f}%)"
                    )
        self.logger.info(log_msg)
        
    def _log_data_info(self, rollout_idx: int, data_info: dict):
        """Formats and logs the data statistics dictionary."""
        log_lines = [f"Rollout {rollout_idx} data statistics:"]
        for key, value in data_info.items():
            if isinstance(value, float):
                log_lines.append(f"  - {key:<20}: {value:.4f}")
            else:
                log_lines.append(f"  - {key:<20}: {value}")
        self.logger.info("\n".join(log_lines))

    def _log_memory_usage(self, rollout_idx: int, stage: str):
        """Logs CPU and GPU memory usage at a given stage."""
        if get_rank() != 0:
            return

        log_msg = f"Rollout {rollout_idx} memory usage after stage: '{stage}'"

        # CPU Memory
        cpu_mem = psutil.virtual_memory()
        log_msg += (
            f"\n  - CPU Memory: Used {cpu_mem.used / (1024**3):.2f} GB / "
            f"Total {cpu_mem.total / (1024**3):.2f} GB ({cpu_mem.percent}%)"
        )

        # GPU Memory
        if torch.cuda.is_available():
            log_msg += "\n  - GPU Memory:"
            for i in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                used_mem = total_mem - free_mem
                log_msg += (
                    f"\n    - Rank {i}: Used {used_mem / (1024**3):.2f} GB / "
                    f"Total {total_mem / (1024**3):.2f} GB ({(used_mem / total_mem) * 100:.2f}%)"
                )
        self.logger.info(log_msg)

    # TODO: advantage 是在 DataFlow 里算好，还是在 train controller 里算？
    # 因为可能有根据 advantage 来判断数据能否进 rl 训练的需求。暂时先放在这
    def _prepare_train_data(self, data_groups, pack_max_length, multimodal_train_infos=None):
        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []

        data_batches = []
        is_multimodal = False
        if multimodal_train_infos and len(multimodal_train_infos) > 0:
            assert len(multimodal_train_infos) == len(data_groups), (
                f"{len(multimodal_train_infos)} vs {len(data_groups)}"
            )
            is_multimodal = True

        for j, group in enumerate(data_groups):
            if not check_valid_dataflow_item(group):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue
            if is_multimodal:
                multimodal_train_info = multimodal_train_infos[j]
            else:
                multimodal_train_info = None

            prompt_ids = group[0].data.extra_info["train_prompt_ids"]
            rewards = [data.env.judger.reward["score"] for data in group]
            rewards_list.extend(rewards)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

            prompt_repeat_k = len(group)
            for i in range(prompt_repeat_k):
                item = group[i].env.rollout.response
                logprobs = None
                if group[i].env.rollout.response_ids is not None:
                    response_ids = group[i].env.rollout.response_ids
                    if isinstance(response_ids, torch.Tensor):
                        response_ids = response_ids.flatten().tolist()
                    logprobs = group[i].env.rollout.logprobs
                    assert len(logprobs) == len(response_ids), f"{len(logprobs)} vs {len(response_ids)}"
                    # 只有 response 部分有 logprobs, 需要前面追加
                    logprobs = [0] * (len(prompt_ids) - 1) + logprobs
                else:
                    response_ids = self.tokenizer(item, return_tensors="pt")["input_ids"].flatten().tolist()
                # 返回的 routed_experts 不包括 eos 的值，实际上也不需要，需要减一
                input_ids = prompt_ids + response_ids[:-1]

                prompt_len_list.append(len(prompt_ids))
                response_len_list.append(len(response_ids))
                advantages_list.extend([advantages[i]] * len(response_ids))

                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids
                assert len(input_ids) <= pack_max_length, (
                    f"{len(input_ids)} vs {pack_max_length}, input_ids: {len(input_ids)}, response_ids: {len(response_ids)}"
                )
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                if logprobs is not None:
                    rollout_logprobs = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
                    assert rollout_logprobs.size() == shifted_labels.size(), (
                        f"{rollout_logprobs.size()} vs {shifted_labels.size()}"
                    )
                else:
                    rollout_logprobs = None

                seq_ctx = get_train_seq_ctx(input_ids, multimodal_train_info, len(response_ids) - 1)
                data_dict = {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels,
                    "advantage": advantages[i].item(),
                    "rollout_logprobs": rollout_logprobs,
                }

                if "routed_experts" in group[i].env.rollout.extra_info:
                    routed_experts = group[i].env.rollout.extra_info["routed_experts"]  # n,layer*expert
                    seq_ctx.rollout_routed_experts = routed_experts  # n,layer,expert

                data_batches.append(data_dict)
        random.shuffle(data_batches)

        advantages_list = np.array(advantages_list)
        info_dict = {
            "batch_size": len(rewards_list),
            "rewards/mean": np.mean(rewards_list),
            "rewards/min": np.min(rewards_list),
            "rewards/max": np.max(rewards_list),
            "advantages/mean": np.mean(advantages_list),
            "advantages/min": np.min(advantages_list),
            "advantages/max": np.max(advantages_list),
            "response_len/mean": np.mean(response_len_list),
            "response_len/min": np.min(response_len_list),
            "response_len/max": np.max(response_len_list),
            "response_len/std": np.std(response_len_list),
            "prompt_len/mean": np.mean(prompt_len_list),
            "prompt_len/min": np.min(prompt_len_list),
            "prompt_len/max": np.max(prompt_len_list),
        }
        return data_batches, info_dict

    def _save_trajectories(self, data_groups, save_path):
        rewards = []

        rollout_response_len_list = []
        version_dict = {i: 0 for i in range(self._dataflow_partial_rollout_step + 1)}

        # NOTE: Since we currently default to token-in token-out, the code for checking whether response_ids have Retokenization Drift is commented out.
        # If you need to debug, you can uncomment it.
        # mismatch_token_ids_count = 0
        # response_len_list = []
        for group in data_groups:
            if not check_valid_dataflow_item(group):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue
            for data in group:
                rewards.append(data.env.judger.reward["score"])
                if data.env.rollout.response_ids is not None:
                    if isinstance(data.env.rollout.response_ids, torch.Tensor):
                        response_ids = data.env.rollout.response_ids.flatten().tolist()
                    else:
                        response_ids = data.env.rollout.response_ids
                    rollout_response_len_list.append(len(response_ids))
                    # response_str = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                    # revert_encode_response_ids = self.tokenizer.encode(response_str, add_special_tokens=False)

                    # response_str_to_ids = self.tokenizer.encode(data.env.rollout.response, add_special_tokens=False)
                    # response_len_list.append(len(response_str_to_ids))

                    # if response_ids != revert_encode_response_ids or response_ids != response_str_to_ids:
                    #     mismatch_token_ids_count += 1
                else:
                    response_ids = self.tokenizer.encode(data.env.rollout.response, add_special_tokens=False)
                    rollout_response_len_list.append(len(response_ids))

                version = data.uid.version
                if version not in version_dict:
                    version_dict[version] = 0
                version_dict[version] += 1

        rewards = torch.tensor(rewards).float()
        rollout_response_lens = None
        if len(rollout_response_len_list) > 0:
            rollout_response_lens = torch.tensor(rollout_response_len_list).float()

        _count = 0
        with open(save_path, "w", encoding="utf-8") as f:
            item = {
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "reward_max": rewards.max().item(),
                "reward_min": rewards.min().item(),
                "response_len_mean": rollout_response_lens.mean().item(),
                "response_len_std": rollout_response_lens.std().item(),
                "response_len_max": rollout_response_lens.max().item(),
                "response_len_min": rollout_response_lens.min().item(),
                "total_len": len(rewards),
                "versions": version_dict,
                # "mismatch_token_ids_count": mismatch_token_ids_count,
            }
            json.dump(item, f, ensure_ascii=False, indent=2)
            f.write("\n")
            for group in data_groups:
                for data in group:
                    item = {
                        "prompt": data.data.extra_info["raw_prompt"],
                        "response": data.env.rollout.response,
                        "response_len": rollout_response_len_list[_count],
                        "label": data.data.reward_model["ground_truth"],
                        "reward": data.env.judger.reward["score"],
                        "version": data.uid.version,
                        "finish_reason": data.env.rollout.finish_reason,
                    }
                    json.dump(item, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                    _count += 1

    def _load_trajectories(self, save_path):
        data_groups = []
        with open(save_path) as f:
            for line in f:
                item = json.loads(line)
                messages = item["messages"]
                responses = item["response"]
                rewards = item["reward"]
                group = []
                for response, reward in zip(responses, rewards):
                    group.append(
                        {
                            "messages": messages,
                            "response_str": response,
                            "reward": reward,
                        }
                    )
                data_groups.append(group)
        return data_groups

    def _compute_metrics(self, data_groups):
        correctness = [1 if data[0]["reward"] > 0 else 0 for data in data_groups]
        acc = sum(correctness) / len(correctness)
        return acc

    def _maybe_save_hf(self):
        if self._hf_interval is None:
            return

        assert self._load_from_hf, (
            "Only support saving to Huggingface format when loading from Huggingface! "
            "You meet this error means `load_from` of trainer is not a Huggingface model path."
        )

        if (self.cur_step + 1) % self._hf_interval != 0 and (self.cur_step + 1) != self._rollout_steps:
            return

        save_hf_path = self.exp_dir / f"hf-{self.cur_step + 1}"
        self.logger.info(f"Saving step {self.cur_step + 1} checkpoints to: {save_hf_path}")
        self.meta.latest_exp.hf_checkpoint_list.append(str(save_hf_path))

        if self._hf_max_keep is not None and len(self.meta.latest_exp.hf_checkpoint_list) > self._hf_max_keep:
            deleted_hf_checkpoints = self.meta.latest_exp.hf_checkpoint_list[: -self._hf_max_keep]
            self.meta.latest_exp.hf_checkpoint_list = self.meta.latest_exp.hf_checkpoint_list[-self._hf_max_keep :]
            for hf_dir in deleted_hf_checkpoints:
                rmtree(hf_dir)

        ray.get(self._train_controller.save_hf.remote(str(save_hf_path)))
        if isinstance(self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer.save_pretrained(str(save_hf_path))
        meta_path = self.work_dir / self.META_PATH

        with meta_path.open("w") as f:
            f.write(self.meta.model_dump_json(indent=2))

    def _init_logger(self, work_dir: Path):
        # Logging system maybe need better design
        logger = get_logger(log_dir=work_dir, tag="RLTrainer")
        return logger

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        set_random_seed(seed)

    def _init_xtuner_meta(self, work_dir: Path, resume: bool) -> XTunerMeta:
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        meta_path = work_dir / self.META_PATH
        if not meta_path.exists():
            meta = XTunerMeta(exps=[])
            with open(meta_path, "w") as f:
                f.write(meta.model_dump_json(indent=2))

        meta = cast(XTunerMeta, XTunerMeta.model_validate(load(meta_path, file_format="json")))

        resume = resume and bool(meta.exps)

        if resume:
            latest_exp = meta.exps[-1]
            latest_exp_history = latest_exp.history[-1]

            begin = cast(int, latest_exp_history.get("end") or latest_exp_history["begin"])
            exp_dir = Path(latest_exp.exp_dir)
            git_dir = exp_dir / f"git-info-begin-{begin}"

            if not git_dir:
                git_dir.mkdir(parents=True, exist_ok=True)

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"

            commit = record_git_info(staged_path, unstaged_path)
            git_info = GitInfo(
                commit=commit,
                staged=str(staged_path),
                unstaged=str(unstaged_path),
            )

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_exp_history = ExpHistory(
                begin=begin,
                timestamp=timestamp,
                git_info=git_info,
            )
            latest_exp.history.append(new_exp_history)
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            exp_dir = work_dir / timestamp
            git_dir = Path(f"{exp_dir}/git-info-begin-{0}")

            if not git_dir.exists():
                git_dir.mkdir(parents=True, exist_ok=True)

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"
            commit = record_git_info(staged_path, unstaged_path)
            git_info = GitInfo(
                commit=commit,
                staged=str(staged_path),
                unstaged=str(unstaged_path),
            )

            new_history = ExpHistory(
                begin=0,
                timestamp=timestamp,
                git_info=git_info,
            )
            new_exp = ExpInfo(history=[new_history], exp_dir=str(exp_dir))
            meta.exps.append(new_exp)
        return meta

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def exp_dir(self) -> Path:
        return Path(self._meta.latest_exp.exp_dir)

    @property
    def meta(self) -> XTunerMeta:
        return self._meta

    @property
    def cur_step(self):
        return self._cur_step

    @property
    def total_epoch(self):
        return self._total_epochs

    @property
    def rollout_steps(self):
        return self._rollout_steps
