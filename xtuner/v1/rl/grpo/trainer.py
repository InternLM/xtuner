import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import cast

import ray
import torch
from mmengine import load
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel
from ray.actor import ActorClass
from typing_extensions import NotRequired, TypedDict

from transformers import AutoTokenizer
from xtuner.v1.config.trainer import ResumeConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.evaluator import Evaluator, EvaluatorConfig
from xtuner.v1.ray.judger import JudgerConfig
from xtuner.v1.ray.rollout import SampleParams
from xtuner.v1.rl.grpo.controller import GRPOTrainingController
from xtuner.v1.rl.grpo.worker import GRPOTrainingWorker, WorkerConfig
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    get_logger,
    is_hf_model_path,
    log_format,
    record_git_info,
)
from xtuner.v1.utils.device import get_device, get_torch_device_module


# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class GitInfo(TypedDict):
    commit: str
    staged: str
    unstaged: str


class ExpHistory(TypedDict):
    begin: int
    timestamp: str
    git_info: GitInfo
    end: NotRequired[int]
    comment: NotRequired[str]


class ExpInfo(BaseModel):
    history: list[ExpHistory]
    exp_dir: str
    latest_checkpoint: str | None = None
    hf_checkpoint_list: list[str] = []


class XTunerMeta(BaseModel):
    exps: list[ExpInfo]

    @property
    def latest_checkpoint(self) -> str | None:
        for exp in self.exps:
            if exp.latest_checkpoint is not None:
                return exp.latest_checkpoint
        return None

    @property
    def latest_exp(self) -> ExpInfo:
        return self.exps[-1]


def bind_train_rollout(
    train_controller,
    env_controller,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return


class Trainer:
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
        evaluator_config: EvaluatorConfig,
        train_worker_cfg: WorkerConfig,
        tokenizer_path: str | Path,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        total_epochs: int,
        enable_evaluate: bool,
        resume_config: ResumeConfig | None = None,
        strict_load: bool = True,
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        seed: int = 42,
        debug: bool = False,
    ):
        # TODO
        rollout_config.model_path = load_from
        train_worker_cfg.load_from = load_from

        self._total_epochs = total_epochs
        self._cur_epoch = 0

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
        self._enable_evaluate = enable_evaluate
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
        self._evaluator = Evaluator.remote(evaluator_config, self._rollout_env_controller)  # type: ignore[attr-defined]
        self._evaluator_sample_params = SampleParams(
            top_p=1.0, temperature=0.0, do_sample=False, max_tokens=dataflow_config.sample_params.max_tokens, top_k=1
        )
        self._global_batch_size = dataflow_config.global_batch_size
        self._eval_step = evaluator_config.evaluate_step
        self._rollout_steps = (
            ray.get(self._rollout_dataflow.get_train_dataset_length.remote())  # type: ignore[attr-defined]
            // dataflow_config.global_batch_size
            * total_epochs
        )
        bind_train_rollout(train_controller=self._train_controller, env_controller=self._rollout_env_controller)
        ray.get(self._train_controller.offload.remote(target="all"))

        self._train_worker_cfg = train_worker_cfg

    def _build_rollout_dataflow(
        self,
        dataflow_cfg: DataFlowConfig,
        rollout_cfg: RolloutConfig,
        judger_cfg: JudgerConfig,
        replay_buffer_config: ReplayBufferConfig,
    ):
        env = cast(ActorClass, SingleTurnEnvironment).remote("grpo", self._pg, rollout_cfg, judger_cfg)
        flow = cast(ActorClass, DataFlow).remote("grpo", dataflow_cfg, replay_buffer_config, env)
        return env, flow

    def _build_train_controller(self, train_worker_cfg: WorkerConfig):
        train_workers = AutoAcceleratorWorkers.from_placement_group(GRPOTrainingWorker, train_worker_cfg, self._pg)
        ray.get([worker.__ray_ready__.remote() for worker in train_workers])
        train_workers = list(train_workers.keys())
        train_controller = cast(ActorClass, GRPOTrainingController).remote(
            workers=train_workers,
        )
        ray.get(train_controller.__ray_ready__.remote())
        return train_controller

    def fit(self):
        self.logger.info("start training")
        if self._enable_evaluate:
            scores, eval_data_groups = ray.get(
                self._evaluator.run.remote(return_samples=True, sample_params=self._evaluator_sample_params)
            )
            trajectory_save_path = self.exp_dir / "initial_trajectory.jsonl"
            self._save_trajectories(eval_data_groups, trajectory_save_path)
            self.logger.info(f"Initial rollout evaluate scores {scores} and start training")
        for rollout_idx in range(1, self._rollout_steps + 1):
            data_groups = ray.get(self._rollout_dataflow.run.remote())
            time.sleep(3)
            ray.get(self._rollout_env_controller.offload_weights_and_kvcache.remote())
            trajectory_save_path = self.exp_dir / f"rollout_idx_{rollout_idx}_trajectory.jsonl"
            self._save_trajectories(data_groups, trajectory_save_path)
            self.logger.info(f"rollout_idx {rollout_idx} finished, saved trajectories to {trajectory_save_path}")
            ray.get(self._train_controller.onload.remote(target="all"))
            self.logger.info("Training controller loaded")
            data_batches = self._prepare_train_data(data_groups, self._train_worker_cfg.pack_max_length)
            self.logger.info(f"Prepared {len(data_batches)} training data batches")
            ray.get(
                self._train_controller.fit.remote(
                    data_batches, pack_max_length=self._train_worker_cfg.pack_max_length, rollout_idx=rollout_idx
                )
            )
            ray.get(self._train_controller.offload.remote(target="optimizer"))
            ray.get(self._rollout_env_controller.onload_weights.remote())
            ray.get(self._train_controller.update_weights.remote())
            self.logger.info("update weights done!!!")
            ray.get(self._train_controller.offload.remote(target="model"))
            ray.get(self._rollout_env_controller.onload_kvcache.remote())
            # evaluate
            if self._enable_evaluate and rollout_idx % self._eval_step == 0:
                scores = ray.get(self._evaluator.run.remote(sample_params=self._evaluator_sample_params))
                self.logger.info(f"evaluate idx {rollout_idx} scores {scores}")
            self._cur_epoch += 1
            self._maybe_save_hf()

    # TODO: advantage 是在 DataFlow 里算好，还是在 train controller 里算？
    # 因为可能有根据 advantage 来判断数据能否进 rl 训练的需求。暂时先放在这
    def _prepare_train_data(self, data_groups, pack_max_length):
        data_batches = []
        for group in data_groups:
            prompt = self.tokenizer.apply_chat_template(
                group[0]["messages"], add_generation_prompt=True, tokenize=False
            )
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten().tolist()
            rewards = [data["reward"] for data in group]
            rewards = torch.tensor(rewards, dtype=torch.float32)
            advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

            prompt_repeat_k = len(group)
            for i in range(prompt_repeat_k):
                item = group[i]["response_str"]
                response_ids = self.tokenizer(item, return_tensors="pt")["input_ids"].flatten().tolist()
                input_ids = prompt_ids + response_ids
                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids + [-100]
                if len(input_ids) > pack_max_length:
                    input_ids = input_ids[:pack_max_length]
                    shifted_labels = shifted_labels[:pack_max_length]
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)
                data_batches.append(
                    dict(
                        seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cpu"),
                        shifted_labels=shifted_labels,
                        advantage=advantages[i].item(),
                    )
                )
        random.shuffle(data_batches)
        return data_batches

    def _save_trajectories(self, data_groups, save_path):
        with open(save_path, "w") as f:
            for group in data_groups:
                response_list = []
                reward_list = []
                for data in group:
                    response_list.append(data["response_str"])
                    reward_list.append(data["reward"])
                item = {
                    "messages": group[0]["messages"],
                    "response": response_list,
                    "label": group[0]["reward_model"]["ground_truth"],
                    "reward": reward_list,
                }
                json.dump(item, f)
                f.write("\n")

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

        if self.cur_epoch % self._hf_interval != 0 and self.cur_epoch != self.total_epoch:
            return

        save_hf_path = self.exp_dir / f"hf-{self.cur_epoch}"
        self.meta.latest_exp.hf_checkpoint_list.append(str(save_hf_path))

        if self._hf_max_keep is not None and len(self.meta.latest_exp.hf_checkpoint_list) > self._hf_max_keep:
            deleted_hf_checkpoints = self.meta.latest_exp.hf_checkpoint_list[: -self._hf_max_keep]
            self.meta.latest_exp.hf_checkpoint_list = self.meta.latest_exp.hf_checkpoint_list[-self._hf_max_keep :]
            for hf_dir in deleted_hf_checkpoints:
                rmtree(hf_dir)

        ray.get(self._train_controller.save_hf.remote(str(save_hf_path)))
        meta_path = self.work_dir / self.META_PATH

        with meta_path.open("w") as f:
            f.write(self.meta.model_dump_json(indent=2))

    def _init_logger(self, work_dir: Path):
        # Logging system maybe need better design
        logger = get_logger()
        logger.remove()
        logger.add(work_dir / f"rank{get_rank()}.log", format=log_format(), backtrace=True, catch=True)
        logger.add(sys.stderr, format=log_format(rank=get_rank()))
        return logger

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
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
            new_exp = ExpInfo(
                history=[new_history],
                exp_dir=str(exp_dir),
                latest_checkpoint=None,
            )
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
    def cur_epoch(self):
        return self._cur_epoch

    @property
    def total_epoch(self):
        return self._total_epochs

    @property
    def rollout_steps(self):
        return self._rollout_steps
