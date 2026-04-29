import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import ray
import requests
import torch
from mmengine.dist import get_rank
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator
from ray.actor import ActorClass
from ray.util.placement_group import PlacementGroup
from transformers import AutoTokenizer
from typing_extensions import Self

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment.lagent.tokenize import tokenize
from xtuner.v1.ray.evaluator import Evaluator, EvaluatorConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.config.advantage import BaseAdvantageConfig, GRPOAdvantageConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import is_hf_model_path, timer
from xtuner.v1.utils.device import get_device, get_torch_device_module
from xtuner.v1.utils.env_check import get_rollout_engine_version

from .rl_trainer import (
    TRAINER_RAY_GET_TIMEOUT,
    RLTrainer,
    RLTrainerConfig,
    bind_train_rollout,
)

# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class AgentRLTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    load_from: str | Path  # Huggingface model path or saved trainer_path
    pg: Any
    environment_config: dict
    dataflow_config: DataFlowConfig
    replay_buffer_config: ReplayBufferConfig
    train_worker_cfg: WorkerConfig
    evaluator_config: EvaluatorConfig | None = None
    tokenizer_path: str | Path
    work_dir: Path | str | None = None
    log_dir: Path | str | None = None
    total_epochs: int
    auto_resume: bool = False
    load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig()
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    checkpoint_no_save_optimizer: bool = False
    skip_checkpoint_validation: bool = False  # Suggest enabled if fsdp_size is larger than 512
    hf_interval: int | None = None
    hf_max_keep: int | None = None
    seed: int = 42
    debug: bool = False
    debug_rollout: bool = False
    rollout_steps: int | None = None
    exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard"
    display_all_workers_log: bool = False
    skip_load_weights: bool = False
    advantage_estimator_config: BaseAdvantageConfig = GRPOAdvantageConfig()

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self

    @field_serializer("replay_buffer_config")
    def serialize_replay_buffer_cfg(self, replay_buffer_config: ReplayBufferConfig) -> str:
        return replay_buffer_config.model_dump(include={"replay_ratio", "replay_weights"})  # type: ignore[return-value]

    @field_serializer("evaluator_config")
    def serialize_evaluator_cfg(self, evaluator_config: EvaluatorConfig) -> str:  # type: ignore[return-value]
        if evaluator_config:
            return evaluator_config.model_dump(exclude={"tokenizer", "dataset_cfg", "compute_metric_func"})  # type: ignore[return-value]
        else:
            return ""

    @field_serializer("pg")
    def serialize_pg(self, pg: PlacementGroup) -> str:
        return f"PlacementGroup(id={pg.id})"

    @field_serializer("environment_config")
    def serialize_environment_config(self, environment_config: dict) -> str:
        return str(environment_config)


class AgentRLTrainer(RLTrainer):
    def __init__(
        self,
        *,
        load_from: str | Path,  # Huggingface model path or saved trainer_path
        pg: PlacementGroup,
        environment_config: Dict,
        dataflow_config: DataFlowConfig,
        replay_buffer_config: ReplayBufferConfig,
        train_worker_cfg: WorkerConfig,
        evaluator_config: EvaluatorConfig | None = None,
        tokenizer_path: str | Path,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        total_epochs: int,
        auto_resume: bool = False,
        load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig(),
        checkpoint_interval: int | None = -1,
        checkpoint_maxkeep: int | None = -1,
        checkpoint_no_save_optimizer: bool = False,
        skip_checkpoint_validation: bool = False,  #
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        seed: int = 42,
        debug: bool = False,
        debug_rollout: bool = False,
        rollout_steps: int | None = None,
        exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard",
        display_all_workers_log: bool = False,
        trainer_cfg: RLTrainerConfig | None = None,
        skip_load_weights: bool = False,
        advantage_estimator_config: BaseAdvantageConfig = GRPOAdvantageConfig(),
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

        if skip_checkpoint_validation:
            patch_default_save_plan()

        self._rl_trainer_cfg = trainer_cfg
        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from

        is_hf_path, error_info = is_hf_model_path(load_from) if load_from is not None else False, ""
        self._load_from_hf = is_hf_path

        if not self._load_from_hf:
            raise NotImplementedError(error_info)

        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_maxkeep = checkpoint_maxkeep
        self._checkpoint_no_save_optimizer = checkpoint_no_save_optimizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self._debug = debug
        self._debug_rollout = debug_rollout
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
        self._auto_resume = auto_resume
        self._meta = self._init_xtuner_meta(work_dir, self._auto_resume)

        if log_dir is None:
            log_dir = self.exp_dir
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.logger = self._init_logger(log_dir)

        self._load_checkpoint_cfg = self._resolve_load_checkpoint_cfg(self._auto_resume, load_checkpoint_cfg)

        train_worker_cfg.log_dir = log_dir
        dataflow_config.worker_log_dir = log_dir
        # if rollout_config is not None:
        #     rollout_config.worker_log_dir = log_dir
        # self._enable_return_routed_experts = rollout_config.enable_return_routed_experts
        self._enable_evaluate = False
        self._enable_initial_evaluate = False
        if evaluator_config:
            evaluator_config.worker_log_dir = log_dir
            self._enable_evaluate = evaluator_config.enable_evaluate
            self._enable_initial_evaluate = evaluator_config.enable_initial_evaluate
        self._pg = pg
        # We need to build train controller first, and then build rollout dataflow to make
        # inference engines know how much memory they can utilize.
        self._train_controller = self._build_train_controller(train_worker_cfg)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            skip_load_weights = True
            # rollout_config.skip_load_weights = True
            self.logger.info(
                f"Skip load rollout weights due to resume from checkpoint {self._load_checkpoint_cfg.checkpoint_path}"
            )

            # resume train worker
            ray.get(self._train_controller.resume.remote(self._load_checkpoint_cfg))

            train_state_path = Path(self._load_checkpoint_cfg.checkpoint_path) / self._SAVE_TRAIN_STATE_PATH
            with train_state_path.open("r") as f:
                train_state = json.load(f)
                self._cur_step = train_state["cur_step"]

        self._rollout_env_controller, self._rollout_dataflow = self._build_rollout_dataflow(
            environment_cfg=environment_config,
            dataflow_cfg=dataflow_config,
            replay_buffer_config=replay_buffer_config,
        )

        rollout_info = ray.get(self._rollout_env_controller.get_rollout_info.remote())
        print(f"rollout_info {rollout_info}")
        self.model_name = rollout_info["rollout_config"].model_name
        api_server_url = rollout_info["api_server_url"]

        # 写死 0.0.0.0:8000
        url = "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1/models/new"
        payload = {
            "model_name": self.model_name,
            "api_key": "sk-admin",
            "api_base": api_server_url,
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        print("register model success", resp.json())

        # import time

        # time.sleep(1000000000)

        self._dataflow_partial_rollout_step = dataflow_config.tail_batch_candidate_steps

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            # resume rollout dataflow
            self.logger.info(f"Resume rollout dataflow from checkpoint {self._load_checkpoint_cfg.checkpoint_path}")
            ray.get(self._rollout_dataflow.resume.remote(self._load_checkpoint_cfg.checkpoint_path))  # type: ignore[union-attr]

        if self._enable_evaluate and evaluator_config:
            self._evaluator = Evaluator.remote(evaluator_config, self._rollout_env_controller)  # type: ignore[attr-defined, union-attr]
            self._eval_step = evaluator_config.evaluate_step
        else:
            pass

        self._global_batch_size = dataflow_config.global_batch_size
        self._rollout_steps = (
            ray.get(self._rollout_dataflow.get_train_dataset_length.remote())  # type: ignore[attr-defined]
            // dataflow_config.global_batch_size
            * total_epochs
        )
        if rollout_steps is not None:
            self._rollout_steps = rollout_steps
            self.logger.info(f"Set rollout steps to {self._rollout_steps} according to rollout_steps arg")

        bind_train_rollout(train_controller=self._train_controller, env_controller=self._rollout_env_controller)
        # update weights if rollout_config.skip_load_weights == True
        # if rollout_config.skip_load_weights:
        if skip_load_weights:
            self.logger.info("Rollout workers skip load weights, update weights from train workers.")
            ray.get(self._train_controller.offload.remote(target="optimizer"))
            ray.get(self._rollout_env_controller.offload.remote())
            ray.get(self._rollout_env_controller.onload_weights.remote())
            ray.get(self._train_controller.update_weights.remote())
            ray.get(self._train_controller.offload.remote(target="model"))
            ray.get(self._rollout_env_controller.onload_kvcache.remote())
            self.logger.info("Rollout workers has updated weights from train workers.")
        else:
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

        self._ray_get_timeout = TRAINER_RAY_GET_TIMEOUT
        self._exp_tracker = self._init_tracker(exp_tracker, log_dir / self._EXP_TRACKING_PATH)
        self._display_all_workers_log = display_all_workers_log

        self._train_results: dict = defaultdict(list)
        self._eval_results: dict = defaultdict(list)

        self._advantage_estimator = advantage_estimator_config.build()

    @classmethod
    def from_config(cls, config: AgentRLTrainerConfig) -> Self:  # type: ignore[override]
        """Create a Trainer instance from a TrainerConfig.

        Args:
            config (TrainerConfig): TrainerConfig instance containing all configuration parameters.

        Returns:
            Self: Trainer instance initialized with the provided config.
        """
        self = cls(
            load_from=config.load_from,
            pg=config.pg,
            environment_config=config.environment_config,
            dataflow_config=config.dataflow_config,
            replay_buffer_config=config.replay_buffer_config,
            train_worker_cfg=config.train_worker_cfg,
            evaluator_config=config.evaluator_config,
            tokenizer_path=config.tokenizer_path,
            work_dir=config.work_dir,
            log_dir=config.log_dir,
            total_epochs=config.total_epochs,
            auto_resume=config.auto_resume,
            load_checkpoint_cfg=config.load_checkpoint_cfg,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_maxkeep=config.checkpoint_maxkeep,
            checkpoint_no_save_optimizer=config.checkpoint_no_save_optimizer,
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            skip_checkpoint_validation=config.skip_checkpoint_validation,
            seed=config.seed,
            debug=config.debug,
            debug_rollout=config.debug_rollout,
            rollout_steps=config.rollout_steps,
            exp_tracker=config.exp_tracker,
            display_all_workers_log=config.display_all_workers_log,
            trainer_cfg=config,  # type: ignore[arg-type]
            skip_load_weights=config.skip_load_weights,
            advantage_estimator_config=config.advantage_estimator_config,
        )
        return self

    def _build_rollout_dataflow(  # type: ignore[override]
        self, environment_cfg: Dict, dataflow_cfg: DataFlowConfig, replay_buffer_config: ReplayBufferConfig
    ):
        from lagent.utils import create_object

        env = create_object(environment_cfg)
        flow = cast(ActorClass, DataFlow).remote("grpo", dataflow_cfg, replay_buffer_config, env)
        return env, flow

    def _initial_evaluate(self):
        """Performs an initial evaluation before the training loop starts."""
        if self._debug_rollout:
            return
        if self._enable_initial_evaluate and self._enable_evaluate and self._evaluator:
            ray.get(self._rollout_env_controller.update_active_workers.remote())
            scores, eval_data_groups = ray.get(self._evaluator.run.remote(return_samples=True))
            trajectory_save_path = self.exp_dir / "eval_0_trajectory.jsonl"
            self._save_trajectories(eval_data_groups, trajectory_save_path, 0, is_eval=True)
            self.logger.info(f"Initial rollout evaluate scores {scores} and start training")
            tb_scores = {f"eval/{k}": v for k, v in scores.items()}
            self._exp_tracker.add_scalars(tag_scalar_dict=tb_scores, global_step=0)
            for name, score in scores.items():
                self._eval_results[name].append((self._cur_step, score))
            self.visulize_results("eval")

    def _rollout_step(self, rollout_idx: int, step_timer_dict: dict):
        rollout_info = super()._rollout_step(rollout_idx, step_timer_dict)
        metrics_results = self._compute_metrics(rollout_info["data_groups"])
        self.logger.info(
            f"train idx {rollout_idx} scores {metrics_results['avg_reward']},"
            f" all-zero group ratio {metrics_results.pop('all_zero_ratio', None)},"
            f" all-one group ratio {metrics_results.pop('all_one_ratio', None)}"
        )
        for metric, result in metrics_results.items():
            self._train_results[metric].append((self._cur_step, result))
        self.visulize_results("train")
        return rollout_info

    def _evaluate_step(self, rollout_idx: int, step_timer_dict: dict):
        """Performs an evaluation step."""
        eval_log_info = {}
        if self._enable_evaluate and self._evaluator and rollout_idx % self._eval_step == 0:
            with timer("evaluation", step_timer_dict):
                scores, eval_data_groups = ray.get(self._evaluator.run.remote(return_samples=True))
                trajectory_save_path = self.exp_dir / f"eval_{rollout_idx}_trajectory.jsonl"
                self._save_trajectories(eval_data_groups, trajectory_save_path, rollout_idx, is_eval=True)
                self.logger.info(f"Evaluate idx {rollout_idx} scores {scores}")
            eval_log_info.update(scores)
            tb_scores = {f"eval/{k}": v for k, v in scores.items()}
            self._exp_tracker.add_scalars(tag_scalar_dict=tb_scores, global_step=rollout_idx)
            for name, score in scores.items():
                self._eval_results[name].append((self._cur_step, score))
            self.visulize_results("eval")
        return eval_log_info

    def _save_trajectories(self, data_groups, save_path, rollout_idx=None, is_eval: bool = False):
        rewards = []
        rollout_response_len_list = []
        for group in data_groups:
            for data in group:
                rewards.append(data.env.judger.reward["score"])
                if data.env.rollout.response_ids is not None:
                    if isinstance(data.env.rollout.response_ids, torch.Tensor):
                        response_ids = data.env.rollout.response_ids.flatten().tolist()
                    else:
                        response_ids = data.env.rollout.response_ids
                    rollout_response_len_list.append(len(response_ids))

        rewards_tensor = torch.tensor(rewards).float()
        rollout_response_lens = None
        if len(rollout_response_len_list) > 0:
            rollout_response_lens = torch.tensor(rollout_response_len_list).float()

        with open(save_path, "w", encoding="utf-8") as f:
            item = {
                "reward_mean": rewards_tensor.mean().item(),
                "reward_std": rewards_tensor.std().item(),
                "reward_max": rewards_tensor.max().item(),
                "reward_min": rewards_tensor.min().item(),
                "total_len": len(rewards_tensor),
            }
            if len(rollout_response_len_list) > 0 and rollout_response_lens is not None:
                item.update(
                    {
                        "rollout_response_len_mean": rollout_response_lens.mean().item(),
                        "rollout_response_len_std": rollout_response_lens.std().item(),
                        "rollout_response_len_max": rollout_response_lens.max().item(),
                        "rollout_response_len_min": rollout_response_lens.min().item(),
                    }
                )
            json.dump(item, f, ensure_ascii=False, indent=2)
            f.write("\n")
            for group in data_groups:
                for data in group:
                    entry = {
                        # "raw_prompt": data.data.extra_info["raw_prompt"],
                        "prompt": [
                            {
                                "role": msg["role"],
                                "content": msg["raw_content"] if "raw_content" in msg else msg["content"],
                            }
                            for msg in data.env.agent.extra_info.get("messages", [])[:-1]
                        ],
                        "response": data.env.rollout.response,
                        "response_len": len(data.env.rollout.response_ids or []),
                        # "label": data.data.reward_model["ground_truth"],
                        "reward": data.env.judger.reward["score"],
                        # "round": sum(msg['role'] == 'assistant' for msg in data.env.agent.extra_info['messages'][:-1]),
                        # "judger_response": data.env.judger.extra_info,
                    }
                    # if "completions" in data.env.agent.extra_info:
                    # entry["completions"] = data.env.agent.extra_info["completions"]

                    json.dump(entry, f, ensure_ascii=False, indent=2)
                    f.write("\n")

    def _compute_metrics(self, data_groups):
        def compute_reward(data_groups):
            total_groups = len(data_groups)
            zero_count = one_count = 0
            all_rewards, all_rewards_by_source = [], defaultdict(list)
            for group in data_groups:
                rewards = []
                for item in group:
                    rewards.append(item.env.judger.reward["score"])
                    all_rewards_by_source[item.data.extra_info.get('origin_data_source', 'none')].append(
                        item.env.judger.reward["score"]
                    )
                if all(r == 0 for r in rewards):
                    zero_count += 1
                elif all(r == 1 for r in rewards):
                    one_count += 1
                all_rewards.extend(rewards)

            zero_ratio = zero_count / total_groups if total_groups > 0 else 0
            one_ratio = one_count / total_groups if total_groups > 0 else 0
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
            avg_reward_by_source = {
                source: (sum(rews) / len(rews) if rews else 0) for source, rews in all_rewards_by_source.items()
            }
            return avg_reward, zero_ratio, one_ratio, avg_reward_by_source

        def compute_tool_turns(data_groups):
            tool_turns = []
            for group in data_groups:
                for data in group:
                    messages = data.env.agent.extra_info.get("messages", [])
                    tool_turn_count = sum(1 for msg in messages if msg["role"] == "tool")
                    tool_turns.append(tool_turn_count)
            avg_tool_turns = sum(tool_turns) / len(tool_turns) if tool_turns else 0
            return avg_tool_turns

        avg_reward, zero_ratio, one_ratio, avg_reward_by_source = compute_reward(data_groups)
        tool_turns = compute_tool_turns(data_groups)
        metrics_results = dict(
            avg_reward=avg_reward,
            all_zero_ratio=zero_ratio,
            all_one_ratio=one_ratio,
            avg_tool_turns=tool_turns,
            **{f"avg_reward_{source}": avg for source, avg in avg_reward_by_source.items()},
        )
        return metrics_results

    def _prepare_train_data(self, data_groups, pack_max_length, multimodal_train_infos=None):
        chat_data_groups, chat_multimodal_train_infos, agent_data_groups = [], [], []
        for j, group in enumerate(data_groups):
            # always place agent messages in the extra_info
            if "messages" in group[0].env.agent.extra_info or "inputs" in group[0].env.agent.extra_info:
                agent_data_groups.append(group)
            else:
                chat_data_groups.append(group)
                if multimodal_train_infos:
                    chat_multimodal_train_infos.append(multimodal_train_infos[j])

        data_batches, info_dict = [], {}
        if chat_data_groups:
            data_batches, info_dict = super()._prepare_train_data(
                chat_data_groups, pack_max_length, chat_multimodal_train_infos
            )
        if not agent_data_groups:
            return data_batches, info_dict

        def _tokenize_agent_messages(data_item):
            if "inputs" in data_item.env.agent.extra_info:
                return data_item.env.agent.extra_info["inputs"]
            return tokenize(
                self.tokenizer,
                data_item.env.agent.extra_info["messages"],
                tools=data_item.env.agent.extra_info.get("tools"),
            )

        with ThreadPoolExecutor(max_workers=64) as executor:
            inputs_list = list(
                executor.map(
                    _tokenize_agent_messages,
                    [group[i] for group in agent_data_groups for i in range(len(group))],
                )
            )

        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []

        # Detailed reward components for logging
        detailed_rewards = {}

        def _extract_score(value, default=0.0):
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, dict):
                score_val = value.get("score", default)
                if isinstance(score_val, (int, float)):
                    return float(score_val)
            return float(default)

        offset = 0
        for group in agent_data_groups:
            rewards = [_extract_score(data.env.judger.reward) for data in group]
            rewards_list.extend(rewards)
            # Collect detailed reward components
            for idx, data in enumerate(group):
                reward_dict = data.env.judger.reward
                for key in reward_dict.keys():
                    if key != "score":  # Skip 'score' as it's already logged separately
                        value = reward_dict[key]
                        if isinstance(value, (int, float)):
                            detailed_rewards.setdefault(key, []).append(float(value))

            rewards = torch.tensor(rewards, dtype=torch.float32)

            prompt_repeat_k = len(group)
            group_inputs = inputs_list[offset : offset + prompt_repeat_k]
            offset += prompt_repeat_k

            advantages = self._advantage_estimator.compute(rewards, group)

            for i in range(prompt_repeat_k):
                rollout = group[i].env.rollout
                inputs = group_inputs[i]
                input_ids, labels, logprobs = inputs["input_ids"], inputs["labels"], inputs["logprobs"]
                input_ids, shifted_labels, logprobs = input_ids[:-1], labels[1:], logprobs[1:]

                response_len_list.append(len(rollout.response_ids))
                prompt_len_list.append(len(input_ids) - len(rollout.response_ids))
                advantages_list.extend([advantages[i]] * len(rollout.response_ids))
                assert (
                    len(input_ids) <= pack_max_length
                ), f"Input ids length {len(input_ids)} exceed pack max length {pack_max_length}."
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)
                rollout_logprobs = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
                assert (
                    rollout_logprobs.size() == shifted_labels.size()
                ), f"{rollout_logprobs.size()} vs {shifted_labels.size()}"

                seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
                seq_ctx.rollout_routed_experts = inputs["routed_experts"]
                data_batches.append(
                    dict(
                        seq_ctx=seq_ctx,
                        shifted_labels=shifted_labels,
                        advantage=advantages[i].item(),
                        rollout_logprobs=rollout_logprobs,
                    )
                )
        random.shuffle(data_batches)
        info_dict.update(
            {
                "agent/batch_size": len(rewards_list),
                "agent/rewards/mean": np.mean(rewards_list),
                "agent/rewards/min": np.min(rewards_list),
                "agent/rewards/max": np.max(rewards_list),
                "agent/advantages/mean": np.mean(advantages_list),
                "agent/advantages/min": np.min(advantages_list),
                "agent/advantages/max": np.max(advantages_list),
                "agent/response_len/mean": np.mean(response_len_list),
                "agent/response_len/min": np.min(response_len_list),
                "agent/response_len/max": np.max(response_len_list),
                "agent/response_len/std": np.std(response_len_list),
                "agent/prompt_len/mean": np.mean(prompt_len_list),
                "agent/prompt_len/min": np.min(prompt_len_list),
                "agent/prompt_len/max": np.max(prompt_len_list),
            }
        )
        return data_batches, info_dict

    def visulize_results(self, stage: Literal["all", "train", "eval"] = "all"):
        def plot_accuracy_curve(data_list, x_label="training_steps", y_label="accuracy", save_path=None):
            """绘制折线图，输入的 data_list 是一个列表，元素为 (x, y) 元组， 其中 x 是横坐标，y 是纵坐标。

            Args:
                data_list (list of tuple): [(x1, y1), (x2, y2), ...]，要求按 x 升序排列
                x_label (str): 横坐标标签
                y_label (str): 纵坐标标签
                save_path (str): 图片保存路径，如果为 None 则不保存
            """
            # 拆分横纵坐标
            x_values = [x for x, _ in data_list]
            y_values = [y for _, y in data_list]

            plt.figure(figsize=(8, 5))
            plt.plot(x_values, y_values, marker="o", linestyle="-", linewidth=2)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"{y_label} vs {x_label}")
            plt.grid(True)

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图片已保存到: {save_path}")

        if stage in ["all", "eval"]:
            for metric, results in self._eval_results.items():
                plot_accuracy_curve(results, y_label=f"eval_{metric}", save_path=self.exp_dir / f"eval_{metric}.png")
        if stage in ["all", "train"]:
            for metric, results in self._train_results.items():
                plot_accuracy_curve(results, y_label=f"train_{metric}", save_path=self.exp_dir / f"train_{metric}.png")
