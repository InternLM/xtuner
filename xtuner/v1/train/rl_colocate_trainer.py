import asyncio
import json
import os
import random
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import List, cast, Any

import ray
import torch
from mmengine import load
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, model_validator
from ray.util.placement_group import placement_group
from typing_extensions import Literal, Self, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1._writer import get_writer
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers, CPUResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.rl.base import (
    TrainingController,
    TrainingControllerProxy,
    TrainingWorkerClass,
    TrainingWorkerProxy,
    WorkerConfig,
    WorkerLogItem,
)
from xtuner.v1.rl.base import TrainingWorker as BaseTrainingWorker
from xtuner.v1.train import ResumeConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger, is_hf_model_path, record_git_info, timer
from xtuner.v1.utils.device import get_device, get_torch_device_module
from xtuner.v1.utils.env_check import get_rollout_engine_version

from xtuner.v1.train.trainer import ExpHistory, ExpInfo, GitInfo, LoadCheckpointConfig, XTunerMeta

from xtuner.v1.data_proto import RolloutState, Status, SampleParams 
from xtuner.v1.rl.base.agent_loop import SingleTurnAgentLoop, AgentLoop
from xtuner.v1.rl.base.agent_loop_manager import AgentLoopManager
from xtuner.v1.rl.base.producer import SyncProduceStrategy, Sampler
from xtuner.v1.rl.base.replay_buffer import ReplayBuffer
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig, NativeJudgerConfig


# TODO: Move DEVICE to `xtuner.utils.device`
PG_READY_TIMEOUT = 30
TRAINER_RAY_GET_TIMEOUT = 5 * 3600  # 5 hour
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def bind_train_rollout(
    train_controller,
    rollout_controller,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(rollout_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return


class TrainInfo(TypedDict):
    data_info: dict[str, float]
    workers_log_item: list[WorkerLogItem]


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


def is_valid_for_training(group_data_items: list[RolloutState], logger) -> bool:
    """Checks if a group of rollout states is valid for a training step.

    Args:
        group_data_items: A list of RolloutState objects.

    Returns:
        True if the group is valid, False otherwise.

    NOTE: Why this check is needed:
    - For system fault tolerance, this check is performed at rollout / dataflow
      time, but we still do it here to ensure training data integrity.
    - 'filtered'/'failed': These items are fundamentally broken or incomplete and
      should not be used for training.
    - 'aborted': These items represent rollouts that were stopped
      prematurely. Using such partial data could lead the model to learn
      undesirable behaviors (e.g., stopping generation too early).
    - Empty response/response_ids: The model's generated response is the core
      of the training data for RL algorithms like PPO. If the response is
      missing, there is nothing to compute rewards on or to train the model with.
    """
    is_abort = any(item.status == Status.ABORTED for item in group_data_items)
    is_filtered = any(item.status == Status.FILTERED for item in group_data_items)
    is_failed = any(item.status == Status.FAILED for item in group_data_items)
    if is_filtered or is_failed or is_abort:
        logger.warning(
            f"Invalid dataflow group found during training, rollout state filtered: {is_filtered}, failed: {is_failed}, aborted: {is_abort}."
        )
        return False
    for item in group_data_items:
        response_valid = item.response is not None and len(item.response) > 0
        ids_valid = item.response_ids is not None and len(item.response_ids) > 0
        if not ids_valid:
            # NOTE: `response_ids` is the critical field for token-in-token-out mode, so we ensure it's not empty.
            logger.warning(
                "Invalid dataflow item found during training: no response or response_ids and skip this item."
            )
            return False
        if not response_valid:
            # NOTE: check valid response string for judger inputs
            logger.warning("Invalid dataflow item found during training: empty response string and skip this item.")
            return False
    return True


class RLColocateTrainer:
    # 弱化Trainer：Trainer中代码尽量少，尽量用componet来组织代码。
    # 目标是像torch一样，让用户自己写init 和 train loop，我们只提供组件。
    def __init__(
        self,
        *,
        resources: AcceleratorResourcesConfig,
        train_worker_cfg: WorkerConfig,
        rollout_config: RolloutConfig,
        judger_config: NativeJudgerConfig,

        # Sampler config
        dataloader_cfg: DataloaderConfig,
        tokenizer_path: str | Path,
        # agent loop config
        sample_params: SampleParams,

        # others
        load_from: str | Path,
        log_dir: Path | str,
        seed: int = 66,

        rollout_steps: int,
        global_batch_size: int,
    ):
        # log
        log_dir = Path(log_dir)
        self.logger = get_logger(log_dir=log_dir, tag="RLTrainer")

        # steps
        self._rollout_steps = rollout_steps
        # self._total_epochs = total_epochs
        self._cur_step = 0
        self._global_train_step = 1
        self.global_batch_size = global_batch_size

        # main components    
        self._pg = AutoAcceleratorWorkers.build_placement_group(resources)

        # override train worker config
        if train_worker_cfg.seed is None:
            self.logger.warning(f"RLTrainer seed {seed} is used as train worker seed.")
            train_worker_cfg.seed = seed
        train_worker_cfg.load_from = load_from
        train_worker_cfg.log_dir = log_dir
        self._train_worker_cfg = train_worker_cfg

        # override rollout config
        rollout_config.worker_log_dir = log_dir

        # build train controller and rollout controller
        self.train_controller = self._build_train_controller(train_worker_cfg)

        self.rollout_controller = self.init_rollout_controller(rollout_config, self._pg)

        # build judger
        judger = judger_config.build_router()
        # TODO: build agent_loop
        # sample_params = SampleParams(max_tokens=self.max_response_length, temperature=0.0)
        hf_checkpoint: str = load_from if isinstance(load_from, str) else load_from.as_posix()
        agent_loop: AgentLoop = SingleTurnAgentLoop(self.rollout_controller, sample_params, hf_checkpoint, judger)
        # TODO: build produce_strategy
        stragegy = SyncProduceStrategy()
        # TODO: build sampler
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        sampler = Sampler(dataloader_cfg=dataloader_cfg, tokenizer=self.tokenizer, prompt_repeat_k=8)
        # TODO: build replay_buffer
        replay_buffer = ReplayBuffer()
        # TODO: build agnet_loop_manager
        self.agent_loop_manager = AgentLoopManager(agent_loop=agent_loop, produce_strategy=stragegy, sampler=sampler, replay_buffer=replay_buffer, task_name="test_gsm8k")

    
    # TODO: simplify with WorkerConfig.build()
    def _build_train_controller(self, train_worker_cfg: WorkerConfig) -> TrainingControllerProxy:
        TrainingWorker = cast(
            TrainingWorkerClass,
            ray.remote(
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                    }
                },
            )(BaseTrainingWorker),
        )
        train_workers: list[TrainingWorkerProxy]
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorker, train_worker_cfg, self._pg)
        ray.wait([worker.ready.remote() for worker in train_workers])
        train_controller = TrainingController.remote(workers=train_workers)
        return train_controller
    
    # TODO: simplify with RolloutConfig.build()
    def init_rollout_controller(self, rollout_cfg: Any, placement_group: Any):
        """Initializes the rollout controller with the appropriate worker
        backend.

        Based on the `rollout_cfg`, this method selects and initializes the corresponding
        rollout worker (e.g., `LMDeployWorker` or `vLLMWorker`). It then creates and
        returns a `RolloutController` to manage these workers.

        Args:
            rollout_cfg (Any): The configuration for the rollout controller.
            placement_group (Any): The placement group for scheduling Ray actors.

        Returns:
            The initialized rollout controller, or None if `rollout_cfg` is not provided.

        Raises:
            NotImplementedError: If the specified rollout backend is not supported.
        """

        rollout_controller = None
        if rollout_cfg is None:
            return rollout_controller

        from xtuner.v1.ray.rollout.controller import RolloutController

        rollout_controller = (
            ray.remote(RolloutController)
            .options(max_concurrency=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)))
            .remote(rollout_cfg, placement_group)
        )  # type: ignore[attr-defined]
        return rollout_controller
    
    def fit(self):
        self.logger.info("Start RL training")
        if self._cur_step >= self._rollout_steps:
            self.logger.info(f"Rollout steps {self._rollout_steps} reached, stop training")
            return
        
        for rollout_idx in range(self._cur_step + 1, self._rollout_steps + 1):
            self.logger.info(f"Rollout {rollout_idx}/{self._rollout_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                # 1. Rollout to generate experience
                # rollout_info = self._rollout_step(rollout_idx, step_timer_dict)
                train_batch: list[list[RolloutState]] = asyncio.run(self.agent_loop_manager.produce_batch(self.global_batch_size))
                rollout_info = {}

                if not self._debug_rollout:
                    with timer("onload", step_timer_dict):
                        ray.get(self.train_controller.onload.remote(target="all"))
                        self.logger.info("Training controller loaded")

                    # 2. Train on the generated experience
                    # TODO: simplify with Packer.pack_pad_dispatch()
                    # train_batch = Packer.pack_pad_dispatch(train_batch)
                    with timer("prepare_data", step_timer_dict):
                        data_batches, data_info = self._prepare_train_data(
                            train_batch, self._train_worker_cfg.pack_max_length
                        )
                    self.logger.info(f"Prepared {len(data_batches)} training data batches")


                    # train_log_info = self._train_step(
                    #     rollout_idx,
                    #     rollout_info["data_groups"],
                    #     rollout_info["multimodal_train_infos"],
                    #     step_timer_dict,
                    # )
                    # metrics = self.train_controller.fit(train_batch)
                    with timer("training", step_timer_dict):
                        workers_log_item: list[WorkerLogItem] = ray.get(
                            self.train_controller.fit.remote(
                             data_batches, pack_max_length=self._train_worker_cfg.pack_max_length, rollout_idx=rollout_idx
                            )
                        )
                    train_log_info: TrainInfo = {
                        "data_info": data_info,
                        "workers_log_item": workers_log_item,
                    }

                    # 3. Synchronize weights and save checkpoints
                    self._sync_weights_and_save(rollout_idx, step_timer_dict)

                    # 4. Evaluate model performance
                    # eval_log_info = self._evaluate_step(rollout_idx, step_timer_dict)
                    eval_log_info = {}
                    # eval_batch: list[RolloutState] = asyncio.run(self.agent_loop_manager.produce_batch(self.eval_batch_size))
                    # eval_metrics = self.evaluator.evaluate(eval_batch)

            self._log_step(rollout_idx, step_timer_dict, rollout_info, train_log_info, eval_log_info)
            self._cur_step = rollout_idx

    def _prepare_train_data(self, data_groups: list[list[RolloutState]], pack_max_length: int):
        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []

        data_batches = []

        for j, group in enumerate(data_groups):
            if not is_valid_for_training(group, self.logger):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue

            prompt_ids = group[0].prompt_ids
            rewards = [
                data.reward["score"] for data in group
            ]
            rewards_list.extend(rewards)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

            prompt_repeat_k = len(group)
            for i in range(prompt_repeat_k):
                item = group[i].response
                logprobs = None
                if group[i].response_ids is not None:
                    response_ids = group[i].response_ids
                    if isinstance(response_ids, torch.Tensor):
                        response_ids = response_ids.flatten().tolist()
                    logprobs = group[i].logprobs
                    if logprobs is not None:
                        assert len(logprobs) == len(response_ids), f"{len(logprobs)} vs {len(response_ids)}"
                        # 只有 response 部分有 logprobs, 需要前面追加
                        logprobs = [0] * (len(prompt_ids) - 1) + logprobs
                    else:
                        logprobs = None
                else:
                    response_ids = self.tokenizer(item, return_tensors="pt")["input_ids"].flatten().tolist()
                # 返回的 routed_experts 不包括 eos 的值，实际上也不需要，需要减一
                input_ids = prompt_ids + response_ids[:-1]

                prompt_len_list.append(len(prompt_ids))
                response_len_list.append(len(response_ids))
                advantages_list.extend([advantages[i]] * len(response_ids))

                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids
                assert len(input_ids) <= pack_max_length, f"{len(input_ids)} vs {pack_max_length}"
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                if logprobs is not None:
                    rollout_logprobs = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
                    assert rollout_logprobs.size() == shifted_labels.size(), (
                        f"{rollout_logprobs.size()} vs {shifted_labels.size()}"
                    )
                else:
                    rollout_logprobs = None

                multimodal_train_info = group[i].mm_info
                seq_ctx = get_train_seq_ctx(input_ids, multimodal_train_info, len(response_ids) - 1)
                data_dict = {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels,
                    "advantage": advantages[i].item(),
                    "rollout_logprobs": rollout_logprobs,
                }

                seq_ctx.rollout_routed_experts = group[i].routed_experts  # n,layer*expert

                data_batches.append(data_dict)
        random.shuffle(data_batches)

        rewards_t = torch.tensor(rewards_list).float() if rewards_list else torch.tensor([0.0]).float()
        advantages_t = torch.tensor(advantages_list).float() if advantages_list else torch.tensor([0.0]).float()
        prompt_len_t = torch.tensor(prompt_len_list).float() if prompt_len_list else torch.tensor([0.0]).float()
        response_len_t = torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()

        info_dict = {
            "batch_size": len(rewards_list),
            "rewards/mean": rewards_t.mean().item(),
            "rewards/min": rewards_t.min().item(),
            "rewards/max": rewards_t.max().item(),
            "advantages/mean": advantages_t.mean().item(),
            "advantages/min": advantages_t.min().item(),
            "advantages/max": advantages_t.max().item(),
            "response_len/mean": response_len_t.mean().item(),
            "response_len/min": response_len_t.min().item(),
            "response_len/max": response_len_t.max().item(),
            "response_len/std": response_len_t.std().item(),
            "prompt_len/mean": prompt_len_t.mean().item(),
            "prompt_len/min": prompt_len_t.min().item(),
            "prompt_len/max": prompt_len_t.max().item(),
        }
        return data_batches, info_dict
    
    def _sync_weights_and_save(self, rollout_idx: int, step_timer_dict: dict):
        """Synchronizes weights and saves checkpoints."""
        with timer("save_ckpt", step_timer_dict):
            ray.get(self.train_controller.offload.remote(target="optimizer"))
            # TODO:
            # self._maybe_save_hf()
            # self._maybe_save_checkpoint()

        with timer("sync_weight", step_timer_dict):
            bind_train_rollout(train_controller=self.train_controller, env_controller=self.rollout_controller)
            ray.get(self.rollout_controller.onload_weights.remote())
            ray.get(self.train_controller.update_weights.remote())
            self.logger.info("Model weights synchronized successfully.")
            ray.get(self.train_controller.offload.remote(target="model"))
            ray.get(self.rollout_controller.onload_kvcache.remote())


if __name__ == "__main__":
    if not ray.is_initialized():
        if os.getenv("RAY_MASTER_ADDR"):
            master_addr = os.getenv("RAY_MASTER_ADDR", "127.0.0.1")
            client_port = os.getenv("RAY_CLIENT_PORT", "10001")
            ray_head_address = f"ray://{master_addr}:{client_port}"
            ray.init(address=ray_head_address)
        else:
            ray.init(num_cpus=128)

    model_path = os.environ["MODEL_PATH"]
    enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", '0')
    data_path = os.environ["DATA_PATH"]
    log_dir = os.environ["WORK_DIR"]  # TODO: work_dir

    # total_epochs = 3  # 5000
    rollout_steps = 100  # 5000
    train_optimizer_steps = 1  # 16
    global_batch_size = 64 * train_optimizer_steps  # 512
    prompt_repeat_k = 5  # 16
    rollout_tp_size = 1  # 2
    rollout_ep_size = 1
    max_prompt_length = 512  # 2048
    max_response_length = 1024  # 20*1024  # 8192
    pack_max_length = 32*1024  # 1*(max_prompt_length + max_response_length)  # 32768

    from xtuner.v1.config import LRConfig
    from xtuner.v1.config import FSDPConfig
    from xtuner.v1.config import AdamWConfig
    from xtuner.v1.rl.grpo import GRPOLossConfig
    from xtuner.v1.model import get_model_config_from_hf
    from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
    import torch.distributed as dist

    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}.")
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    exp_name = 'test_gsm8k'

    # resources
    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=8 * WORLD_SIZE,
        num_cpus_per_worker=12,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )

    # rollout config
    rollout_config = RolloutConfig(
        env=exp_name,
        device=resources.accelerator,
        model_path=model_path,
        dtype="bfloat16",
        tensor_parallel_size=rollout_tp_size,
        expert_parallel_size=rollout_ep_size,
        gpu_memory_utilization=0.8,
        context_length = max_response_length + max_prompt_length,
        # rollout_max_batch_size_per_instance=512,
        enable_return_routed_experts=True if enable_return_routed_experts == "1" else False,
    )

    # judger config
    judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")

    # worker config
    lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
    fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)

    model_cfg = get_model_config_from_hf(Path(model_path))
    if hasattr(model_cfg, "balancing_loss_cfg"):
        model_cfg.balancing_loss_cfg = None
    if hasattr(model_cfg, "z_loss_cfg"):
        model_cfg.z_loss_cfg = None
    optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1)
    
    loss_cfg = GRPOLossConfig(
        policy_loss_cfg=dict(
            cliprange_high=0.28,  # TODO: 0.2
            cliprange_low=0.2,
            loss_type=os.environ.get("LOSS_TYPE", "vanilla"),
            clip_ratio_c=10.0,
            log_prob_diff_min=-20.0,
            log_prob_diff_max=20.0,
        ),
        ignore_idx=-100,
        use_kl_loss=False,  # TODO: True
        kl_loss_coef=0.0,  # TODO: 0.001
        kl_loss_type="low_var_kl",
        mode=os.environ.get("LOSS_MODE", "chunk"),
        chunk_size=512,
    )
    train_worker_cfg: WorkerConfig = WorkerConfig(
        model_cfg=model_cfg,
        load_from=model_path,
        optim_cfg=optim_cfg,
        loss_cfg=loss_cfg,
        lr_cfg=lr_cfg,
        fsdp_cfg=fsdp_cfg,
        sp_size=int(os.environ.get("SP_SIZE", "1")),
        optimizer_steps=train_optimizer_steps,
        pack_max_length=pack_max_length,
    )

    # dataloader config
    train_dataset = DatasetConfig(name=exp_name, anno_path=data_path)
    tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)

    train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
    dataloader_cfg = DataloaderConfig(
        dataset_config_list=train_dataset_cfg,
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    )

    # agent loop config
    sample_params = SampleParams(
        max_tokens=max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    )

    # Finally, build the trainer
    trainer = RLColocateTrainer(
        resources=resources,
        train_worker_cfg=train_worker_cfg,
        rollout_config=rollout_config,
        judger_config=judger_config,

        dataloader_cfg=dataloader_cfg,
        tokenizer_path=model_path,

        sample_params=sample_params,

        load_from=model_path,
        log_dir=log_dir,
        seed=123,

        rollout_steps=rollout_steps,
        global_batch_size=global_batch_size,
    )

    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()
