# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

# set -x
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# export PYTHONPATH="$(pwd)"

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=data/gsm8k/train.parquet \
#     data.val_files=data/gsm8k/test.parquet \
#     data.train_batch_size=1024 \
#     data.max_prompt_length=512 \
#     data.max_response_length=1024 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/cpfs01/shared/llm_razor/huanghaian/new_model/Qwen3-8B \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.actor.strategy="fsdp2" \
#     actor_rollout_ref.model.use_fused_kernels=True \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console"]' \
#     trainer.project_name='verl_grpo_example_gsm8k' \
#     trainer.experiment_name='qwen3_8b_function_rm' \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=200 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 \
#     2>&1 | tee -a "outputs/gsk8k_grpo_0818_training_log.txt"


import os
import re
from pathlib import Path
import ray
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist

from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    FSDPConfig,
    LRConfig,
    BalancingLossConfig,
    ZLossConfig
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.model.dense.qwen3 import Qwen3_8BConfig
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig
from xtuner.v1.datasets import RLTextTokenizeFnConfig
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    Float8Config,
    FSDPConfig,
    LRConfig,
    BalancingLossConfig,
    ZLossConfig,
)
from xtuner.v1.rl.grpo.config import WorkerConfig, LossConfig
from xtuner.v1.rl.grpo.trainer import Trainer
from xtuner.v1.ray.dataflow.flow import SampleParams


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]


def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Rollout Test Script")
    parser.add_argument("--total-epochs", type=int)
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--rollout-global-batch-size", type=int, default=128)
    parser.add_argument("--train-optimizer-steps", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--prompt-repeat-k", type=int, default=8)
    parser.add_argument("--pack-max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--optimizer-disable-foreach", action="store_true")  # save memory usage during opt.step()
    return parser.parse_args()


def main(args):
    ray.init(num_cpus=128, ignore_reinit_error=True)
    load_from = args.model_path
    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_accelerators_per_worker=1,
        num_cpus_per_worker=12,
        num_workers=args.num_workers,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    rollout_config = RolloutConfig(
        env="test_env",
        model_path=args.model_path,
        model_name=os.path.basename(args.model_path).lower(),
        tokenizer_path=args.model_path,
        rollout_cross_node_comm=False,
        max_running_requests=16,
        tensor_parallel_size=8,
        expert_parallel_size=1,
        gpus_per_node=args.gpus_per_node, # gpu: 8, npu: 16
        dtype="bfloat16",
        skip_load_weights=False,
    )
    dataflow_config = DataFlowConfig(
        env="test",
        max_concurrent=args.max_concurrent,
        prompt_repeat_k=args.prompt_repeat_k,
        global_batch_size=args.rollout_global_batch_size,
        sample_params=SampleParams(max_tokens=args.max_response_length),
    )
    judger_config = {"judger_type": "xtuner.v1.ray.judger.gsm8k.GSM8KJudgerWorker"}
    dataset_cfg = [
        {
        "dataset": DatasetConfig(name="gsm8k",
                                 anno_path=DATA_PATH,
                                 sample_ratio=1.0),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=args.max_prompt_length + 2),
        },
    ]
    dataloader_cfg = DataloaderConfig(
        pack_max_length=args.pack_max_length,
        collator='fake_collator',
        pack_level='none',
    )
    train_worker_cfg: WorkerConfig = WorkerConfig(
        model_cfg=Qwen3_8BConfig(),
        optim_cfg=AdamWConfig(lr=1e-6, foreach=False if args.optimizer_disable_foreach else None),
        loss_cfg=LossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.28,
                cliprange_low=0.2,
                loss_type="vanilla",
            ),
            ignore_idx=-100,
            use_kl_loss=True,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl",
            mode="chunk", 
            chunk_size=512),
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
        fsdp_cfg=FSDPConfig(
            torch_compile=False,
            cpu_offload=False,
            ep_size=1,
        ),
        load_from=args.model_path,
        sp_size=1,
        optimizer_steps=args.train_optimizer_steps,
        pack_max_length=args.pack_max_length,
    )
    trainer = Trainer(
        load_from=load_from,
        resources=resources,
        rollout_config=rollout_config,
        dataflow_config=dataflow_config,
        judger_config=judger_config,
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        train_worker_cfg=train_worker_cfg,
        tokenizer_path=args.model_path,
        work_dir=args.work_dir,
        total_epochs=args.total_epochs,
    )
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()
    main(args)
