import os
import re
from pathlib import Path
import ray
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from transformers import AutoTokenizer

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
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.rollout import SampleParams
from xtuner.v1.ray.evaluator import EvaluatorConfig
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
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.rl.grpo.config import WorkerConfig, LossConfig
from xtuner.v1.rl.grpo.trainer import Trainer

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Rollout Test Script")
    parser.add_argument("--total-epochs", type=int)
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=TRAIN_DATA_PATH)
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
    parser.add_argument("--policy-loss-type", type=str, default="vanilla")
    parser.add_argument("--enable-evaluate", action="store_true")
    parser.add_argument("--evaluate-step", type=int, default=1)
    parser.add_argument("--evaluate-ratio", type=float, default=1)
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
        tensor_parallel_size=2,
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
    from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
    gsm8k_judger_config = GSM8KJudgerConfig()
    judger_cfg = JudgerConfig(
        reward_judger_configs={"openai/gsm8k": gsm8k_judger_config}
    )
    train_dataset_cfg = [
        {
        "dataset": DatasetConfig(name="gsm8k",
                                 anno_path=TRAIN_DATA_PATH,
                                 sample_ratio=1.0),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=args.max_prompt_length),
        },
    ]
    eval_dataset_cfg = [
        {
        "dataset": DatasetConfig(name="gsm8k",
                                 anno_path=TEST_DATA_PATH,
                                 sample_ratio=1.0),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=args.max_prompt_length),
        },
    ]
    dataloader_cfg = DataloaderConfig(
        pack_max_length=args.pack_max_length,
        collator='fake_collator',
        pack_level='none',
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    evaluator_cfg = EvaluatorConfig(
        dataset_cfg=eval_dataset_cfg,
        tokenizer=tokenizer, 
        max_concurrent=args.max_concurrent,
        eval_sample_ratio=args.evaluate_ratio, 
        evaluate_step=args.evaluate_step,
        compute_metric_func=None
    )
    replay_buffer_cfg = ReplayBufferConfig(
        dataset_cfg=train_dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        tokenizer=tokenizer,
        postprocessor=None
    )
    train_worker_cfg: WorkerConfig = WorkerConfig(
        model_cfg=Qwen3Dense8BConfig(),
        optim_cfg=AdamWConfig(lr=1e-6, foreach=False if args.optimizer_disable_foreach else None),
        loss_cfg=LossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.2,
                cliprange_low=0.2,
                loss_type=args.policy_loss_type,
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
        judger_config=judger_cfg,
        replay_buffer_config=replay_buffer_cfg,
        evaluator_config=evaluator_cfg,
        train_worker_cfg=train_worker_cfg,
        tokenizer_path=args.model_path,
        work_dir=args.work_dir,
        total_epochs=args.total_epochs,
        enable_evaluate=args.enable_evaluate
    )
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()
    main(args)
