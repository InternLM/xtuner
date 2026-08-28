"""Intern-S2-Preview text RL (DAPO + GSM8K) example.

Validated ETE settings: rollout TP1/EP4, AsyncProduce + AsyncReplay, MTP4,
skip_load_weights, freeze routers, vocab_size=251392.
Uses Qwen3_5_VLMoE35BA3Config as the 35B-A3B MoE backbone for text RL.

Required env:
  WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH

Recommended env:
  export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1
    # Intern-S2 / Qwen3.5 use GatedDeltaNet (Mamba/SSM). By default LMDeploy stores the
    # recurrent state in the activation dtype (bf16/fp16). Setting this to 1 forces
    # FP32 recurrent state for better numerical stability (roughly 2x recurrent-state
    # memory). Keep this enabled for RL to reduce mismatch/kl drift.
  export XTUNER_USE_LMDEPLOY=1
  export XTUNER_USE_FA3=1
  export PERMUTE_COMPUTE_DTYPE=fp32

Launch::

  bash examples/v1/scripts/run_rl.sh \\
    examples/v1/config/rl_interns2_preview_gsm8k_dapo.py \\
    lmdeploy $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
"""

import os

import ray

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    SamplerConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig

# Strongly recommended before launch (see module docstring):
#   export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

# basic settings (validated by ETE)
experimental_name = "interns2_preview_gsm8k_dapo"
total_train_steps = 15
evaluate_step = 15
train_optimizer_steps = 1
train_batch_size = 64 * train_optimizer_steps
prompt_repeat_k = 5
rollout_tp_size = 1
rollout_ep_size = 4
max_prompt_length = 512
max_response_length = 1024
pack_max_length = 32 * 1024

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * NNODE,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    skip_load_weights=True,
    gpu_memory_utilization=0.8,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=True,
    extra_rollout_config=dict(
        lmdeploy_trust_remote_code=True,
        lmdeploy_log_level="INFO",
        lmdeploy_uvicorn_log_level="INFO",
    ),
)

# 3. judger
judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)

# 4. train worker
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
model_cfg = Qwen3_5_VLMoE35BA3Config()
model_cfg.float8_cfg = None
model_cfg.text_config.ep_size = 1
model_cfg.text_config.z_loss_cfg = None
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.freeze_routers = True
model_cfg.text_config.mtp_config = MTPConfig(
    num_layers=4,
    loss_scaling_factor=1.0,
    detach_mtp_lm_head_weight=True,
    detach_mtp_inputs=True,
    share_weights=True,
)
model_cfg.text_config.vocab_size = 251392
optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.28,
        cliprange_low=0.2,
        loss_type=os.environ.get("LOSS_TYPE", "vanilla"),
        clip_ratio_c=10.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=512,
)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# 5. train agent loop manager
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)
dataloader_cfg = DataloaderConfig(
    dataset_config_list=[{"dataset": train_dataset, "tokenize_fn": tokenizer_config}],
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
    return_routed_experts=True,
)


def group_sample_filter_func(group_samples):
    valid_samples = []
    for s in group_samples:
        if s.response_ids is not None:
            valid_samples.append(s)
        elif s.routed_experts is not None and isinstance(s.routed_experts, ray.ObjectRef):
            ray.internal.free([s.routed_experts], local_only=False)

    rewards = [(d.reward or {}).get("score", 0.0) for d in valid_samples]
    if len(set(rewards)) == 1:
        print(f"filter all same reward sample: {rewards}")
        return False
    return True


agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=training_sample_params,
        ),
        judger_config=judger_config,
        is_valid_sample_fn=group_sample_filter_func,
        produce_strategy_config=AsyncProduceStrategyConfig(
            over_sample_threshold=1,
            enable_partial_rollout=1,
            max_staleness=1000000,
        ),
        sampler_config=SamplerConfig(
            dataloader_cfg=dataloader_cfg,
            prompt_repeat_k=prompt_repeat_k,
        ),
    ),
)

# 6. eval agent loop manager
eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path, sample_ratio=1.0)
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}],
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=1,
    top_p=1.0,
    temperature=0.0,
    min_tokens=0,
    return_routed_experts=False,
)
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="eval_task",
        agent_loop_config=SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=evaluation_sample_params,
        ),
        judger_config=judger_config,
        sampler_config=SamplerConfig(
            dataloader_cfg=eval_dataloader_cfg,
            prompt_repeat_k=1,
        ),
    ),
)

# 7. trainer
trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=AsyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=EvaluatorConfig(compute_metric_func=None),
    load_from=model_path,
    total_train_steps=total_train_steps,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
    exp_tracker="jsonl",
)
