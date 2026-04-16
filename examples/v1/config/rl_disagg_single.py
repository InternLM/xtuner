"""RL Disaggregated Trainer example config (GRPO + GSM8K).

This config uses a mocked Disaggregated weight-sync hook until the real cross-device weight update module lands.

Required env vars: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
Common optional env vars:
  TRAIN_NUM_WORKERS=4, ROLLOUT_NUM_WORKERS=4, TRAIN_BATCH_SIZE=64,
  TOTAL_TRAIN_STEPS=45, SYNC_WEIGHTS_INTERVAL=1,
  OVER_SAMPLE_THRESHOLD=0.0, PARTIAL_ROLLOUT=0,
  TAIL_BATCH_TRIGGER_SIZE=0, TAIL_BATCH_STALE_THRESHOLD=0, ENABLE_EVALUATE=0

Mode mapping in the current design:
  Mode 1 (On-Policy):
    SYNC_WEIGHTS_INTERVAL=1
    OVER_SAMPLE_THRESHOLD=0.0
    PARTIAL_ROLLOUT=0
  Mode 2 (Stream Off-Policy):
    SYNC_WEIGHTS_INTERVAL>1
    OVER_SAMPLE_THRESHOLD=0.0
    PARTIAL_ROLLOUT=0
  Mode 3 (Async Stale):
    OVER_SAMPLE_THRESHOLD>0.0
    PARTIAL_ROLLOUT=0
  Mode 4 (Async Partial Rollout):
    OVER_SAMPLE_THRESHOLD>0.0
    PARTIAL_ROLLOUT=1

Responsibility split:
  - trainer / step scheduling:
      TRAIN_BATCH_SIZE, TOTAL_TRAIN_STEPS, SYNC_WEIGHTS_INTERVAL
  - producer / replay-buffer policy:
      OVER_SAMPLE_THRESHOLD, PARTIAL_ROLLOUT,
      TAIL_BATCH_TRIGGER_SIZE, TAIL_BATCH_STALE_THRESHOLD
"""

import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.agent_loop import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    SamplerConfig,
    SingleTurnAgentLoopConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_disaggregated_trainer import (
    RLDisaggregatedTrainerConfig,
)


# env
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")


# basic settings
experimental_name = "disaggregated_grpo_gsm8k"
total_train_steps = int(os.environ.get("TOTAL_TRAIN_STEPS", "16"))
evaluate_step = int(os.environ.get("EVALUATE_STEP", str(total_train_steps)))
train_optimizer_steps = int(os.environ.get("TRAIN_OPTIMIZER_STEPS", "1"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", str(32 * train_optimizer_steps)))
sync_weights_interval = int(os.environ.get("SYNC_WEIGHTS_INTERVAL", "1"))
over_sample_threshold = float(os.environ.get("OVER_SAMPLE_THRESHOLD", "0.0"))
partial_rollout = os.environ.get("PARTIAL_ROLLOUT", "0") == "1"
tail_batch_trigger_size = int(os.environ.get("TAIL_BATCH_TRIGGER_SIZE", "0"))
tail_batch_stale_threshold = int(os.environ.get("TAIL_BATCH_STALE_THRESHOLD", "0"))
prompt_repeat_k = int(os.environ.get("PROMPT_REPEAT_K", "4"))
rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", "1"))
rollout_ep_size = int(os.environ.get("ROLLOUT_EP_SIZE", "1"))
max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
max_response_length = int(os.environ.get("MAX_RESPONSE_LENGTH", "1024"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", str(32 * 1024)))
enable_evaluate = os.environ.get("ENABLE_EVALUATE", "0") == "1"

# execution knobs:
# - sync_weights_interval controls how many train steps share one weight-sync interval
# - over_sample_threshold / partial_rollout feed the train-task produce strategy
# - tail_batch_* controls replay-buffer recycling policy inside AsyncProduceStrategy


# 1. resources: default 4 GPUs for training and 4 GPUs for rollout.
train_resources = AcceleratorResourcesConfig(
    accelerator=os.environ.get("ACCELERATOR", "GPU"),
    num_workers=int(os.environ.get("TRAIN_NUM_WORKERS", "4")),
    num_cpus_per_worker=float(os.environ.get("TRAIN_CPUS_PER_WORKER", "12")),
    cpu_memory_per_worker=int(os.environ.get("TRAIN_CPU_MEMORY_PER_WORKER", str(16 * 1024**3))),
)

rollout_resources = AcceleratorResourcesConfig(
    accelerator=os.environ.get("ACCELERATOR", "GPU"),
    num_workers=int(os.environ.get("ROLLOUT_NUM_WORKERS", "4")),
    num_cpus_per_worker=float(os.environ.get("ROLLOUT_CPUS_PER_WORKER", "12")),
    cpu_memory_per_worker=int(os.environ.get("ROLLOUT_CPU_MEMORY_PER_WORKER", str(16 * 1024**3))),
)


# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=rollout_resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.8")),
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=(enable_return_routed_experts == "1"),
)


# 3. judger
judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k", judger_type="router")


# 4. train worker
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
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)


# 5. train agent loop manager
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)
train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
sampler_config = SamplerConfig(
    dataloader_cfg=dataloader_cfg,
    prompt_repeat_k=prompt_repeat_k,
)
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
)
if over_sample_threshold > 0 or partial_rollout:
    produce_strategy_config = AsyncProduceStrategyConfig(
        over_sample_threshold=over_sample_threshold,
        enable_partial_rollout=partial_rollout,
        tail_batch_trigger_size=tail_batch_trigger_size,
        tail_batch_stale_threshold=tail_batch_stale_threshold,
    )
else:
    produce_strategy_config = SyncProduceStrategyConfig()
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=produce_strategy_config,
        sampler_config=sampler_config,
    ),
)


# 6. eval agent loop manager
eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path, sample_ratio=1.0)
eval_dataset_cfg = [{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}]
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=eval_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
eval_sampler_config = SamplerConfig(
    dataloader_cfg=eval_dataloader_cfg,
    prompt_repeat_k=1,
)
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=1,
    top_p=1.0,
    temperature=0.0,
    min_tokens=0,
)
eval_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=evaluation_sample_params,
)
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="eval_task",
        agent_loop_config=eval_agent_loop_config,
        judger_config=judger_config,
        sampler_config=eval_sampler_config,
    ),
)


# 7. evaluator
evaluator_config = EvaluatorConfig(compute_metric_func=None)


# 8. RL Disaggregated Trainer Config
trainer = RLDisaggregatedTrainerConfig(
    train_resources=train_resources,
    rollout_resources=rollout_resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    train_batch_size=train_batch_size,
    total_train_steps=total_train_steps,
    sync_weights_interval=sync_weights_interval,
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=int(os.environ.get("SEED", "123")),
    debug_rollout=os.environ.get("DEBUG_ROLLOUT", "0") == "1",
)
