"""GLM-5.3-Flash text RL (GRPO + GSM8K) example.

Text-only RL on the published VL checkpoint: ``get_model_config_from_hf`` maps
``glm5_next`` to ``Glm53TextMoEConfig`` (same path as ``sft_glm53.py``). Prompt
format comes from the checkpoint tokenizer ``apply_chat_template``.

Required env:
  WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH

Recommended env:
  export XTUNER_USE_LMDEPLOY=1
  export XTUNER_DETERMINISTIC=true

Launch::

  bash examples/v1/scripts/run_rl.sh \\
    examples/v1/config/rl_glm53_25B_grpo_gsm8k.py \\
    lmdeploy $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
"""

import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig

work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "1")
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

experimental_name = os.environ.get("EXPERIMENTAL_NAME", "glm53_grpo_gsm8k")
total_train_steps = int(os.environ.get("TOTAL_TRAIN_STEPS", "15"))
evaluate_step = int(os.environ.get("EVALUATE_STEP", str(total_train_steps)))
train_optimizer_steps = int(os.environ.get("TRAIN_OPTIMIZER_STEPS", "1"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "16")) * train_optimizer_steps
prompt_repeat_k = int(os.environ.get("PROMPT_REPEAT_K", "5"))
rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", "1"))
rollout_ep_size = int(os.environ.get("ROLLOUT_EP_SIZE", "8"))
max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
max_response_length = int(os.environ.get("MAX_RESPONSE_LENGTH", "1024"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", "8192"))

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * NNODE,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.5")),
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=(enable_return_routed_experts == "1"),
    extra_rollout_config=dict(
        lmdeploy_trust_remote_code=True,
        lmdeploy_log_level="INFO",
        lmdeploy_uvicorn_log_level="INFO",
    ),
)

judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)

lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=int(os.environ.get("FSDP_EP_SIZE", "1")),
)
model_cfg = get_model_config_from_hf(Path(model_path))
model_cfg.dispatcher = os.environ.get("DISPATCHER", "all2all")
model_cfg.compile_cfg = False
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None
if hasattr(model_cfg, "attention") and hasattr(model_cfg.attention, "sparse_mla_backend"):
    model_cfg.attention.sparse_mla_backend = os.environ.get("SPARSE_MLA_BACKEND", "flash_mla_cudnn").strip().lower()
if "INDEXER_BACKEND" in os.environ and hasattr(model_cfg, "attention"):
    model_cfg.attention.indexer_backend = os.environ["INDEXER_BACKEND"].strip().lower()

optim_cfg = AdamWConfig(lr=float(os.environ.get("LR", "1e-6")), foreach=False, weight_decay=0.1)
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
    chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "512")),
)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=int(os.environ.get("SP_SIZE", "2")),
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)
train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
sampler_config = SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=prompt_repeat_k)
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
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=sampler_config,
    ),
)

eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path, sample_ratio=1.0)
eval_dataset_cfg = [{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}]
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=eval_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="eval_task",
        agent_loop_config=SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=SampleParams(
                max_tokens=max_response_length,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                min_tokens=0,
            ),
        ),
        judger_config=judger_config,
        sampler_config=SamplerConfig(dataloader_cfg=eval_dataloader_cfg, prompt_repeat_k=1),
    ),
)

trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
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
    seed=int(os.environ.get("SEED", "123")),
    debug_rollout=False,
    exp_tracker="jsonl",
)
