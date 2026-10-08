"""Disaggregated PPO config for Qwen3.5-35B-A3B (VL MoE).

训练和 rollout 分卡，算法是 PPO：GAE 加同进程 critic。
数据是 GSM8K jsonl。打分只认 ``####`` 后的数字：对了 +1，没有 ``####`` 或数字不对都是 0。
需设置: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
可选: TRAIN_NUM_WORKERS, ROLLOUT_NUM_WORKERS, TRAIN_EP_SIZE, ROLLOUT_EP_SIZE,
ROLLOUT_TP_SIZE, TRAIN_BATCH_SIZE, TOTAL_TRAIN_STEPS, SP_SIZE
"""

import os
from copy import deepcopy

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GAEAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    DisaggAgentLoopManagerConfig,
    DisaggAsyncProduceStrategyConfig,
    DisaggTaskSpecConfig,
    SamplerConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.loss import CriticLossConfig, GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.trainer import CriticWorkerConfig, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLDisaggregatedTrainerConfig


def _set_head_type(config, head_type: str):
    """Set the output-head kind on a compose model."""
    config.text_config.head_type = head_type
    return config


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]

experimental_name = "disaggregated_ppo_qwen35_35b_a3b"
total_train_steps = int(os.environ.get("TOTAL_TRAIN_STEPS", "16"))
evaluate_step = int(os.environ.get("EVALUATE_STEP", str(total_train_steps)))
train_optimizer_steps = int(os.environ.get("TRAIN_OPTIMIZER_STEPS", "8"))
critic_num_passes = int(os.environ.get("CRITIC_NUM_PASSES", "2"))
critic_optimizer_steps = int(os.environ.get("CRITIC_OPTIMIZER_STEPS", "8"))
actor_lr = float(os.environ.get("ACTOR_LR", "1e-6"))
critic_lr = float(os.environ.get("CRITIC_LR", "5e-6"))
actor_lr_warmup_updates = int(os.environ.get("ACTOR_LR_WARMUP_UPDATES", "50"))
critic_lr_warmup_updates = int(os.environ.get("CRITIC_LR_WARMUP_UPDATES", "32"))
actor_scheduler_steps = total_train_steps * train_optimizer_steps
critic_scheduler_steps = total_train_steps * critic_num_passes * critic_optimizer_steps
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "32"))
sync_weights_interval = int(os.environ.get("SYNC_WEIGHTS_INTERVAL", "1"))
over_sample_threshold = float(os.environ.get("OVER_SAMPLE_THRESHOLD", "0.0"))
partial_rollout = os.environ.get("PARTIAL_ROLLOUT", "0") == "1"
tail_batch_trigger_size = int(os.environ.get("TAIL_BATCH_TRIGGER_SIZE", "-1"))
max_staleness = int(os.environ.get("MAX_STALENESS", "0"))
prompt_repeat_k = int(os.environ.get("PROMPT_REPEAT_K", "4"))
rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", "1"))
rollout_ep_size = int(os.environ.get("ROLLOUT_EP_SIZE", "1"))
train_ep_size = int(os.environ.get("TRAIN_EP_SIZE", "1"))
max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "1024"))
max_response_length = int(os.environ.get("MAX_RESPONSE_LENGTH", "4096"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", str(5 * 1024)))
sp_size = int(os.environ.get("SP_SIZE", "1"))
enable_evaluate = os.environ.get("ENABLE_EVALUATE", "1") == "1"
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "1") == "1"
# 4 卡 EP、fsdp_size=1 时，单卡常驻参数约 40GB（fp32）。Adam 的 exp_avg/exp_avg_sq
# 再要约 80GB，H200 140GB 在 critic 第一次 step 时会 OOM。状态放 CPU，逐步换入 GPU。
swap_optimizer = os.environ.get("SWAP_OPTIMIZER", "1").lower() in ("1", "true", "yes", "on")

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

rollout_config = RolloutConfig(
    fp32_lm_head=True,
    env=experimental_name,
    device=rollout_resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.8")),
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=enable_return_routed_experts,
)

# 与 GSM8K 的 #### 答案对齐：对了 +1，抽错或没有 #### 都是 0。
judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")

actor_lr_cfg = LRConfig(lr_type="constant", warmup_ratio=actor_lr_warmup_updates, lr_min=actor_lr)
critic_lr_cfg = LRConfig(lr_type="constant", warmup_ratio=critic_lr_warmup_updates, lr_min=critic_lr)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=train_ep_size, fp32_head=True)

model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
model_cfg.text_config.ep_size = train_ep_size
actor_model_cfg = _set_head_type(deepcopy(model_cfg), "lm_head")
critic_model_cfg = _set_head_type(deepcopy(model_cfg), "value_head")
critic_model_cfg.text_config.mesh_prefix = "critic"
critic_model_cfg.text_config.mtp_config = None

optim_cfg = AdamWConfig(
    lr=actor_lr,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.1,
    foreach=False,
    skip_grad_norm_threshold=5.0,
    eps=1e-15,
    swap_optimizer=swap_optimizer,
)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.28,
        cliprange_low=0.2,
        loss_type=os.environ.get("LOSS_TYPE", "vanilla"),
        clip_ratio_c=3.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=512,
    rollout_is=RolloutImportanceSampling(
        rollout_is_level="token",
        rollout_is_mode="both",
        rollout_is_threshold=(5, 0),
        rollout_is_mask_threshold=(5, 0.5),
        rollout_is_veto_threshold=(5, 0),
    ),
)
critic_cfg = CriticWorkerConfig(
    model_cfg=critic_model_cfg,
    load_from=model_path,
    optim_cfg=AdamWConfig(
        lr=critic_lr,
        betas=(0.9, 0.95),
        max_grad_norm=10.0,
        weight_decay=0.1,
        foreach=False,
        eps=1e-15,
        swap_optimizer=swap_optimizer,
    ),
    loss_cfg=CriticLossConfig(cliprange_value=0.6, ignore_idx=-100),
    lr_cfg=critic_lr_cfg,
    fsdp_cfg=deepcopy(fsdp_cfg),
    sp_size=sp_size,
    optimizer_steps=train_optimizer_steps,
    num_passes=critic_num_passes,
    optimizer_steps_per_pass=critic_optimizer_steps,
    scheduler_steps=critic_scheduler_steps,
)
train_worker_cfg = WorkerConfig(
    model_cfg=actor_model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=actor_lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=sp_size,
    optimizer_steps=train_optimizer_steps,
    scheduler_steps=actor_scheduler_steps,
    pack_max_length=pack_max_length,
    critic_cfg=critic_cfg,
)

train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
tokenizer_config = RLQwen3VLTokenizeFnConfig(
    processor_path=model_path,
    max_length=max_prompt_length,
    chat_template="qwen3.5-vl",
    add_generation_prompt=True,
    enable_thinking=True,
)
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
produce_strategy_config = DisaggAsyncProduceStrategyConfig(
    over_sample_threshold=over_sample_threshold,
    enable_partial_rollout=partial_rollout,
    tail_batch_trigger_size=tail_batch_trigger_size,
    max_staleness=max_staleness,
)
agent_loop_manager_cfg = DisaggAgentLoopManagerConfig(
    tasks=DisaggTaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=produce_strategy_config,
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

evaluator_config = EvaluatorConfig(compute_metric_func=None)

trainer = RLDisaggregatedTrainerConfig(
    train_resources=train_resources,
    rollout_resources=rollout_resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=AsyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GAEAdvantageConfig(
        gae_gamma=1.0,
        gae_lambda=1.0,
        reward_scope="segment",
        normalize_actor_advantage=True,
    ),
    total_train_steps=total_train_steps,
    sync_weights_interval=sync_weights_interval,
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=True,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=int(os.environ.get("SEED", "123")),
)
