"""PPO colocated trainer config (GSM8K).

用法：通过环境变量传入路径后，由 CLI 加载本配置并 trainer.build().fit()。
需设置: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
可选: WORLD_SIZE, ENABLE_RETURN_ROUTED_EXPERTS, LOSS_TYPE, LOSS_MODE, SP_SIZE

Critic 与 actor 共用同一个 TrainingWorker 进程。GAE 在 actor 更新前完成，
outcome 奖励是样本的 reward["score"]。第一次跑通使用 SP_SIZE=1。
"""

import os
from copy import deepcopy

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.rl.advantage import GAEAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.loss import CriticLossConfig, GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trace import TraceConfig
from xtuner.v1.rl.trainer import CriticWorkerConfig, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


def _set_head_type(config, head_type: str):
    """Set the output-head kind for dense or composed model configs."""
    if hasattr(config, "text_config"):
        config.text_config.head_type = head_type
    else:
        config.head_type = head_type
    return config


# env
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")
enable_trace = True
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

# basic settings
experimental_name = "ppo_gsm8k"
total_train_steps = 45
evaluate_step = 45
train_optimizer_steps = 1
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", 64 * train_optimizer_steps))
prompt_repeat_k = 5
rollout_tp_size = 1
rollout_ep_size = 4
max_prompt_length = 512
max_response_length = 1024
pack_max_length = 32 * 1024
sp_size = int(os.environ.get("SP_SIZE", "1"))

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * NNODE,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.8,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=(enable_return_routed_experts == "1"),
)

# 3. judger
judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)

# 4. train worker
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=4)

# Qwen3.5-35B-A3B：与 intern-s2 / autotest 同一套 compose backbone。
# GSM8K 只训 text；MTP 仅 actor 需要，critic 的 value_head 禁止 MTP。
model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
model_cfg.float8_cfg = None
model_cfg.compile_cfg = False
model_cfg.text_config.ep_size = 4
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.z_loss_cfg = None
model_cfg.text_config.mtp_config = MTPConfig(num_layers=1)
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
actor_model_cfg = _set_head_type(deepcopy(model_cfg), "lm_head")
critic_model_cfg = _set_head_type(deepcopy(model_cfg), "value_head")
critic_model_cfg.text_config.mtp_config = None
critic_cfg = CriticWorkerConfig(
    model_cfg=critic_model_cfg,
    load_from=model_path,
    optim_cfg=AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1),
    loss_cfg=CriticLossConfig(cliprange_value=0.2, ignore_idx=-100),
    lr_cfg=lr_cfg,
    fsdp_cfg=deepcopy(fsdp_cfg),
    sp_size=sp_size,
    optimizer_steps=train_optimizer_steps,
)
train_worker_cfg = WorkerConfig(
    model_cfg=actor_model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=sp_size,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
    critic_cfg=critic_cfg,
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

# 8. RL Colocate Trainer Config（CLI 通过 config["trainer"].build() 得到 Trainer）
trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    total_train_steps=total_train_steps,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GAEAdvantageConfig(
        gamma=1.0,
        lam=0.95,
        reward_scope="segment",
    ),
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
)
