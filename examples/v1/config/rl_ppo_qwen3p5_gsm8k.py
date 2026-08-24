"""PPO + GSM8K on Qwen3.5-VL-MoE-35B-A3B (RL Colocate Trainer).

PPO differs from the GRPO configs in this directory by replacing the
group-relative baseline with a learned value function:

* ``advantage_estimator_config=GAEAdvantageConfig(...)`` selects token-level
  GAE, which requires a critic.
* ``critic_cfg`` configures that critic. Its model config is derived from the
  actor's with ``as_value_config``, which swaps the vocabulary head for a scalar
  value head while keeping every backbone checkpoint key, so the critic can be
  initialized directly from the actor checkpoint.
* ``kl_reward_cfg`` folds the reference-KL penalty into the token reward instead
  of the loss, so it flows through GAE into the value targets.

Because the actor and critic are both full 35B models that share one placement
group, they are never resident on the accelerator at the same time; the worker
swaps them per phase. Both sets of weights and Adam states live in host memory
between phases, which roughly doubles the CPU memory a GRPO run needs -- see
``cpu_memory_per_worker`` below.

Usage: set the environment variables, then let the CLI load this config and call
``trainer.build().fit()``.
Required: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
Optional: WORLD_SIZE, TRAIN_BATCH_SIZE, SP_SIZE, CRITIC_WARMUP_STEPS, KL_COEF
"""

import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.value import as_value_config
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
from xtuner.v1.rl.loss import GRPOLossConfig, ValueLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import CriticWorkerConfig, KLRewardConfig, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


# env
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
NNODE = int(os.environ.get("WORLD_SIZE", "8"))

# basic settings
experimental_name = "ppo_gsm8k_qwen3p5"
# Overridable so a short smoke run can reuse this config unchanged.
total_train_steps = int(os.environ.get("TOTAL_TRAIN_STEPS", "200"))
evaluate_step = int(os.environ.get("EVALUATE_STEP", "50"))
train_optimizer_steps = int(os.environ.get("TRAIN_OPTIMIZER_STEPS", "4"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", 64 * train_optimizer_steps))
enable_evaluate = os.environ.get("ENABLE_EVALUATE", "1") == "1"
# PPO learns a per-token baseline, so it does not need several samples per
# prompt to form one. A small k still helps rollout throughput and keeps the
# reward distribution observable.
prompt_repeat_k = int(os.environ.get("PROMPT_REPEAT_K", "2"))
rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", "2"))
rollout_ep_size = 1
max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
max_response_length = int(os.environ.get("MAX_RESPONSE_LENGTH", "1024"))
sp_size = int(os.environ.get("SP_SIZE", "1"))
# Must be divisible by sp_size: GAE runs on the gathered full sequence, so the
# gather has to return exactly pack_max_length.
pack_max_length = 32 * 1024
assert pack_max_length % sp_size == 0

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * NNODE,
    # Actor and critic each keep fp32 weights plus Adam moments in host memory
    # (~12 bytes per parameter each), so a 35B PPO run needs roughly twice the
    # host memory of the equivalent GRPO run. At 64 workers that is about
    # 13 GiB of state per worker before pinned staging buffers and the Ray
    # object store, which is why the GRPO default of 16 GiB is not enough.
    cpu_memory_per_worker=48 * 1024**3,  # 48 GB
    num_cpus_per_worker=12,
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
)

# 3. judger
judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)

# 4. actor
actor_model_cfg = Qwen3_5_VLMoE35BA3Config()
actor_loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.2,
        cliprange_low=0.2,
        loss_type="vanilla",
        clip_ratio_c=10.0,
    ),
    ignore_idx=-100,
    # PPO applies the KL penalty to the reward instead; see kl_reward_cfg.
    use_kl_loss=False,
    mode="chunk",
    chunk_size=512,
)

# 5. critic
# The value config keeps the actor's backbone and replaces its vocabulary head
# with a scalar one, so `load_mode="init_from_actor"` reads every backbone
# tensor from the actor checkpoint and initializes only the value head.
critic_model_cfg = as_value_config(actor_model_cfg)
critic_cfg = CriticWorkerConfig(
    model_cfg=critic_model_cfg,
    # The value head starts untrained, so the critic tolerates -- and needs -- a
    # larger learning rate than the policy.
    optim_cfg=AdamWConfig(lr=5e-6, foreach=False, weight_decay=0.0),
    loss_cfg=ValueLossConfig(loss_type="clipped", value_clip=0.5),
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=5e-6),
    fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, mesh_prefix="critic"),
    load_mode="init_from_actor",
    # Value regression has no trust region to violate, so the critic can reuse
    # a batch more aggressively than the policy.
    num_passes=2,
    optimizer_steps_per_pass=train_optimizer_steps,
    scheduler_steps=total_train_steps * train_optimizer_steps * 2,
    # A fresh value head predicts noise, so the first advantages are noise too.
    # Train the critic alone for a few steps before it starts steering the policy.
    warmup_steps=int(os.environ.get("CRITIC_WARMUP_STEPS", "5")),
)

train_worker_cfg = WorkerConfig(
    model_cfg=actor_model_cfg,
    load_from=model_path,
    optim_cfg=AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.0),
    loss_cfg=actor_loss_cfg,
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=1e-6),
    fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1),
    sp_size=sp_size,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
    scheduler_steps=total_train_steps * train_optimizer_steps,
    critic_cfg=critic_cfg,
    # Subtract beta * KL(policy || reference) from the token reward so the
    # penalty is discounted by GAE and learned by the critic. Set KL_COEF=0 to
    # disable it and let the policy drift freely.
    kl_reward_cfg=KLRewardConfig(
        coef=float(os.environ.get("KL_COEF", "0.001")),
        kl_type="low_var_kl",
        behavior_logprobs="old",
    ),
)

# 6. train agent loop manager
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)
train_dataset_cfg = [
    {
        "dataset": DatasetConfig(name=experimental_name, anno_path=data_path),
        "tokenize_fn": tokenizer_config,
    }
]
sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=train_dataset_cfg,
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=prompt_repeat_k,
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=SampleParams(
                max_tokens=max_response_length,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                min_tokens=0,
            ),
        ),
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=sampler_config,
    ),
)

# 7. eval agent loop manager
eval_dataset_cfg = [
    {
        "dataset": DatasetConfig(name=experimental_name, anno_path=eval_data_path, sample_ratio=1.0),
        "tokenize_fn": tokenizer_config,
    }
]
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
        sampler_config=SamplerConfig(
            dataloader_cfg=DataloaderConfig(
                dataset_config_list=eval_dataset_cfg,
                pack_max_length=pack_max_length,
                collator="fake_collator",
                pack_level="none",
            ),
            prompt_repeat_k=1,
        ),
    ),
)

# 8. trainer
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
    # Selecting GAE is what turns this into PPO; the trainer validates that a
    # critic is configured and pushes gamma/lambda down to the worker.
    advantage_estimator_config=GAEAdvantageConfig(
        # gamma=1.0 is standard for RLHF: episodes are short and a discount
        # would bias against long correct answers.
        gamma=1.0,
        gae_lambda=0.95,
        normalize_advantage=True,
    ),
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
)
