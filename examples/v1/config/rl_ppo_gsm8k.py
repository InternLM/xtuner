"""PPO role-wiring example based on the GSM8K colocated RL config.

This example intentionally focuses on configuration and role construction.
Critic values and per-sample GAE run inside ``TrainingController.fit`` before
the actor update. The outcome reward is the sample ``reward["score"]``.
Environment variables are inherited from ``rl_grpo_gsm8k_judge``.
"""

import os
from copy import deepcopy

from examples.v1.config.rl_grpo_gsm8k_judge import *  # noqa: F401,F403

from xtuner.v1.rl.advantage import GAEAdvantageConfig
from xtuner.v1.rl.loss import CriticLossConfig
from xtuner.v1.rl.trainer import CriticWorkerConfig, WorkerConfig


def _set_head_type(config, head_type: str):
    """Set the output-head kind for dense or composed model configs."""
    if hasattr(config, "text_config"):
        config.text_config.head_type = head_type
    else:
        config.head_type = head_type
    return config


actor_model_cfg = _set_head_type(deepcopy(model_cfg), "lm_head")
critic_model_cfg = _set_head_type(deepcopy(model_cfg), "value_head")

critic_cfg = CriticWorkerConfig(
    model_cfg=critic_model_cfg,
    load_from=model_path,
    optim_cfg=AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1),
    loss_cfg=CriticLossConfig(cliprange_value=0.2, ignore_idx=-100),
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
    fsdp_cfg=deepcopy(fsdp_cfg),
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    optimizer_steps=train_optimizer_steps,
)

train_worker_cfg = WorkerConfig(
    model_cfg=actor_model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
    critic_cfg=critic_cfg,
)

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
        gae_gamma=1.0,
        gae_lambda=0.95,
        reward_scope="segment",
    ),
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
    trace_config=trace_config,
)
