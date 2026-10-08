"""PPO colocated trainer config (DAPO math, Qwen2.5-Math-1.5B).

用法：通过环境变量传入路径后，由 CLI 加载本配置并 trainer.build().fit()。
需设置: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
DATA_PATH 与 EVAL_DATA_PATH 是 xpuyu 数据清单 json，不是 jsonl。
可选: WORLD_SIZE, LOSS_TYPE, LOSS_MODE, SP_SIZE, COMPASS_EVAL_HOSTS

Actor 与 critic 都是 Qwen2 dense。Critic 用 value head，mesh 前缀是 ``critic``。
上下文不超过模型的 4096。Actor 每个 rollout 更新 8 次，critic 更新 2 * 8 次。
"""

import json
import math
import os
from copy import deepcopy
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import Qwen2DenseConfig
from xtuner.v1.rl.advantage import GAEAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    SamplerConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig, MathCompassJudgerConfig, MathRuleJudgerConfig
from xtuner.v1.rl.loss import CriticLossConfig, GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.trainer import CriticWorkerConfig, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


def keep_nonuniform_reward_group(rollout_states: list[RolloutState]) -> bool:
    """Keep a group only when its rewards are not all the same."""
    if len(rollout_states) < 2:
        return False
    rewards: list[float] = []
    for state in rollout_states:
        prompt_ids = state.extra_fields.get("train_prompt_ids") or state.prompt_ids
        response_ids = state.response_ids
        logprobs = state.logprobs
        reward = state.reward
        if not (
            prompt_ids
            and isinstance(state.response, str)
            and state.response
            and response_ids
            and logprobs is not None
            and isinstance(reward, dict)
            and "score" in reward
        ):
            return False
        try:
            score = float(reward["score"])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score):
            return False
        rewards.append(score)
    return len(set(rewards)) > 1


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
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

# data_source -> judger branch. Eval jsonl uses these exact data_source values.
DATA_JUDGER_MAPPING = {
    "math_dapo": {"math_integer": 1.0},
    "AIME2025": {"math_integer": 1.0},
    "AMC23": {"math_integer": 1.0},
    "MATH500": {"math_latex": 1.0},
    "OlympiadBench": {"math_symbolic": 1.0},
    "Minerva": {"math_compass": 1.0},
}
COMPASS_EVAL_HOSTS = [
    host.strip()
    for host in os.environ.get("COMPASS_EVAL_HOSTS", "100.104.170.77:23332,100.104.170.77:23333").split(",")
    if host.strip()
]


def _parse_xpuyu_manifest(path: str, tokenize_fn: RLTextTokenizeFnConfig, max_prompt_length: int) -> list[dict]:
    """Expand an xpuyu dataset manifest into dataloader entries.

    Args:
        path (str): Manifest JSON path.
        tokenize_fn (RLTextTokenizeFnConfig): Base tokenizer config copied per dataset.
        max_prompt_length (int): Prompt token limit after the chat template.

    Returns:
        list[dict]: Dataset and tokenizer pairs for ``DataloaderConfig``.
    """
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    dataset_cfg = []
    for name, ds_cfg in manifest.items():
        annotations = ds_cfg["annotation"]
        if isinstance(annotations, str):
            annotations = [annotations]
        for annotation in annotations:
            if not Path(annotation).is_file():
                raise FileNotFoundError(f"dataset {name} annotation not found: {annotation}")
            dataset_cfg.append(
                {
                    "dataset": DatasetConfig(
                        name=name,
                        anno_path=annotation,
                        sample_ratio=float(ds_cfg["sample_ratio"]),
                    ),
                    "tokenize_fn": tokenize_fn.model_copy(
                        update={
                            "max_length": max_prompt_length,
                            "system_prompt": ds_cfg.get("system_message"),
                            "data_judger_mapping": DATA_JUDGER_MAPPING,
                        }
                    ),
                }
            )
    return dataset_cfg


def _math_judger(compass_hosts: list[str], include_compass: bool) -> ComposedJudgerConfig:
    """Build the baseline math judger branches.

    Args:
        compass_hosts (list[str]): CompassVerifier hosts. Empty disables the no-answer fallback.
        include_compass (bool): Whether to add the Minerva-only Compass branch.

    Returns:
        ComposedJudgerConfig: Judger routed by ``RolloutState.data_source``.
    """
    cpu = CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1)
    branches = {
        kind: MathRuleJudgerConfig(
            judger_name=kind,
            kind=rule_kind,
            compass_hosts=compass_hosts,
            cpu_resources=cpu,
        )
        for kind, rule_kind in (
            ("math_integer", "integer"),
            ("math_latex", "latex"),
            ("math_symbolic", "symbolic"),
        )
    }
    if include_compass:
        branches["math_compass"] = MathCompassJudgerConfig(
            judger_name="math_compass",
            compass_hosts=compass_hosts,
            cpu_resources=cpu,
        )
    return ComposedJudgerConfig(branches=branches)


# basic settings
experimental_name = "ppo_qwen25_math_1p5b"
total_train_steps = 128
evaluate_step = 20
train_optimizer_steps = 8
critic_num_passes = 2
critic_optimizer_steps = 8
actor_lr = 1e-6
critic_lr = 5e-6
actor_lr_warmup_updates = 50
critic_lr_warmup_updates = 32
actor_scheduler_steps = total_train_steps * train_optimizer_steps
critic_scheduler_steps = total_train_steps * critic_num_passes * critic_optimizer_steps
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "256"))
prompt_repeat_k = 16
rollout_tp_size = 1
# Qwen2.5-Math-1.5B max_position_embeddings is 4096.
max_prompt_length = 1024
max_response_length = 3072
eval_max_prompt_length = 1344
eval_max_response_length = 2752
pack_max_length = 4096
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
    gpu_memory_utilization=0.4,
    context_length=max_response_length + max_prompt_length,
)

# 3. judger
# 训练只用整数规则。评估在抽不到答案时再问 Compass，Minerva 整套交给 Compass。
judger_config = _math_judger([], include_compass=False)
eval_judger_config = _math_judger(COMPASS_EVAL_HOSTS, include_compass=True)

# 4. train worker
actor_lr_cfg = LRConfig(lr_type="constant", warmup_ratio=actor_lr_warmup_updates, lr_min=actor_lr)
critic_lr_cfg = LRConfig(lr_type="constant", warmup_ratio=critic_lr_warmup_updates, lr_min=critic_lr)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False)

model_cfg = Qwen2DenseConfig.from_hf(model_path)
model_cfg.float8_cfg = None
model_cfg.compile_cfg = False
optim_cfg = AdamWConfig(
    lr=actor_lr,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.1,
    foreach=False,
    skip_grad_norm_threshold=5.0,
    eps=1e-15,
)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.28,
        cliprange_low=0.2,
        loss_type=os.environ.get("LOSS_TYPE", "mask_pg"),
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
actor_model_cfg = _set_head_type(deepcopy(model_cfg), "lm_head")
critic_model_cfg = _set_head_type(deepcopy(model_cfg), "value_head")
critic_model_cfg.mesh_prefix = "critic"
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

# 5. train agent loop manager
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length, data_judger_mapping=DATA_JUDGER_MAPPING)
train_dataset_cfg = _parse_xpuyu_manifest(data_path, tokenizer_config, max_prompt_length)
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
produce_strategy_config = AsyncProduceStrategyConfig(
    over_sample_threshold=1.0,
    enable_partial_rollout=True,
    max_staleness=1,
    max_token_staleness=None,
    tail_batch_trigger_size=-1,
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=judger_config,
        is_valid_sample_fn=keep_nonuniform_reward_group,
        produce_strategy_config=produce_strategy_config,
        sampler_config=sampler_config,
    ),
)

# 6. eval agent loop manager
eval_dataset_cfg = _parse_xpuyu_manifest(eval_data_path, tokenizer_config, eval_max_prompt_length)
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
    max_tokens=eval_max_response_length,
    top_k=0,
    top_p=0.95,
    temperature=0.8,
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
        judger_config=eval_judger_config,
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
    replay_buffer_config=AsyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    total_train_steps=total_train_steps,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GAEAdvantageConfig(
        gae_gamma=1.0,
        gae_lambda=1.0,
        reward_scope="segment",
        normalize_actor_advantage=True,
    ),
    enable_evaluate=True,
    enable_initial_evaluate=True,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=42,
    debug_rollout=False,
)
