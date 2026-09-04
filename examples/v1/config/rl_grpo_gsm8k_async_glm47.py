"""Generic GSM8K async GRPO config used by the GLM-4.7 rollout job."""

import json
import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.float8 import Float8Config, ScalingGranularity
from xtuner.v1.model import get_model_config_from_hf
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
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
num_nodes = int(os.environ.get("WORLD_SIZE", "1"))
enable_return_routed_experts = (
    os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "1") == "1"
)

experimental_name = "grpo_gsm8k"
total_train_steps = int(os.environ.get("TOTAL_TRAIN_STEPS", "45"))
evaluate_step = int(os.environ.get("EVALUATE_STEP", str(total_train_steps)))
train_optimizer_steps = int(os.environ.get("TRAIN_OPTIMIZER_STEPS", "1"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "64"))
prompt_repeat_k = int(os.environ.get("PROMPT_REPEAT_K", "5"))
rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", "1"))
rollout_ep_size = int(os.environ.get("ROLLOUT_EP_SIZE", "4"))
train_ep_size = int(os.environ.get("TRAIN_EP_SIZE", "4"))
max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
max_response_length = int(os.environ.get("MAX_RESPONSE_LENGTH", "1024"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", str(32 * 1024)))
enable_evaluate = os.environ.get("ENABLE_EVALUATE", "1") == "1"
enable_fp8 = os.environ.get("FP8", "0") == "1"
enable_mtp = os.environ.get("ENABLE_MTP", "1") == "1"
mtp_num_layers = int(os.environ.get("MTP_NUM_LAYERS", "2"))
over_sample_threshold = float(os.environ.get("OVER_SAMPLE_THRESHOLD", "0.8"))
partial_rollout = os.environ.get("PARTIAL_ROLLOUT", "1") == "1"
max_staleness = int(os.environ.get("MAX_STALENESS", "0"))
tail_batch_trigger_size = int(os.environ.get("TAIL_BATCH_TRIGGER_SIZE", "64"))
enable_group_filter = os.environ.get("ENABLE_GROUP_FILTER", "1") == "1"

if enable_fp8:
    os.environ.setdefault("XTUNER_RL_FP8_QUANTIZE_IN_BF16", "1")

model_cfg = get_model_config_from_hf(Path(model_path))
language_model_cfg = getattr(model_cfg, "text_config", model_cfg)
language_model_cfg.float8_cfg = (
    Float8Config(
        scaling_granularity_gemm=None,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )
    if enable_fp8
    else None
)
language_model_cfg.ep_size = train_ep_size
language_model_cfg.z_loss_cfg = None
language_model_cfg.balancing_loss_cfg = None
language_model_cfg.freeze_routers = True
if hasattr(language_model_cfg.attention, "sparse_mla_backend"):
    language_model_cfg.attention.sparse_mla_backend = os.environ.get(
        "SPARSE_MLA_BACKEND", "tilelang"
    )
language_model_cfg.mtp_config = (
    MTPConfig(
        num_layers=mtp_num_layers,
        loss_scaling_factor=1.0,
        detach_mtp_lm_head_weight=True,
        detach_mtp_inputs=True,
        share_weights=True,
    )
    if enable_mtp
    else None
)
model_cfg.compile_cfg = None

with (Path(model_path) / "config.json").open() as config_file:
    hf_model_type = json.load(config_file)["model_type"]
default_speculative_algorithm = (
    "qwen3_5_mtp" if hf_model_type == "qwen3_5_moe" else "deepseek_mtp"
)

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * num_nodes,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=int(
        os.environ.get("CPU_MEMORY_PER_WORKER_GIB", "32")
    )
    * 1024**3,
)

extra_rollout_config = {
    "lmdeploy_backend": "pytorch",
    "lmdeploy_log_level": os.environ.get("LMDEPLOY_LOG_LEVEL", "ERROR"),
    "lmdeploy_uvicorn_log_level": os.environ.get(
        "LMDEPLOY_UVICORN_LOG_LEVEL", "ERROR"
    ),
}
if enable_mtp:
    extra_rollout_config.update(
        lmdeploy_speculative_algorithm=os.environ.get(
            "ROLLOUT_SPECULATIVE_ALGORITHM", default_speculative_algorithm
        ),
        lmdeploy_speculative_num_draft_tokens=int(
            os.environ.get("ROLLOUT_SPECULATIVE_NUM_DRAFT_TOKENS", "3")
        ),
    )

rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=float(
        os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.8")
    ),
    context_length=max_response_length + max_prompt_length,
    enable_float8=enable_fp8,
    skip_load_weights=os.environ.get("SKIP_LOAD_WEIGHTS", "0") == "1",
    enable_return_routed_experts=enable_return_routed_experts,
    fp32_lm_head=True,
    rollout_timeout=36000,
    rollout_max_batch_size_per_instance=32 * rollout_ep_size,
    extra_rollout_config=extra_rollout_config,
)

judger_resources = CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1)
train_judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=judger_resources,
)
eval_judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=judger_resources,
)

lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=train_ep_size,
    fp32_lm_head=True,
)
optim_cfg = AdamWConfig(
    lr=1e-6,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.1,
    foreach=False,
    skip_grad_norm_threshold=None,
    eps=1e-15,
)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg={
        "cliprange_high": 0.28,
        "cliprange_low": 0.2,
        "loss_type": "vanilla",
        "clip_ratio_c": 3.0,
        "log_prob_diff_min": -20.0,
        "log_prob_diff_max": 20.0,
    },
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
    rollout_is=RolloutImportanceSampling(
        rollout_is_level="token",
        rollout_is_mode="mask",
        rollout_is_threshold=(5.0, 0.5),
    ),
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

tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
train_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[
        {"dataset": train_dataset, "tokenize_fn": tokenizer_config}
    ],
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
    return_routed_experts=enable_return_routed_experts,
)


def group_samples_filter_func(rollout_states: list[RolloutState]) -> bool:
    rewards = [
        state.reward["score"]
        for state in rollout_states
        if state.response_ids is not None
    ]
    return len(set(rewards)) != 1


produce_strategy_kwargs = {
    "over_sample_threshold": over_sample_threshold,
    "enable_partial_rollout": partial_rollout,
    "max_staleness": max_staleness,
    "tail_batch_trigger_size": tail_batch_trigger_size,
}
if enable_group_filter:
    produce_strategy_kwargs["is_valid_sample_fn"] = group_samples_filter_func

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=training_sample_params,
        ),
        judger_config=train_judger_config,
        produce_strategy_config=AsyncProduceStrategyConfig(
            **produce_strategy_kwargs
        ),
        sampler_config=SamplerConfig(
            dataloader_cfg=train_dataloader_cfg,
            prompt_repeat_k=prompt_repeat_k,
        ),
    ),
)

eval_dataset = DatasetConfig(
    name=experimental_name,
    anno_path=eval_data_path,
    sample_ratio=1.0,
)
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[
        {"dataset": eval_dataset, "tokenize_fn": tokenizer_config}
    ],
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
        judger_config=eval_judger_config,
        sampler_config=SamplerConfig(
            dataloader_cfg=eval_dataloader_cfg,
            prompt_repeat_k=1,
        ),
    ),
)

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
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    hf_interval=10,
    work_dir=work_dir,
    seed=int(os.environ.get("SEED", "123")),
    debug_rollout=False,
)
