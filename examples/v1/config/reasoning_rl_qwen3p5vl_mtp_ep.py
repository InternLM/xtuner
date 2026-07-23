import json
import os

from transformers import AutoTokenizer

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import AgentLoopManagerConfig, SamplerConfig, TaskSpecConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig, DapoMathJudgerConfig, GEO3KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import RolloutImportanceSampling, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, get_eos_token
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig
from xtuner.v1.rl.utils import CPUResourcesConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    TaskSpecConfig,
    SamplerConfig,
    AsyncProduceStrategyConfig
)
import ray

# export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1
# export XTUNER_ACTIVATION_OFFLOAD=1

def _as_list(value):
    return value if isinstance(value, list) else [value]


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
meta_data_path = os.environ["DATA_PATH"]
eval_meta_data_path = os.environ.get("EVAL_DATA_PATH", "")

debug_rollout_dir = os.environ.get("DEBUG_ROLLOUT_DIR", "")
debug_train = os.environ.get("DEBUG_TRAIN", False)
debug_rollout = os.environ.get("DEBUG_ROLLOUT", False)

enable_evaluate = eval_meta_data_path is not None and eval_meta_data_path != ""

# basic settings
experimental_name = "reasoning_rl_qwen3p5vl_mtp_ep"
total_epochs = 15
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", 256))
prompt_repeat_k = 8
rollout_tp_size = 1
rollout_ep_size = 4
max_prompt_length = 2048
max_response_length = 8192
pack_max_length = 32768
train_optimizer_steps = 8
hf_interval = 15
checkpoint_interval = 50
evaluate_step = 5
enable_initial_evaluate = os.environ.get("ENABLE_INITIAL_EVALUATE", False)
train_ep_size=4
swap_optimizer = os.environ.get("SWAP_OPTIMIZER", "0").lower() in ("1", "true", "yes", "on")

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

# 2. rollout
rollout_config = RolloutConfig(
    fp32_lm_head=True,
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    skip_load_weights=True,
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.6,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=True,
    rollout_max_batch_size_per_instance=512,
    extra_rollout_config=dict(
        lmdeploy_log_level="INFO", 
        lmdeploy_uvicorn_log_level="INFO",
        lmdeploy_speculative_algorithm='qwen3_5_mtp',
        # MTP draft tokens trade throughput for extra activation memory; try 3 if still tight.
        lmdeploy_speculative_num_draft_tokens=3,
    ),
    health_check_interval_seconds=300,
    health_check_failure_threshold=3,
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=1,
    top_p=1.0,
    temperature=0.0,
    min_tokens=0,
    return_routed_experts=False,
)

# 3. datasets
with open(meta_data_path, "r", encoding="utf-8") as f:
    ds_collections = json.load(f)

train_dataset_cfg = []
for name, data in ds_collections.items():
    annotations = _as_list(data["annotation"])
    for annotation in annotations:
        train_dataset_cfg.append(
            {
                "dataset": DatasetConfig(
                    name=name,
                    anno_path=annotation,
                    media_root=data.get("media_root", ""),
                    sample_ratio=data.get("sample_ratio", 1.0),
                    class_name="VLMJsonlDataset",
                ),
                "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                    processor_path=model_path,
                    max_length=max_prompt_length,
                    system_message=data.get("system_message", None),
                    chat_template="qwen3.5-vl",
                    add_generation_prompt=True,
                    enable_thinking=True,
                ),
            }
        )

if enable_evaluate:
    with open(eval_meta_data_path, "r", encoding="utf-8") as f:
        eval_ds_collections = json.load(f)

    eval_dataset_cfg = []
    for name, data in eval_ds_collections.items():
        annotations = _as_list(data["annotation"])
        for annotation in annotations:
            eval_dataset_cfg.append(
                {
                    "dataset": DatasetConfig(
                        name=name,
                        anno_path=annotation,
                        media_root=data.get("media_root", ""),
                        sample_ratio=data.get("sample_ratio", 1.0),
                        class_name="VLMJsonlDataset",
                    ),
                    "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                        processor_path=model_path,
                        max_length=max_prompt_length,
                        system_message=data.get("system_message", None),
                        chat_template="qwen3.5-vl",
                        add_generation_prompt=True,
                        enable_thinking=True,
                        ignore_multimodal_info=True,
                    ),
                }
            )
    eval_dataloader_cfg = DataloaderConfig(
        dataset_config_list=eval_dataset_cfg,
        num_workers=8,
        collator="fake_collator",
        pack_level="none",
        pack_max_length=pack_max_length,
    )

dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    num_workers=8,
    collator="fake_collator",
    pack_level="none",
    pack_max_length=pack_max_length,
)


# 4. judger
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
eos_token_id = get_eos_token(model_path)
eos_token_str = tokenizer.convert_ids_to_tokens(eos_token_id)
judger_config = ComposedJudgerConfig(
    branches={
        "math_dapo": DapoMathJudgerConfig(
            judger_name="dapo_math",
            eos_token=eos_token_str,
            enable_overlong_buffer=True,
            max_response_len=max_response_length,
            overlong_buffer_len=4096,
            overlong_penalty_factor=1.0,
            tokenizer=tokenizer,
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
        "hiyouga/geometry3k": GEO3KJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
    },
)

from xtuner.v1.float8 import Float8Config, ScalingGranularity
# 5. train worker
model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
float8_cfg = Float8Config(
    scaling_granularity_gemm=None,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
model_cfg.float8_cfg = float8_cfg
model_cfg.text_config.ep_size = train_ep_size
model_cfg.text_config.z_loss_cfg = None
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.freeze_routers = True
model_cfg.compile_cfg = None
model_cfg.text_config.mtp_config = MTPConfig(
    num_layers=3, 
    loss_scaling_factor=1.0,
    detach_mtp_lm_head_weight=True,
    detach_mtp_inputs=True,
    share_weights=True,
)
optim_cfg = AdamWConfig(lr=1e-6, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False, swap_optimizer=swap_optimizer)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.28,
        cliprange_low=0.2,
        loss_type="vanilla",
        clip_ratio_c=10.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
    rollout_is=RolloutImportanceSampling(
        rollout_is_level="token",
        rollout_is_mode="both",
        rollout_is_threshold=(5, 0.5),
        rollout_is_mask_threshold=(5, 0.5),
        rollout_is_veto_threshold=(20, 0),
    ),
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=train_ep_size, fp32_lm_head=True)
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

def group_sample_filter_func(group_samples):
    # filter all correct or all wrong sample
    valid_samples = []
    for s in group_samples:
        if s.response_ids is not None:
            valid_samples.append(s)
        else:
            if s.routed_experts is not None:
                routed_experts = s.routed_experts
                if isinstance(routed_experts, ray.ObjectRef):
                    ray.internal.free([s.routed_experts], local_only=False)

    # filter all same reward sample
    rewards = [(d.reward or {}).get("score", 0.0) for d in valid_samples]
    if len(set(rewards)) == 1:
        print(f"filter all same reward sample: {rewards}")
        return False
    else:
        return True


produce_strategy_config = AsyncProduceStrategyConfig(
    over_sample_threshold=1,
    enable_partial_rollout=1,
    is_valid_sample_fn=group_sample_filter_func,
    max_staleness=3,
)
# produce_strategy_config= SyncProduceStrategyConfig(is_valid_sample_fn=group_sample_filter_func)

# 6. agent loop managers
agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=produce_strategy_config,
        sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=prompt_repeat_k),
    ),
)

if enable_evaluate:
    eval_agent_loop_config = SingleTurnAgentLoopConfig(
        hf_checkpoint=model_path,
        sample_params=evaluation_sample_params,
    )
    eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
        tasks=TaskSpecConfig(
            task_name="eval_task",
            agent_loop_config=eval_agent_loop_config,
            judger_config=judger_config,
            sampler_config=SamplerConfig(dataloader_cfg=eval_dataloader_cfg, prompt_repeat_k=1),
        ),
    )
    enable_evaluate=True
else:
    eval_agent_loop_manager_cfg = None
    enable_evaluate=False

# 7. trainer
trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=AsyncReplayBufferConfig(), # AsyncReplayBufferConfig()
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=EvaluatorConfig(compute_metric_func=None),
    load_from=model_path,
    total_epochs=total_epochs,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_initial_evaluate,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=1,
    debug_rollout_dir=debug_rollout_dir,
    debug_train=debug_train,
    debug_rollout=debug_rollout,
    exp_tracker="jsonl"
)
