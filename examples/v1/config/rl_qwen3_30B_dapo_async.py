import os
from copy import deepcopy

from transformers import AutoTokenizer
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets import RLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.ray.base import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.rl_trainer import RLTrainerConfig
from xtuner.v1.rl.base.rollout_is import RolloutImportanceSampling
from xtuner.v1.model import get_model_config_from_hf



work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ.get("EVAL_DATA_PATH")
enable_evaluate = True if eval_data_path != "" else False
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))
enable_return_routed_experts = 1
enbale_partial_rollout = 1
staleness_threshold = 0.2
tail_batch_candidate_steps = 2
tail_batch_trigger_size = global_batch_size
max_response_length= 8192
enable_float8_rollout = 0

# basic settings
experimental_name = "dapo_math"
total_epochs = 1
prompt_repeat_k = 16
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 2048
pack_max_length = 32768
train_optimizer_steps = 16
hf_interval = 50
enable_initial_evaluate = True
evaluate_step = 5

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
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
    enable_float8=enable_float8_rollout,
    context_length = max_response_length + max_prompt_length,
    rollout_max_batch_size_per_instance=512,
    allow_over_concurrency_ratio=4,
    rollout_timeout=7200.0,
    enable_return_routed_experts=enable_return_routed_experts,
    extra_rollout_config=dict(lmdeploy_log_level="ERROR", lmdeploy_uvicorn_log_level="ERROR"),
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.top_p = 0.7

# dataset
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path) if enable_evaluate else None
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer_config = RLTokenizeFnConfig(max_length=max_prompt_length)

train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
eval_dataset_cfg = [{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}] if enable_evaluate else []

dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, collator="fake_collator", pack_level="none")

# 3. judger
from xtuner.v1.utils.rl_test_utils import get_eos_token
eos_token_id = get_eos_token(model_path)
eos_token_str = tokenizer.convert_ids_to_tokens(eos_token_id)
dapomath_judger_config = DapoMathJudgerConfig(
    judger_name="dapo_math", 
    eos_token=eos_token_str,
    enable_overlong_buffer = True, 
    max_response_len =max_response_length, 
    overlong_buffer_len=4096, 
    overlong_penalty_factor=1.0, 
    tokenizer=tokenizer)
judger_cfg = JudgerConfig(reward_judger_configs=[dapomath_judger_config])

# 4. dataflow and evaluator
dataflow_config = DataFlowConfig(
    env=experimental_name,
    prompt_repeat_k=prompt_repeat_k,
    global_batch_size=global_batch_size,
    sample_params=training_sample_params,
    enable_partial_rollout=enbale_partial_rollout,
    staleness_threshold=staleness_threshold,
    tail_batch_candidate_steps=tail_batch_candidate_steps,
    tail_batch_trigger_size=tail_batch_trigger_size
)

def dapo_compute_metric(samples):
    return {"accuracy": sum(s.env.judger.reward["acc"] > 0 for s in samples) / len(samples)}


evaluator_cfg = EvaluatorConfig(
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_initial_evaluate,
    dataset_cfg=eval_dataset_cfg,
    tokenizer=tokenizer,
    evaluate_step=evaluate_step,
    compute_metric_func=dapo_compute_metric,
    sample_params=evaluation_sample_params,
    max_concurrent=1024,
) if enable_evaluate else None

def group_sample_filter_func(group_samples):
    rewards = [d.env.judger.reward["score"] for d in group_samples]
    if len(set(rewards)) == 1:
        print(f"filter all same reward sample: {rewards}")
        return []
    else:
        return group_samples

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=tokenizer,
    # postprocessor_func=group_sample_filter_func
)

# 5. Train worker
model_cfg = Qwen3MoE30BA3Config(freeze_routers=True)
optim_cfg = AdamWConfig(lr=1e-6, betas=(0.9, 0.95), max_grad_norm=1.0, weight_decay=0.1, foreach=False, skip_grad_norm_threshold=0.9, eps=1e-15)
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
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
train_worker_cfg: WorkerConfig = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=1,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# 6. RL Trainer
trainer = RLTrainerConfig(
    load_from=model_path,
    resources=resources,
    rollout_config=rollout_config,
    dataflow_config=dataflow_config,
    judger_config=judger_cfg,
    replay_buffer_config=replay_buffer_cfg,
    evaluator_config=evaluator_cfg,
    train_worker_config=train_worker_cfg,
    tokenizer_path=model_path,
    work_dir=work_dir,
    total_epochs=total_epochs,
    hf_interval=hf_interval,
)
