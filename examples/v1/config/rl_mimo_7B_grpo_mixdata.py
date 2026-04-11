import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets import DataloaderConfig, DatasetConfig, RLTokenizeFnConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.ray.base import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.base.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.rl_trainer import RLTrainerConfig
from xtuner.v1.rl.config.advantage import GRPOAdvantageConfig

work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
meta_data_path = os.environ["DATA_PATH"]

# basic settings
experimental_name = "mimo_grpo_mixdata"
total_epochs = 15
global_batch_size = 32
prompt_repeat_k = 8
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 4096
max_response_length = 8192
pack_max_length = 32768
train_optimizer_steps = 1
hf_interval = 150

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.7,
    context_length=max_prompt_length + max_response_length,
    rollout_max_batch_size_per_instance=512,
    extra_rollout_config={"sglang_speculative_algorithm": "EAGLE",
                          "sglang_speculative_num_steps": 3,
                          "sglang_speculative_eagle_topk": 1,
                          "sglang_speculative_num_draft_tokens": 4,
                          'sglang_log_level': 'info',
                          'sglang_log_level_http': 'info',
                          },
)

training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
ds_collections = json.loads(Path(meta_data_path).read_text())

train_dataset_cfg = []
for name, data_cfg in ds_collections.items():
    train_dataset_cfg.append(
        {
            "dataset": DatasetConfig(
                name=name,
                anno_path=data_cfg["annotation"],
                sample_ratio=data_cfg.get("sample_ratio", 1.0),
                class_name="JsonlDataset",
            ),
            "tokenize_fn": RLTokenizeFnConfig(max_length=max_prompt_length),
        }
    )

dataloader_config = DataloaderConfig(
    num_workers=8,
    collator="fake_collator",
    pack_level="none",
)

# 3. judger
from xtuner.v1.utils.rl_test_utils import get_eos_token

eos_token_id = get_eos_token(model_path)
eos_token_str = tokenizer.convert_ids_to_tokens(eos_token_id)
dapomath_judger_config = DapoMathJudgerConfig(
    judger_name="dapo_math",
    eos_token=eos_token_str,
    enable_overlong_buffer=True,
    max_response_len=max_response_length,
    overlong_buffer_len=4096,
    overlong_penalty_factor=1.0,
    tokenizer=tokenizer,
)
judger_cfg = JudgerConfig(reward_judger_configs=[dapomath_judger_config])

# 4. dataflow
dataflow_config = DataFlowConfig(
    env=experimental_name,
    prompt_repeat_k=prompt_repeat_k,
    global_batch_size=global_batch_size,
    sample_params=training_sample_params,
)

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=tokenizer,
)

# 5. Train worker
model_cfg = get_model_config_from_hf(Path(model_path))
if getattr(model_cfg, "mtp_config", None) is not None:
    model_cfg.mtp_config.loss_scaling_factor = 0.2

optim_cfg = AdamWConfig(lr=1e-6, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False)
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
    chunk_size=512
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
train_worker_cfg = WorkerConfig(
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
    train_worker_config=train_worker_cfg,
    tokenizer_path=model_path,
    work_dir=work_dir,
    total_epochs=total_epochs,
    hf_interval=hf_interval,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
)
