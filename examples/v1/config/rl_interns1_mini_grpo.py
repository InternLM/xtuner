import os
from copy import deepcopy

from transformers import AutoTokenizer
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.model.compose.intern_s1 import InternS1MiniConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.rl_trainer import RLTrainerConfig
from xtuner.v1.datasets import RLTokenizeFnConfig, DatasetConfig, InternS1VLTokenizeFnConfig, DataloaderConfig
from xtuner.v1.ray.judger.geo3k import GEO3KJudgerConfig


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_evaluate = True if eval_data_path != "" else False
media_root = os.environ["MEDIA_ROOT"]

# basic settings
experimental_name = "grpo_geo3k"
total_epochs = 15
global_batch_size = 1024
prompt_repeat_k = 5
rollout_tp_size = 2
rollout_ep_size = 1
max_prompt_length = 4096  # Note: 不设置大一点，大部分数据都会被过滤掉
max_response_length = 2048
pack_max_length = 32768
train_optimizer_steps = 4
hf_interval = 15
enable_initial_evaluate = True
evaluate_step = 10

# grpo quick test:
# total_epochs = 3
# global_batch_size = 64
# prompt_repeat_k = 5
# rollout_tp_size = 1
# rollout_ep_size = 1
# max_prompt_length = 512
# max_response_length = 1024
# pack_max_length = 32768
# train_optimizer_steps = 1
# hf_interval = 100
# enable_initial_evaluate = True
# evaluate_step = 15

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
    gpu_memory_utilization=0.75,
    context_length=max_prompt_length+max_response_length,
    extra_rollout_config={
            "sglang_grammar_backend": 'none',
        }
    # rollout_max_batch_size_per_instance=16,  # optional
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.top_p = 1.0
evaluation_sample_params.temperature = 0.0
evaluation_sample_params.top_k = 1

# dataset: 不需要修改
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path) if enable_evaluate else None
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

tokenize_fn_cfg = InternS1VLTokenizeFnConfig(model_cfg=InternS1MiniConfig(), max_length=max_prompt_length)
train_dataset_cfg = [
    {
            "dataset": DatasetConfig(name="geo3k",
                                     anno_path=data_path,
                                     class_name='VLMJsonlDataset',
                                     media_root=media_root,
                                     sample_ratio=1.0),
            "tokenize_fn": RLTokenizeFnConfig(max_length=max_prompt_length,
                                              tokenize_fn_cfg=tokenize_fn_cfg),
    }
]

eval_dataset_cfg = []
if enable_evaluate:
    eval_dataset_cfg = [
        {
            "dataset": DatasetConfig(name="geo3k",
                                     anno_path=eval_data_path,
                                     class_name='VLMJsonlDataset',
                                     media_root=media_root,
                                     sample_ratio=1.0),
            "tokenize_fn": RLTokenizeFnConfig(max_length=max_prompt_length,
                                              tokenize_fn_cfg=tokenize_fn_cfg,
                                              ignore_multimodal_info=True),
        }
    ]

dataloader_config = DataloaderConfig(num_workers=8,
                                     collator="fake_collator",
                                     pack_level="none")

# 3. judger
geo3k_judger_config = GEO3KJudgerConfig()
judger_cfg = JudgerConfig(reward_judger_configs=[geo3k_judger_config])

# 4. dataflow and evaluator
dataflow_config = DataFlowConfig(
    env=experimental_name,
    prompt_repeat_k=prompt_repeat_k,
    global_batch_size=global_batch_size,
    sample_params=training_sample_params,
    # max_concurrent=64, # optional 
)

evaluator_cfg = EvaluatorConfig(
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_initial_evaluate,
    dataset_cfg=eval_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=model_path,
    evaluate_step=evaluate_step,
    compute_metric_func=None,
    sample_params=evaluation_sample_params,
) if enable_evaluate else None

# replay buffer config: : 不需要修改
replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg, dataloader_cfg=dataloader_config, tokenizer=model_path
)

# 5. Train worker
# NOTE: modify model_cfg
model_cfg = InternS1MiniConfig(freeze_vision=True, freeze_projector=True)
optim_cfg = AdamWConfig(lr=1e-6, foreach=False)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.2,
        cliprange_low=0.2,
        loss_type="vanilla",
    ),
    ignore_idx=-100,
    use_kl_loss=True,
    kl_loss_coef=0.001,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False)
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
