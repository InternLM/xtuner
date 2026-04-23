import os
from copy import deepcopy

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense8BConfig
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GEO3KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_evaluate = eval_data_path != ""
media_root = os.environ["MEDIA_ROOT"]

# basic settings
experimental_name = "grpo_geo3k"
total_epochs = 15
global_batch_size = 64
prompt_repeat_k = 5
rollout_tp_size = 2
rollout_ep_size = 1
max_prompt_length = 1024
max_response_length = 2048
pack_max_length = 32768
train_optimizer_steps = 4
hf_interval = 30
enable_initial_evaluate = True
evaluate_step = 15

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
    gpu_memory_utilization=0.75,
    context_length=max_response_length + max_prompt_length,
)

# sampling params
training_sample_params = SampleParams(max_tokens=max_response_length)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.top_p = 1.0
evaluation_sample_params.temperature = 0.0
evaluation_sample_params.top_k = 1

# 3. datasets
train_dataset_cfg = [
    {
        "dataset": DatasetConfig(
            name="geo3k",
            anno_path=data_path,
            class_name="VLMJsonlDataset",
            media_root=media_root,
            sample_ratio=1.0,
        ),
        "tokenize_fn": RLQwen3VLTokenizeFnConfig(processor_path=model_path, max_length=max_prompt_length),
    }
]
eval_dataset_cfg = [
    {
        "dataset": DatasetConfig(
            name="geo3k",
            anno_path=eval_data_path if enable_evaluate else data_path,
            class_name="VLMJsonlDataset",
            media_root=media_root,
            sample_ratio=1.0,
        ),
        "tokenize_fn": RLQwen3VLTokenizeFnConfig(
            processor_path=model_path,
            max_length=max_prompt_length,
            ignore_multimodal_info=True,
        ),
    }
]
dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
    num_workers=8,
)
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=eval_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
    num_workers=8,
)

# 4. judger
judger_config = GEO3KJudgerConfig()

# 5. train worker
model_cfg = Qwen3VLDense8BConfig(freeze_vision=True, freeze_projector=True)
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
fsdp_cfg = FSDPConfig(cpu_offload=False)
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
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=prompt_repeat_k),
    ),
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
        sampler_config=SamplerConfig(dataloader_cfg=eval_dataloader_cfg, prompt_repeat_k=1),
    ),
)

# 7. trainer
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
    total_epochs=total_epochs,
    train_batch_size=global_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_evaluate and enable_initial_evaluate,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    hf_interval=hf_interval,
    exp_tracker="jsonl",
)
