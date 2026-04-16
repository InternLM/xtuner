"""RL Colocate Trainer 示例配置（GRPO + GSM8K）。

用法：通过环境变量传入路径后，由 CLI 加载本配置并 trainer_cfg.build().fit()。
需设置: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
可选: WORLD_SIZE, ENABLE_RETURN_ROUTED_EXPERTS, LOSS_TYPE, LOSS_MODE, SP_SIZE
"""
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense8BConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.judger import GEO3KJudgerConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.trainer.worker import WorkerConfig
from xtuner.v1.rl.agent_loop import AgentLoopManagerConfig, TaskSpecConfig, SingleTurnAgentLoopConfig, SyncProduceStrategyConfig, SamplerConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.train.rl_colocate_trainer import RLColocateTrainerConfig

# env
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")
NNODE = int(os.environ.get("WORLD_SIZE", "1"))
media_root = os.environ["MEDIA_ROOT"]

# basic settings
experimental_name = "grpo_geo3k"
rollout_steps = 45  # TODO: total_epoch
evaluate_step = 45
train_optimizer_steps = 4
global_batch_size = 1024
prompt_repeat_k = 5
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 1024
max_response_length = 2048
pack_max_length = 32 * 1024

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
judger_config = GEO3KJudgerConfig(num_ray_actors=1)

# 4. train worker
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)

# TODO: support get_model_config_from_hf
model_cfg = Qwen3VLDense8BConfig(freeze_vision=True, freeze_projector=True)

if hasattr(model_cfg.text_config, "balancing_loss_cfg"):
    model_cfg.text_config.balancing_loss_cfg = None
if hasattr(model_cfg.text_config, "z_loss_cfg"):
    model_cfg.text_config.z_loss_cfg = None
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

# 5. train agent loop manager
train_dataset_cfg = [
    {
            "dataset": DatasetConfig(name="geo3k",
                                     anno_path=data_path,
                                     class_name='VLMJsonlDataset',
                                     media_root=media_root,
                                     sample_ratio=1.0),
            "tokenize_fn": RLQwen3VLTokenizeFnConfig(processor_path=model_path, 
                                                     max_length=max_prompt_length),
    }
]

dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
    num_workers=8,
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
eval_dataset_cfg = [
    {
            "dataset": DatasetConfig(name="geo3k",
                                     anno_path=eval_data_path,
                                     class_name='VLMJsonlDataset',
                                     media_root=media_root,
                                     sample_ratio=1.0),
            "tokenize_fn": RLQwen3VLTokenizeFnConfig(processor_path=model_path, 
                                                     max_length=max_prompt_length,
                                                     ignore_multimodal_info=True),
    }
]

eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=eval_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
    num_workers=8,
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
    train_worker_cfg=train_worker_cfg,  # TODO: uniform naming of cfg and config
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    rollout_steps=rollout_steps,
    global_batch_size=global_batch_size,
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
)
