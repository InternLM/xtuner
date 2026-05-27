import json
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import AgentInSandboxLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import RolloutImportanceSampling, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig
from recipe.tb2_eval.xtuner_dataset import RLTB2EvalTokenizeFnConfig
from recipe.tb2_rl.xtuner_dataset import RLTB2RLTokenizeFnConfig

def _as_list(value):
    return value if isinstance(value, list) else [value]

work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
model_name = os.environ["MODEL_NAME"]
meta_data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ.get("EVAL_DATA_PATH", "")
eval_media_root = os.environ.get("EVAL_MEDIA_ROOT", "")

debug_rollout_dir = os.environ.get("DEBUG_ROLLOUT_DIR", "")
debug_train = os.environ.get("DEBUG_TRAIN", False)
debug_rollout = os.environ.get("DEBUG_ROLLOUT", False)

enable_evaluate = eval_data_path is not None and eval_data_path != ""

# basic settings
experimental_name = "tb2_rl"
total_epochs = 15
global_batch_size = 256
prompt_repeat_k = 8
rollout_tp_size = 2
rollout_ep_size = 1
max_prompt_length = 2048
max_response_length = 65536
pack_max_length = max_response_length+max_prompt_length
train_optimizer_steps = 8
hf_interval = 15

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

# 2. rollout
rollout_config = RolloutConfig(
    model_name=model_name,
    fp32_lm_head=True,
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.8,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=True,
    rollout_max_batch_size_per_instance=512,
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
    return_routed_experts=True,
    return_logprob=True,
    return_token_ids=True,
)
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=1,
    top_p=1.0,
    temperature=0.0,
    min_tokens=0,
    return_routed_experts=False,
    return_logprob=False,
    return_token_ids=False,
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
                "tokenize_fn": RLTB2RLTokenizeFnConfig(
                    max_length=max_prompt_length
                ),
            }
        )

if enable_evaluate:
    with open(eval_data_path, "r", encoding="utf-8") as f:
        ds_collections = json.load(f)

    eval_dataset_cfg = []
    for name, data in ds_collections.items():
        annotations = _as_list(data["annotation"])
        for annotation in annotations:
            eval_dataset_cfg.append(
                {
                    "dataset": DatasetConfig(
                        name=f"{experimental_name}_eval",
                        anno_path=annotation,
                        media_root=data.get("eval_media_root", ""),
                        sample_ratio=data.get("sample_ratio", 1.0),
                        class_name="VLMJsonlDataset",
                    ),
                    "tokenize_fn": RLTB2EvalTokenizeFnConfig(
                        max_length=max_prompt_length
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

# 5. train worker
model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
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
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, fp32_lm_head=True)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=2,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# 6. agent loop managers
agent_loop_config = AgentInSandboxLoopConfig(
    hf_checkpoint=model_path,
    max_concurrent_samples=512,
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=prompt_repeat_k),
    ),
)

if enable_evaluate:
    eval_agent_loop_config = AgentInSandboxLoopConfig(
        hf_checkpoint=model_path,
        max_concurrent_samples=512,
        mode="eval",
        sample_params=evaluation_sample_params,
    )
    eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
        tasks=TaskSpecConfig(
            task_name="eval_task",
            agent_loop_config=eval_agent_loop_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=SamplerConfig(dataloader_cfg=eval_dataloader_cfg, prompt_repeat_k=1),
        ),
    )
    enable_evaluate = True
else:
    eval_agent_loop_manager_cfg = None
    enable_evaluate = False

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
    enable_initial_evaluate=False,
    evaluate_step=1,
    work_dir=work_dir,
    hf_interval=hf_interval,
    debug_rollout_dir=debug_rollout_dir,
    debug_train=debug_train,
    debug_rollout=debug_rollout,
)
