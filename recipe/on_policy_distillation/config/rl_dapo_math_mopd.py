import json
import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig, GEO3KJudgerConfig, GSM8KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.on_policy_distillation import (
    OPDConfig,
    OPDTeacherConfig,
    OPDTeacherLaunchConfig,
)
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import (
    AcceleratorResourcesConfig,
    CPUResourcesConfig,
)
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
gsm8k_teacher_model_path = os.environ["GSM8K_TEACHER_MODEL_PATH"]
geo3k_teacher_model_path = os.environ["GEO3K_TEACHER_MODEL_PATH"]
meta_data_path = os.environ["DATA_PATH"]
eval_meta_data_path = os.environ.get("EVAL_DATA_PATH", "")
NNODE = int(os.environ.get("WORLD_SIZE", "1"))


def _as_list(value):
    return value if isinstance(value, list) else [value]


# Training shape aligned with verl PR #6051:
# examples/on_policy_distillation_trainer/run_qwen3_mopd_gsm8k_geo3k.sh.
# Teacher roles and model families follow the GSM8K/Geo3K experiment.
experimental_name = "dapo_math_mopd"
total_epochs = 15
train_batch_size = 128
prompt_repeat_k = 1
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 1024
max_response_length = 2048
pack_max_length = max_prompt_length + max_response_length
max_num_tokens = pack_max_length
train_optimizer_steps = 1
enable_evaluate = bool(eval_meta_data_path)
evaluate_step = 5
eval_prompt_repeat_k = 1
checkpoint_interval = 200

# 1. resources: two colocated Student workers, plus one GPU per Teacher.
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=2 * NNODE,
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
    gpu_memory_utilization=0.6,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=False,
    rollout_max_batch_size_per_instance=2048,
)

# 3. train worker
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=1,
    reduce_dtype="float32",
)
model_cfg = get_model_config_from_hf(Path(model_path))
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None
optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1, betas=(0.9, 0.98))
loss_cfg = GRPOLossConfig(
    policy_loss_cfg={
        "cliprange_high": 0.2,
        "cliprange_low": 0.2,
        "loss_type": "vanilla",
        "clip_ratio_c": 10.0,
        "log_prob_diff_min": -20.0,
        "log_prob_diff_max": 20.0,
    },
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
)
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

# 4. train agent loop manager
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
                    chat_template="qwen3-vl",
                    add_generation_prompt=True,
                    enable_thinking=True,
                ),
            }
        )

dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    num_workers=8,
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
    skip_special_tokens=False,
    return_routed_experts=False,
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
        produce_strategy_config=produce_strategy_config,
        sampler_config=sampler_config,
    ),
)

# 5. evaluation
eval_agent_loop_manager_cfg = None
evaluator_config = None
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
                        chat_template="qwen3-vl",
                        add_generation_prompt=True,
                        enable_thinking=True,
                        ignore_multimodal_info=True,
                    ),
                }
            )

    eval_judger_config = ComposedJudgerConfig(
        branches={
            "openai/gsm8k": GSM8KJudgerConfig(
                judger_name="openai/gsm8k",
                cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
            ),
            "hiyouga/geometry3k": GEO3KJudgerConfig(
                cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
            ),
        }
    )
    eval_dataloader_cfg = DataloaderConfig(
        dataset_config_list=eval_dataset_cfg,
        num_workers=8,
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    )
    eval_sampler_config = SamplerConfig(
        dataloader_cfg=eval_dataloader_cfg,
        prompt_repeat_k=eval_prompt_repeat_k,
    )
    evaluation_sample_params = SampleParams(
        max_tokens=max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
        skip_special_tokens=False,
        return_routed_experts=False,
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
    evaluator_config = EvaluatorConfig()

# 6. multi-teacher pure on-policy distillation
#
# This is the only topology block that needs to be edited for an experiment.
# Every training record's data_source must have an entry in
# data_source_teacher_map. Teacher model paths are read from
# GSM8K_TEACHER_MODEL_PATH and GEO3K_TEACHER_MODEL_PATH.
opd_config = OPDConfig(
    mode="pg-opd",
    task_adv_weight=0.0,
    opd_adv_weight=1.0,
    teachers=[
        OPDTeacherConfig(
            name="gsm8k_teacher",
            endpoint="http://127.0.0.1:13141",
            launch_config=OPDTeacherLaunchConfig(
                model_path=gsm8k_teacher_model_path,
                cuda_visible_devices="6",
                tensor_parallel_size=1,
                expert_parallel_size=1,
                context_length=max_num_tokens,
                max_batch_size=max_num_tokens,
                gpu_memory_utilization=0.8,
            ),
        ),
        OPDTeacherConfig(
            name="geo3k_teacher",
            endpoint="http://127.0.0.1:13142",
            launch_config=OPDTeacherLaunchConfig(
                model_path=geo3k_teacher_model_path,
                cuda_visible_devices="7",
                tensor_parallel_size=1,
                expert_parallel_size=1,
                context_length=max_num_tokens,
                max_batch_size=max_num_tokens,
                gpu_memory_utilization=0.8,
            ),
        ),
    ],
    data_source_teacher_map={
        "openai/gsm8k": "gsm8k_teacher",
        "hiyouga/geometry3k": "geo3k_teacher",
    },
)

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
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    opd_config=opd_config,
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_evaluate,
    evaluate_step=evaluate_step,
    total_epochs=total_epochs,
    work_dir=work_dir,
    seed=1234,
    debug_rollout=False,
)
