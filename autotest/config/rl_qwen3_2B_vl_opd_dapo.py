import json
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3VLDense2BConfig
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherConfig,
    RolloutTeacherLaunchConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _dataset_cfg(meta_path, model_path, max_prompt_length, source_name, ignore_multimodal_info=False):
    with open(meta_path, encoding="utf-8") as f:
        ds_collections = json.load(f)
    if source_name not in ds_collections:
        raise KeyError(f"Data source {source_name!r} is missing from {meta_path}")
    data = ds_collections[source_name]
    dataset_cfg = []
    for annotation in _as_list(data["annotation"]):
        dataset_cfg.append(
            {
                "dataset": DatasetConfig(
                    name=source_name,
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
                    ignore_multimodal_info=ignore_multimodal_info,
                ),
            }
        )
    return dataset_cfg


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
teacher_model_path = os.environ["TEACHER_MODEL_PATH"]
meta_data_path = os.environ["DATA_PATH"]
eval_meta_data_path = os.environ.get("EVAL_DATA_PATH", "")
nnode = int(os.environ.get("WORLD_SIZE", "1"))
# Meta key in opd_*_meta.json. Samples inside that file use openai/gsm8k.
data_source_name = os.environ.get("OPD_DATA_SOURCE", "gsm8k")
teacher_data_source = os.environ.get("OPD_TEACHER_DATA_SOURCE", "openai/gsm8k")

experimental_name = "qwen3_2b_opd_k1_single_teacher"
total_epochs = 15
total_train_steps = 50
train_batch_size = 128
teacher_num_replicas = 1
teacher_replica_num_workers = 1
enable_prefix_caching = True
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

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    # 8 GPUs minus the one reserved for the single teacher, unless the launcher overrides it.
    num_workers=int(os.environ.get("XTUNER_RL_NUM_WORKERS", str(7 * nnode))),
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

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

lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, reduce_dtype="float32")
model_cfg = Qwen3VLDense2BConfig()
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None
optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1, betas=(0.9, 0.98))
loss_cfg = DistillationLossConfig(
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
    loss_mode="k1",
    use_policy_gradient=True,
    task_adv_weight=0.0,
    distillation_loss_weight=1.0,
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

train_dataset_cfg = _dataset_cfg(meta_data_path, model_path, max_prompt_length, data_source_name)
dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    num_workers=8,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
sampler_config = SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=prompt_repeat_k)
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    min_tokens=0,
    skip_special_tokens=False,
    return_logprob=True,
    return_token_ids=True,
    return_routed_experts=False,
)
judger_config = GSM8KJudgerConfig(
    judger_name=teacher_data_source,
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=SingleTurnAgentLoopConfig(hf_checkpoint=model_path, sample_params=training_sample_params),
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=sampler_config,
    ),
)

eval_agent_loop_manager_cfg = None
evaluator_config = None
if enable_evaluate:
    eval_dataset_cfg = _dataset_cfg(
        eval_meta_data_path,
        model_path,
        max_prompt_length,
        data_source_name,
        ignore_multimodal_info=True,
    )
    eval_dataloader_cfg = DataloaderConfig(
        dataset_config_list=eval_dataset_cfg,
        num_workers=8,
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    )
    evaluation_sample_params = SampleParams(
        max_tokens=max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_tokens=0,
        skip_special_tokens=False,
        return_logprob=True,
        return_token_ids=True,
        return_routed_experts=False,
    )
    eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
        tasks=TaskSpecConfig(
            task_name="eval_task",
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint=model_path, sample_params=evaluation_sample_params
            ),
            judger_config=judger_config,
            sampler_config=SamplerConfig(dataloader_cfg=eval_dataloader_cfg, prompt_repeat_k=eval_prompt_repeat_k),
        ),
    )
    evaluator_config = EvaluatorConfig()

distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(
            name="teacher",
            num_replicas=teacher_num_replicas,
            enable_prefix_caching=enable_prefix_caching,
            launch_config=RolloutTeacherLaunchConfig(
                model_path=teacher_model_path,
                num_workers=teacher_replica_num_workers,
                server_port=13141,
                dtype="bfloat16",
                tensor_parallel_size=teacher_replica_num_workers,
                expert_parallel_size=1,
                context_length=max_num_tokens,
                max_batch_size=max_num_tokens,
                gpu_memory_utilization=0.8,
            ),
        ),
    ],
    data_source_teacher_map={teacher_data_source: "teacher"},
)

trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    distillation_config=distillation_config,
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_evaluate,
    evaluate_step=evaluate_step,
    total_train_steps=total_train_steps,
    total_epochs=total_epochs,
    work_dir=work_dir,
    seed=1234,
    debug_rollout=False,
    exp_tracker="jsonl",
)
