"""Qwen3-VL-2B multi-teacher OPD e2e config (50 steps).

Step-1/2 metric baselines of the reference run (full precision in
``tests/rl/test_opd_two_step_accuracy.py::MOPD_GOLDEN``):

  distillation/reduced_distillation_kl        step1=0.3442  step2=0.3924
  distillation/reduced_distillation_abs_loss  step1=0.4967  step2=0.5339
  response/rewards/mean                       step1=0.4477  step2=0.4703

Training shape (batch / prompt / response / eval cadence / seed) matches that
reference run. API is rewritten for the post-refactor distillation stack in this repo:

  legacy (xtuner-opd)              -> current (this branch)
  --------------------------------   --------------------------------
  OPDConfig / OPDTeacher*          -> DistillationConfig / RolloutTeacher*
  GRPOLossConfig + mode=pg-opd     -> DistillationLossConfig(loss_mode=k1,
                                       use_policy_gradient=True,
                                       task_adv_weight=0.0,
                                       distillation_loss_weight=1.0)
  trainer.opd_config               -> trainer.distillation_config
  reduced_llm_loss (log)           -> distillation/reduced_distillation_kl
                                       (+ distillation/opd_reverse_kl)
  rewards/mean                     -> rewards/mean (unchanged)
  eval gsm8k/score, geo3k/score    -> eval/gsm8k/acc, eval/geo3k/acc
"""

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
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig, GEO3KJudgerConfig, GSM8KJudgerConfig
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherConfig,
    RolloutTeacherLaunchConfig,
)
from xtuner.v1.rl.loss import DistillationLossConfig
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


EVAL_DATA_SOURCE_TAGS = {
    "openai/gsm8k": "gsm8k",
    "hiyouga/geometry3k": "geo3k",
}


def mopd_compute_metric(samples):
    """Emit per-source accuracy so TensorBoard logs eval/gsm8k/acc, eval/geo3k/acc."""
    scores_by_source = {data_source: [] for data_source in EVAL_DATA_SOURCE_TAGS}
    for sample in samples:
        data_source = sample.extra_fields.get("origin_data_source")
        if data_source not in scores_by_source:
            raise ValueError(f"Unexpected evaluation data source: {data_source!r}")

        reward = sample.reward or {}
        if "score" not in reward:
            raise ValueError(f"Missing reward score for evaluation data source: {data_source!r}")
        scores_by_source[data_source].append(float(reward["score"]))

    missing_sources = [data_source for data_source, scores in scores_by_source.items() if not scores]
    if missing_sources:
        raise ValueError(f"Missing evaluation samples for data sources: {missing_sources}")

    all_scores = [score for scores in scores_by_source.values() for score in scores]
    metrics = {"score": sum(all_scores) / len(all_scores)}
    for data_source, tag in EVAL_DATA_SOURCE_TAGS.items():
        source_scores = scores_by_source[data_source]
        metrics[f"{tag}/acc"] = sum(source_scores) / len(source_scores)
    return metrics


# Training shape aligned with baseline 20260731 mopd run (batch=128, eval every 5).
experimental_name = "qwen3_vl_2b_mopd_e2e"
total_epochs = 15
total_train_steps = int(os.environ.get("TOTAL_TRAIN_STEPS", "50"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))
# Baseline used one replica per teacher (1 GPU each); keep that for 8-GPU e2e.
teacher_num_replicas = int(os.environ.get("MOPD_TEACHER_NUM_REPLICAS", "1"))
teacher_replica_num_workers = int(os.environ.get("MOPD_TEACHER_REPLICA_NUM_WORKERS", "1"))
enable_prefix_caching = os.environ.get("MOPD_ENABLE_PREFIX_CACHING", "0") == "1"
prompt_repeat_k = 1
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 1024
max_response_length = 2048
pack_max_length = max_prompt_length + max_response_length
max_num_tokens = pack_max_length
train_optimizer_steps = 1
enable_evaluate = bool(eval_meta_data_path)
evaluate_step = int(os.environ.get("EVALUATE_STEP", "5"))
eval_prompt_repeat_k = 1
checkpoint_interval = 200

# 1. resources: colocated Student workers; Teacher GPUs are reserved outside Ray.
# Baseline 1x8: 2 Teacher GPUs + 6 Student GPUs. Prefer launcher-computed
# XTUNER_RL_NUM_WORKERS when present (after Teacher placement).
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=int(os.environ.get("XTUNER_RL_NUM_WORKERS", str(6 * NNODE))),
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
# PG-OPD (sampled-token k1) requires identity sampling + logprob/token ids.
# DistillationConfig.validate_trainer enforces this via validate_opd_sample_params.
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
agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
)
produce_strategy_config = SyncProduceStrategyConfig()
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=ComposedJudgerConfig(
            branches={
                "openai/gsm8k": GSM8KJudgerConfig(
                    judger_name="openai/gsm8k",
                    cpu_resources=CPUResourcesConfig(
                        num_workers=1,
                        num_cpus_per_worker=1,
                    ),
                ),
                "hiyouga/geometry3k": GEO3KJudgerConfig(
                    cpu_resources=CPUResourcesConfig(
                        num_workers=1,
                        num_cpus_per_worker=1,
                    ),
                ),
            }
        ),
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
                cpu_resources=CPUResourcesConfig(
                    num_workers=1,
                    num_cpus_per_worker=1,
                ),
            ),
            "hiyouga/geometry3k": GEO3KJudgerConfig(
                cpu_resources=CPUResourcesConfig(
                    num_workers=1,
                    num_cpus_per_worker=1,
                ),
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
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_tokens=0,
        skip_special_tokens=False,
        return_logprob=True,
        return_token_ids=True,
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
    evaluator_config = EvaluatorConfig(compute_metric_func=mopd_compute_metric)

# 6. multi-teacher pure on-policy distillation (sampled-token k1).
# loss_config must be the same object as train_worker_cfg.loss_cfg.
# data_source keys must match RolloutState.extra_fields["origin_data_source"].
distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(
            name="gsm8k_teacher",
            num_replicas=teacher_num_replicas,
            enable_prefix_caching=enable_prefix_caching,
            launch_config=RolloutTeacherLaunchConfig(
                model_path=gsm8k_teacher_model_path,
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
        RolloutTeacherConfig(
            name="geo3k_teacher",
            num_replicas=teacher_num_replicas,
            enable_prefix_caching=enable_prefix_caching,
            launch_config=RolloutTeacherLaunchConfig(
                model_path=geo3k_teacher_model_path,
                num_workers=teacher_replica_num_workers,
                server_port=13142,
                dtype="bfloat16",
                tensor_parallel_size=teacher_replica_num_workers,
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
)
