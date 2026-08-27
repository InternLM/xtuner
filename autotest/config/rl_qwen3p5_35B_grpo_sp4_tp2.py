import os

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig, RLTextTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import GEO3KJudgerConfig, GSM8KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
gsm8k_data_path = os.environ["DATA_PATH"]
gsm8k_eval_data_path = os.environ["EVAL_DATA_PATH"]
geo3k_data_path = os.environ["GEO3K_DATA_PATH"]
geo3k_eval_data_path = os.environ["GEO3K_EVAL_DATA_PATH"]
media_root = os.environ["MEDIA_ROOT"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")
NNODE = int(os.environ.get("WORLD_SIZE", "2"))

experimental_name = "qwen3p5_grpo_sp4_tp2"
total_train_steps = 16
evaluate_step = 16
train_optimizer_steps = 1
train_batch_size = 64 * train_optimizer_steps
prompt_repeat_k = 5
rollout_tp_size = 1
rollout_ep_size = 1
gsm8k_max_prompt_length = 1024
geo3k_max_prompt_length = 2048
max_response_length = 8192
max_prompt_length = geo3k_max_prompt_length
pack_max_length = 32 * 1024

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * NNODE,
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
    gpu_memory_utilization=0.8,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=(enable_return_routed_experts == "1"),
    extra_rollout_config=dict(
        lmdeploy_log_level="INFO",
        lmdeploy_uvicorn_log_level="INFO",
    ),
)

gsm8k_judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)
geo3k_judger_config = GEO3KJudgerConfig(
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)

lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, tp_size=2)
model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
model_cfg.compile_cfg = False
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.z_loss_cfg = None
optim_cfg = MuonConfig(lr=1e-6, weight_decay=0.1)
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
    sp_size=4,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

gsm8k_tokenize_fn = RLTextTokenizeFnConfig(max_length=gsm8k_max_prompt_length)
geo3k_tokenize_fn = RLQwen3VLTokenizeFnConfig(
    processor_path=model_path,
    max_length=geo3k_max_prompt_length,
    chat_template="qwen3.5-vl",
    add_generation_prompt=True,
    enable_thinking=True,
)
geo3k_eval_tokenize_fn = RLQwen3VLTokenizeFnConfig(
    processor_path=model_path,
    max_length=geo3k_max_prompt_length,
    chat_template="qwen3.5-vl",
    add_generation_prompt=True,
    enable_thinking=True,
    ignore_multimodal_info=True,
)
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
)

gsm8k_train_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="gsm8k", anno_path=gsm8k_data_path),
                "tokenize_fn": gsm8k_tokenize_fn,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=prompt_repeat_k,
)
geo3k_train_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(
                    name="geo3k",
                    anno_path=geo3k_data_path,
                    class_name="VLMJsonlDataset",
                    media_root=media_root,
                    sample_ratio=1.0,
                ),
                "tokenize_fn": geo3k_tokenize_fn,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
        num_workers=8,
    ),
    prompt_repeat_k=prompt_repeat_k,
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=[
        TaskSpecConfig(
            task_name="train_task:gsm8k",
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint=model_path,
                sample_params=training_sample_params,
            ),
            judger_config=gsm8k_judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=gsm8k_train_sampler_config,
        ),
        TaskSpecConfig(
            task_name="train_task:geo3k",
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint=model_path,
                sample_params=training_sample_params,
            ),
            judger_config=geo3k_judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=geo3k_train_sampler_config,
        ),
    ],
)

gsm8k_eval_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="gsm8k_eval", anno_path=gsm8k_eval_data_path, sample_ratio=1.0),
                "tokenize_fn": gsm8k_tokenize_fn,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=1,
)
geo3k_eval_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(
                    name="geo3k_eval",
                    anno_path=geo3k_eval_data_path,
                    class_name="VLMJsonlDataset",
                    media_root=media_root,
                    sample_ratio=1.0,
                ),
                "tokenize_fn": geo3k_eval_tokenize_fn,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
        num_workers=8,
    ),
    prompt_repeat_k=1,
)

eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=[
        TaskSpecConfig(
            task_name="eval_task:gsm8k",
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint=model_path,
                sample_params=evaluation_sample_params,
            ),
            judger_config=gsm8k_judger_config,
            sampler_config=gsm8k_eval_sampler_config,
        ),
        TaskSpecConfig(
            task_name="eval_task:geo3k",
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint=model_path,
                sample_params=evaluation_sample_params,
            ),
            judger_config=geo3k_judger_config,
            sampler_config=geo3k_eval_sampler_config,
        ),
    ],
)

evaluator_config = EvaluatorConfig(compute_metric_func=None)

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
    total_train_steps=total_train_steps,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    sync_weights_interval=2,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
    exp_tracker="jsonl",
)
