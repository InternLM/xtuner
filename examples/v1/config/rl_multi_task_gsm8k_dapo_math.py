"""RL Colocate Trainer 示例配置（Multi-Task: GSM8K + DAPO Math）。

需设置环境变量：
    WORK_DIR
    MODEL_PATH
    GSM8K_DATA_PATH
    GSM8K_EVAL_DATA_PATH
    DAPO_DATA_PATH
    DAPO_EVAL_DATA_PATH

可选环境变量：
    WORLD_SIZE
    ENABLE_RETURN_ROUTED_EXPERTS
    LOSS_TYPE
    LOSS_MODE
    SP_SIZE
    GSM8K_TASK_WEIGHT
    DAPO_TASK_WEIGHT
"""

import os
from pathlib import Path

from transformers import AutoTokenizer

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.agent_loop import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SingleTurnAgentLoopConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import DapoMathJudgerConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, get_eos_token
from xtuner.v1.train.rl_colocate_trainer import RLColocateTrainerConfig

work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
gsm8k_data_path = os.environ["GSM8K_DATA_PATH"]
gsm8k_eval_data_path = os.environ["GSM8K_EVAL_DATA_PATH"]
dapo_data_path = os.environ["DAPO_DATA_PATH"]
dapo_eval_data_path = os.environ["DAPO_EVAL_DATA_PATH"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

experimental_name = "multi_task_gsm8k_dapo_math"
rollout_steps = 50
evaluate_step = 5
train_optimizer_steps = 8
global_batch_size = 128
gsm8k_task_weight = float(os.environ.get("GSM8K_TASK_WEIGHT", "1.0"))
dapo_task_weight = float(os.environ.get("DAPO_TASK_WEIGHT", "1.0"))
rollout_tp_size = 1
rollout_ep_size = 1
gsm8k_prompt_repeat_k = 5
dapo_prompt_repeat_k = 8
gsm8k_max_prompt_length = 512
dapo_max_prompt_length = 2048
gsm8k_max_response_length = 1024
dapo_max_response_length = 8192
max_prompt_length = dapo_max_prompt_length
max_response_length = dapo_max_response_length
pack_max_length = 32768

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
    rollout_max_batch_size_per_instance=2048,
)

eos_token_id = get_eos_token(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
eos_token_str = tokenizer.convert_ids_to_tokens(eos_token_id)
judger_config = DapoMathJudgerConfig(
    judger_name="dapo_math",
    eos_token=eos_token_str,
    enable_overlong_buffer=True,
    max_response_len=max_response_length,
    overlong_buffer_len=4096,
    overlong_penalty_factor=1.0,
    tokenizer=tokenizer,
)

lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
model_cfg = get_model_config_from_hf(Path(model_path))
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None
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

gsm8k_train_tokenizer_config = RLTextTokenizeFnConfig(max_length=gsm8k_max_prompt_length)
dapo_train_tokenizer_config = RLTextTokenizeFnConfig(max_length=dapo_max_prompt_length)

gsm8k_train_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="gsm8k", anno_path=gsm8k_data_path),
                "tokenize_fn": gsm8k_train_tokenizer_config,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=gsm8k_prompt_repeat_k,
)
dapo_train_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="dapo_math", anno_path=dapo_data_path),
                "tokenize_fn": dapo_train_tokenizer_config,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=dapo_prompt_repeat_k,
)

gsm8k_train_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=gsm8k_max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    ),
)
dapo_train_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=dapo_max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    ),
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=[
        TaskSpecConfig(
            task_name="train_task:dapo_math",
            weight=dapo_task_weight,
            agent_loop_config=dapo_train_agent_loop_config,
            judger_config=judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=dapo_train_sampler_config,
        ),
        TaskSpecConfig(
            task_name="train_task:gsm8k",
            weight=gsm8k_task_weight,
            agent_loop_config=gsm8k_train_agent_loop_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=gsm8k_train_sampler_config,
        ),
    ],
)

gsm8k_eval_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="gsm8k_eval", anno_path=gsm8k_eval_data_path, sample_ratio=1.0),
                "tokenize_fn": gsm8k_train_tokenizer_config,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=1,
)
dapo_eval_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="dapo_math_eval", anno_path=dapo_eval_data_path, sample_ratio=1.0),
                "tokenize_fn": dapo_train_tokenizer_config,
            }
        ],
        pack_max_length=pack_max_length,
        collator="fake_collator",
        pack_level="none",
    ),
    prompt_repeat_k=1,
)

gsm8k_eval_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=gsm8k_max_response_length,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        min_tokens=0,
    ),
)
dapo_eval_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=dapo_max_response_length,
        top_k=1,
        top_p=0.7,
        temperature=0.0,
        min_tokens=0,
    ),
)

eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=[
        TaskSpecConfig(
            task_name="eval_task:dapo_math",
            weight=dapo_task_weight,
            agent_loop_config=dapo_eval_agent_loop_config,
            judger_config=judger_config,
            sampler_config=dapo_eval_sampler_config,
        ),
        TaskSpecConfig(
            task_name="eval_task:gsm8k",
            weight=gsm8k_task_weight,
            agent_loop_config=gsm8k_eval_agent_loop_config,
            sampler_config=gsm8k_eval_sampler_config,
        ),
    ],
)


def compute_metric(samples):
    return {"accuracy": sum(sample.reward["acc"] > 0 for sample in samples) / len(samples)}


evaluator_config = EvaluatorConfig(compute_metric_func=compute_metric)

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
    global_batch_size=global_batch_size,
    enable_evaluate=True,
    enable_initial_evaluate=False,
    rollout_steps=rollout_steps,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
)
