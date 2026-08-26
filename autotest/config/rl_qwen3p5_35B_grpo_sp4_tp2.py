import json
import os

from transformers import AutoTokenizer

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
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
from xtuner.v1.rl.judger import DapoMathJudgerConfig, GSM8KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig, get_eos_token
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _math_dapo_annotation(meta_path: str) -> str:
    with open(meta_path, encoding="utf-8") as f:
        ds_collections = json.load(f)
    if "math_dapo" not in ds_collections:
        raise KeyError(f"math_dapo not found in {meta_path}, keys={list(ds_collections)}")
    annotations = _as_list(ds_collections["math_dapo"]["annotation"])
    if not annotations:
        raise ValueError(f"math_dapo.annotation is empty in {meta_path}")
    return annotations[0]


def _build_sampler(anno_path: str, name: str, tokenize_fn, prompt_repeat_k: int, sample_ratio: float | None = None):
    dataset_kwargs = {"name": name, "anno_path": anno_path}
    if sample_ratio is not None:
        dataset_kwargs["sample_ratio"] = sample_ratio
    return SamplerConfig(
        dataloader_cfg=DataloaderConfig(
            dataset_config_list=[
                {
                    "dataset": DatasetConfig(**dataset_kwargs),
                    "tokenize_fn": tokenize_fn,
                }
            ],
            pack_max_length=pack_max_length,
            collator="fake_collator",
            pack_level="none",
        ),
        prompt_repeat_k=prompt_repeat_k,
    )


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
gsm8k_data_path = os.environ["DATA_PATH"]
gsm8k_eval_data_path = os.environ["EVAL_DATA_PATH"]
dapo_meta_path = os.environ["DAPO_META_PATH"]
dapo_data_path = os.environ.get("DAPO_DATA_PATH") or _math_dapo_annotation(dapo_meta_path)
dapo_eval_meta_path = os.environ.get("DAPO_EVAL_META_PATH", "")
dapo_eval_data_path = os.environ.get("DAPO_EVAL_DATA_PATH", "")
if not dapo_eval_data_path and dapo_eval_meta_path:
    dapo_eval_data_path = _math_dapo_annotation(dapo_eval_meta_path)
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")
NNODE = int(os.environ.get("WORLD_SIZE", "2"))

experimental_name = "qwen3p5_grpo_sp4_tp2"
total_train_steps = 16
evaluate_step = 16
train_optimizer_steps = 1
train_batch_size = 64 * train_optimizer_steps
gsm8k_prompt_repeat_k = 5
dapo_prompt_repeat_k = 5
rollout_tp_size = 1
rollout_ep_size = 1
gsm8k_max_prompt_length = 512
dapo_max_prompt_length = 2048
max_response_length = 8192
max_prompt_length = dapo_max_prompt_length
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

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
eos_token_id = get_eos_token(model_path)
eos_token_str = tokenizer.convert_ids_to_tokens(eos_token_id)
gsm8k_judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)
dapo_judger_config = DapoMathJudgerConfig(
    judger_name="dapo_math",
    eos_token=eos_token_str,
    enable_overlong_buffer=True,
    max_response_len=max_response_length,
    overlong_buffer_len=4096,
    overlong_penalty_factor=1.0,
    tokenizer=tokenizer,
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)

lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, tp_size=2)
model_cfg = Qwen3_5_VLMoE35BA3Config()
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
dapo_tokenize_fn = RLTextTokenizeFnConfig(max_length=dapo_max_prompt_length)

gsm8k_train_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    ),
)
dapo_train_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    ),
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=[
        TaskSpecConfig(
            task_name="train_task:gsm8k",
            weight=1.0,
            agent_loop_config=gsm8k_train_agent_loop_config,
            judger_config=gsm8k_judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=_build_sampler(
                gsm8k_data_path,
                "gsm8k",
                gsm8k_tokenize_fn,
                gsm8k_prompt_repeat_k,
            ),
        ),
        TaskSpecConfig(
            task_name="train_task:dapo_math",
            weight=1.0,
            agent_loop_config=dapo_train_agent_loop_config,
            judger_config=dapo_judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=_build_sampler(
                dapo_data_path,
                "dapo_math",
                dapo_tokenize_fn,
                dapo_prompt_repeat_k,
            ),
        ),
    ],
)

eval_tasks = [
    TaskSpecConfig(
        task_name="eval_task:gsm8k",
        weight=1.0,
        agent_loop_config=SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=SampleParams(
                max_tokens=max_response_length,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                min_tokens=0,
            ),
        ),
        judger_config=gsm8k_judger_config,
        sampler_config=_build_sampler(
            gsm8k_eval_data_path,
            "gsm8k_eval",
            gsm8k_tokenize_fn,
            prompt_repeat_k=1,
            sample_ratio=1.0,
        ),
    )
]
if dapo_eval_data_path:
    eval_tasks.append(
        TaskSpecConfig(
            task_name="eval_task:dapo_math",
            weight=1.0,
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint=model_path,
                sample_params=SampleParams(
                    max_tokens=max_response_length,
                    top_k=1,
                    top_p=1.0,
                    temperature=0.0,
                    min_tokens=0,
                ),
            ),
            judger_config=dapo_judger_config,
            sampler_config=_build_sampler(
                dapo_eval_data_path,
                "dapo_math_eval",
                dapo_tokenize_fn,
                prompt_repeat_k=1,
                sample_ratio=1.0,
            ),
        )
    )
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(tasks=eval_tasks)

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
