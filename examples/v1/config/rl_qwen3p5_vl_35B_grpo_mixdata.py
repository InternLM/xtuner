import json
import os

from transformers import AutoTokenizer

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import AgentLoopManagerConfig, SamplerConfig, SyncProduceStrategyConfig, TaskSpecConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import DapoMathJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout import RolloutEndpointConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import RolloutImportanceSampling, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, get_eos_token
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


def _as_list(value):
    return value if isinstance(value, list) else [value]


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
meta_data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ.get("EVAL_DATA_PATH", "")
eval_media_root = os.environ.get("EVAL_MEDIA_ROOT", "")

debug_rollout_dir = os.environ.get("DEBUG_ROLLOUT_DIR", "")
debug_train = os.environ.get("DEBUG_TRAIN", False)
debug_rollout = os.environ.get("DEBUG_ROLLOUT", False)
rollout_endpoint_mode = os.environ.get("ROLLOUT_ENDPOINT_MODE", "1")

enable_evaluate = eval_data_path is not None and eval_data_path != ""

# basic settings
experimental_name = "grpo_mix_data"
total_epochs = 15
global_batch_size = 64
prompt_repeat_k = 8
rollout_tp_size = 2
rollout_ep_size = 1
max_prompt_length = 2048
max_response_length = 8192
pack_max_length = 32768
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

rollout_endpoint_kind_map = {
    "1": "worker_local",
    "2": "worker_http",
    "3": "worker_extern",
}
if rollout_endpoint_mode not in rollout_endpoint_kind_map:
    raise ValueError("ROLLOUT_ENDPOINT_MODE must be one of: 1, 2, 3.")

# Mode 1 (worker_local): AgentLoop holds a WorkerLocalRouter object and calls worker Ray actors directly.
# Mode 2 (worker_http): trainer starts WorkerHttpRouter after rollout workers are ready, and AgentLoop gets its
# runtime URL from rollout_controller metadata.
# Mode 3 (worker_extern): AgentLoop uses WORKER_EXTERN_ROUTER_URL, while WorkerExternRouter periodically
# registers/removes rollout worker URLs to/from that external router.
rollout_endpoint_config = RolloutEndpointConfig(
    kind=rollout_endpoint_kind_map[rollout_endpoint_mode],
    base_url=os.environ.get("WORKER_EXTERN_ROUTER_URL") if rollout_endpoint_mode == "3" else None,
    worker_http_router_host=os.environ.get("WORKER_HTTP_ROUTER_HOST", "0.0.0.0"),
    worker_http_router_port=int(os.environ.get("WORKER_HTTP_ROUTER_PORT", "8081")),
    worker_extern_router_worker_api_key=os.environ.get("ROLLOUT_WORKER_API_KEY") or None,
)

# sampling params
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
                "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                    processor_path=model_path,
                    max_length=max_prompt_length,
                    system_message=data.get("system_message", None),
                    chat_template="qwen3.5-vl",
                    add_generation_prompt=True,
                    enable_thinking=True,
                ),
            }
        )

if enable_evaluate:
    eval_dataset_cfg = [
        {
            "dataset": DatasetConfig(
                name=f"{experimental_name}_eval",
                anno_path=eval_data_path,
                media_root=eval_media_root,
                sample_ratio=1.0,
                class_name="VLMJsonlDataset",
            ),
            "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                processor_path=model_path,
                max_length=max_prompt_length,
                chat_template="qwen3.5-vl",
                add_generation_prompt=True,
                enable_thinking=True,
                ignore_multimodal_info=True,
            ),
        }
    ]
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


# 4. judger
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
eos_token_id = get_eos_token(model_path)
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

if enable_evaluate:
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
    enable_evaluate=True
else:
    eval_agent_loop_manager_cfg = None
    enable_evaluate=False

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
    rollout_endpoint_config=rollout_endpoint_config,
)
