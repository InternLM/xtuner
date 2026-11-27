"""Qwen3.6-35B-A3B（VL MoE）+ GDP agentic 多轮工具 smoke 配置（E2 行为闭环）。

模板：glm53-rubric-process-reward/configs/qwen35_35b_moe_rubric_ppo.py（PRM 线验证过，
policy 同为 35B-A3B VL MoE，与文本 agentic RL（CyberGym）同族参考配置）。
适配：
- SingleTurnAgentLoop → GdpToolAgentLoopConfig（recipe/gdp_pdf_agentic/agent_loop.py，model_family=qwen35_vl）：
  多轮工具调用（get_doc_meta/get_page_text/get_page_image），wrapper 与 Qwen3.5
  chat_template 实测渲染对齐（<|im_end|>\n<|im_start|>user\n<tool_response>...）。
- 数据：gdp_agentic_qwen_smoke.jsonl（qwen JSON <tool_call> 指令变体）。
- judger：RubricJudger 单分支（process），judge 用 Qwen3.5-9B（小 VL，能看页图透传）。
目的：用有能力的 VL policy 验证行为闭环——自主工具调用（tool_calls/mean>0）、
多轮、判分有区分度；这是 GLM 减层 policy（输出乱码）无法判别的部分。
"""

import json
import os
from pathlib import Path

from recipe.gdp_pdf_agentic.agent_loop import GdpToolAgentLoopConfig
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.judger import ComposedJudgerConfig, RubricJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


DATA_ROOT = Path(
    os.environ.get(
        "GDP_DATA_DIR",
        "/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/glm53-multimodal-agentic-rl/data/gdp_pdf",
    )
)

model_path = os.environ.get("MODEL_PATH", "/mnt/shared-storage-user/llmrazor-share/model/Qwen3.6-35B-A3B")
data_path = os.environ.get("DATA_PATH", str(DATA_ROOT / "gdp_agentic_qwen_smoke_config.json"))
work_dir = os.environ.get("WORK_DIR", str(Path.cwd() / "runs/qwen35_vl_gdp_agentic_smoke"))
judge_endpoint = os.environ.get("JUDGE_ENDPOINT", "http://127.0.0.1:18000/v1/chat/completions")
judge_model = os.environ.get("JUDGE_MODEL", "Qwen3.5-9B")
data_judger_mapping = {"gdp_pdf": {"process": 1.0}}

num_workers = int(os.environ.get("NUM_WORKERS", "8"))
ep_size = int(os.environ.get("EP_SIZE", "4"))
sp_size = int(os.environ.get("SP_SIZE", "1"))
rollout_ep_size = int(os.environ.get("ROLLOUT_EP_SIZE", "4"))
agent_max_turns = int(os.environ.get("AGENT_MAX_TURNS", "6"))
max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "8192"))
max_response_length = int(os.environ.get("MAX_RESPONSE_LENGTH", "4096"))
pack_max_length = int(
    os.environ.get("PACK_MAX_LENGTH", str(max_prompt_length + max_response_length * agent_max_turns))
)
_max_total_env = os.environ.get("MAX_TOTAL_TOKENS")
max_total_tokens = (int(_max_total_env) or None) if _max_total_env is not None else pack_max_length
if num_workers % ep_size or num_workers % sp_size or num_workers % rollout_ep_size:
    raise ValueError("Worker count must be divisible by train EP/SP and rollout EP.")

model_cfg = Qwen3_5_VLMoE35BA3Config()
model_cfg.only_llm_forward = False
model_cfg.freeze_vision = True
model_cfg.freeze_projector = True
model_cfg.compile_cfg = False
text_config = model_cfg.text_config
text_config.compile_cfg = False
text_config.ep_size = ep_size

fsdp_cfg = FSDPConfig(
    ep_size=ep_size,
    fp32_lm_head=True,
    torch_compile=False,
    cpu_offload=False,
)
rollout_config = RolloutConfig(
    env="qwen35_vl_gdp_agentic_smoke",
    worker_log_dir=Path(work_dir) / "rollout_workers",
    device="GPU",
    model_path=model_path,
    model_name="qwen36-35b-moe-policy",
    dtype="bfloat16",
    dist_port_base=int(os.environ.get("ROLLOUT_PORT_BASE", "27000")),
    tensor_parallel_size=1,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.6")),
    context_length=pack_max_length,
    rollout_max_batch_size_per_instance=16,
    max_prefill_token_num=int(os.environ.get("MAX_PREFILL_TOKEN_NUM", "4096")),
    enable_return_routed_experts=False,
    skip_load_weights=False,
    enable_prefix_caching=True,
    enable_float8=False,
    fp32_lm_head=True,
    extra_rollout_config={
        "lmdeploy_backend": "pytorch",
        "distributed_executor_backend": "ray",
        "lmdeploy_log_level": "INFO",
    },
)

with open(data_path, encoding="utf-8") as handle:
    collections = json.load(handle)
dataset_entries = []
for name, data in collections.items():
    for annotation in data["annotation"] if isinstance(data["annotation"], list) else [data["annotation"]]:
        dataset_entries.append(
            {
                "dataset": DatasetConfig(
                    name=name,
                    class_name="VLMJsonlDataset",
                    anno_path=annotation,
                    media_root=data.get("media_root", ""),
                    sample_ratio=data.get("sample_ratio", 1.0),
                    cache_dir=Path(work_dir) / "dataset_cache",
                ),
                "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                    processor_path=model_path,
                    max_length=max_prompt_length,
                    chat_template="qwen3.5-vl",
                    add_generation_prompt=True,
                    enable_thinking=True,
                    ignore_multimodal_info=False,
                    data_judger_mapping=data_judger_mapping,
                ),
            }
        )
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_entries,
    collator="fake_collator",
    pack_level="none",
    pack_max_length=pack_max_length,
)
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    return_logprob=True,
    return_token_ids=True,
    return_routed_experts=True,
)
_default_judge_chat_kwargs = {"enable_thinking": False} if "qwen" in judge_model.lower() else None
_judge_chat_kwargs_env = os.environ.get("JUDGE_CHAT_TEMPLATE_KWARGS")
judge_chat_template_kwargs = (
    json.loads(_judge_chat_kwargs_env) if _judge_chat_kwargs_env else _default_judge_chat_kwargs
)
rubric_judger_kwargs: dict = {
    "judger_name": "process",
    "endpoint": judge_endpoint,
    "model": judge_model,
    "request_timeout": float(os.environ.get("JUDGE_TIMEOUT", "300")),
    "max_retries": int(os.environ.get("JUDGE_MAX_RETRIES", "2")),
    "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "4096")),
    "chat_template_kwargs": judge_chat_template_kwargs,
}
_rubric_fields = RubricJudgerConfig.model_fields
if "max_evidence_chars" in _rubric_fields and os.environ.get("JUDGE_MAX_EVIDENCE_CHARS"):
    rubric_judger_kwargs["max_evidence_chars"] = int(os.environ["JUDGE_MAX_EVIDENCE_CHARS"])
if "max_rubrics_per_request" in _rubric_fields and os.environ.get("JUDGE_MAX_RUBRICS_PER_REQ"):
    _max_rubrics = int(os.environ["JUDGE_MAX_RUBRICS_PER_REQ"])
    rubric_judger_kwargs["max_rubrics_per_request"] = _max_rubrics if _max_rubrics > 0 else None
if "on_failure" in _rubric_fields and os.environ.get("JUDGE_ON_FAILURE"):
    rubric_judger_kwargs["on_failure"] = os.environ["JUDGE_ON_FAILURE"]
if "evidence_match_mode" in _rubric_fields and os.environ.get("JUDGE_EVIDENCE_MATCH_MODE"):
    rubric_judger_kwargs["evidence_match_mode"] = os.environ["JUDGE_EVIDENCE_MATCH_MODE"]
if "evidence_min_overlap" in _rubric_fields and os.environ.get("JUDGE_EVIDENCE_MIN_OVERLAP"):
    rubric_judger_kwargs["evidence_min_overlap"] = float(os.environ["JUDGE_EVIDENCE_MIN_OVERLAP"])

judger_config = ComposedJudgerConfig(
    branches={
        "process": RubricJudgerConfig(**rubric_judger_kwargs),
    },
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="qwen35_vl_gdp_agentic_smoke",
        agent_loop_config=GdpToolAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=training_sample_params,
            enable_batch_judge=True,
            max_turns=agent_max_turns,
            max_tool_response_chars=int(os.environ.get("TOOL_MAX_CHARS", "4000")),
            model_family="qwen35_vl",
            max_total_tokens=max_total_tokens,
            final_turn_hint=os.environ.get("FINAL_TURN_HINT", "1").lower() in {"1", "true", "yes", "on"},
            strict_think_tool_parse=os.environ.get("STRICT_THINK_TOOL_PARSE", "1").lower()
            in {"1", "true", "yes", "on"},
            force_tool_calls=[spec for spec in os.environ.get("FORCE_TOOL_CALLS", "").split(";") if spec.strip()],
        ),
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=SamplerConfig(
            dataloader_cfg=dataloader_config,
            prompt_repeat_k=int(os.environ.get("PROMPT_REPEAT_K", "2")),
        ),
    )
)

learning_rate = float(os.environ.get("ACTOR_LR", "1e-6"))
loss_cfg = GRPOLossConfig(
    policy_loss_cfg={
        "loss_type": "vanilla",
        "cliprange_high": 0.2,
        "cliprange_low": 0.2,
        "clip_ratio_c": 3.0,
        "log_prob_diff_min": -20.0,
        "log_prob_diff_max": 20.0,
    },
    mode="chunk",
    chunk_size=512,
    use_kl_loss=False,
)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    sp_size=sp_size,
    optim_cfg=AdamWConfig(lr=learning_rate, weight_decay=0.1, foreach=False, swap_optimizer=False),
    loss_cfg=loss_cfg,
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=learning_rate),
    fsdp_cfg=fsdp_cfg,
    pack_max_length=pack_max_length,
    pack_strategy=os.environ.get("PACK_STRATEGY", "legacy"),
)
trainer = RLColocateTrainerConfig(
    resources=AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=num_workers,
        num_cpus_per_worker=10,
        cpu_memory_per_worker=24 * 1024**3,
    ),
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    load_from=model_path,
    total_train_steps=int(os.environ.get("TOTAL_TRAIN_STEPS", "2")),
    train_batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", "4")),
    advantage_estimator_config=GRPOAdvantageConfig(),
    sync_weights_interval=1,
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "-1")),
    checkpoint_maxkeep=int(os.environ.get("CHECKPOINT_MAXKEEP", "-1")),
    enable_evaluate=False,
    enable_initial_evaluate=False,
    work_dir=work_dir,
    seed=123,
    exp_tracker="tensorboard",
)
