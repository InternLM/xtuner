"""Matched Qwen3.5 configuration for the MoonEP/DeepEP 20-step gate.

The dispatcher, MTP switch, fixed pack length and output directory are the
only run-varying inputs.  All other workload choices are deliberately shared.
"""

import os

import torch

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.moe.qwen3_5_text import MOE_EP_COMPILE_CFG
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.train import TrainerConfig


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    return value


backend = _required_env("MOONEP_ACCEPTANCE_BACKEND")
if backend not in {"deepep", "moonep"}:
    raise ValueError(f"MOONEP_ACCEPTANCE_BACKEND must be deepep or moonep, got {backend!r}")

mtp_enabled = bool(int(_required_env("MOONEP_ACCEPTANCE_MTP")))
pack_length = int(_required_env("MOONEP_ACCEPTANCE_PACK_LENGTH"))
if pack_length <= 0:
    raise ValueError("MOONEP_ACCEPTANCE_PACK_LENGTH must be positive")
if os.environ.get("XTUNER_ACTIVATION_OFFLOAD", "0") != "0":
    raise ValueError("formal MoonEP acceptance runs require XTUNER_ACTIVATION_OFFLOAD=0")
if os.environ.get("XTUNER_USE_CUTLASS_GROUP_GEMM", "0") == "1":
    raise ValueError("formal MoonEP acceptance runs require the Triton grouped-GEMM backend")

model_cfg = Qwen3_5_VLMoE35BA3Config(only_llm_forward=True)
text_cfg = model_cfg.text_config
text_cfg.ep_size = 4
text_cfg.dispatcher = backend
text_cfg.moonep_staging_reference = False
text_cfg.router_compute_dtype = "float32"
text_cfg.router_async_offload = False
# The installed FlashAttention package metadata has no importable extension in
# pt212_cu132. Flex attention keeps both dispatcher runs on the same real-model
# workload instead of depending on that broken optional binary.
text_cfg.attention.attn_impl = "flex_attention"
# FlexAttention intentionally compiles behind a graph break so its BlockMask
# tensors become fixed-layout inputs to the kernel graph. Keep all default
# Qwen3.5 compile targets, but let MHA form that one required boundary.
text_cfg.compile_cfg = MOE_EP_COMPILE_CFG | {
    "xtuner.v1.module.attention.mha.MultiHeadAttention.forward": {"fullgraph": False}
}
text_cfg.mtp_config = MTPConfig(num_layers=1) if mtp_enabled else None

dataset_cfg = [
    {
        "dataset": DatasetConfig(
            name="alpaca",
            anno_path=_required_env("MOONEP_ACCEPTANCE_DATA_PATH"),
            sample_ratio=1.0,
        ),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=262144),
    }
]
dataloader_cfg = DataloaderConfig(
    dataset_config_list=dataset_cfg,
    pack_to_max_length=True,
    pack_max_length=pack_length,
    pack_level="hard",
)

profile_step_env = os.environ.get("MOONEP_ACCEPTANCE_PROFILE_STEP")
profile_step = int(profile_step_env) if profile_step_env else None

trainer = TrainerConfig(
    load_from=_required_env("MOONEP_ACCEPTANCE_MODEL_PATH"),
    tokenizer_path=_required_env("MOONEP_ACCEPTANCE_MODEL_PATH"),
    model_cfg=model_cfg,
    optim_cfg=AdamWConfig(lr=6e-5, foreach=False),
    lr_cfg=LRConfig(lr_type="cosine", lr_min=1e-6),
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    fsdp_cfg=FSDPConfig(
        ep_size=4,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        torch_compile=True,
        cpu_offload=False,
    ),
    dataloader_cfg=dataloader_cfg,
    global_batch_size=8,
    intra_layer_micro_batch=1,
    sp_size=1,
    total_step=20,
    work_dir=_required_env("MOONEP_ACCEPTANCE_WORK_DIR"),
    seed=0,
    strict_load=False,
    auto_resume=False,
    debug_skip_save=True,
    exp_tracker="jsonl",
    profile_step=profile_step,
    profile_time=profile_step is not None,
    profile_memory=False,
)
