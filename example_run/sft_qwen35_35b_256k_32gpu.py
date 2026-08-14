import json
import os
import shutil
from pathlib import Path
from typing import Any

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import (
    PretrainTokenizeFunctionConfig,
    Qwen3VLTokenizeFnConfig,
)
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import (
    QWEN3VL_COMPILE_CFG,
)
# from xtuner.v1.model.moe.moe import MTPConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig


# This vision-layer compile rule is incompatible with Qwen3.5-35B-A3B.
QWEN3VL_COMPILE_CFG.pop(
    "xtuner.v1.model.compose.qwen3_vl.modeling_vision."
    "Qwen3VLVisionLayer.forward",
    None,
)


def _get_int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


# Paths are provided by one of the submit scripts in this directory.
ceph_config = os.getenv("CEPH_CONFIG", "")
meta_data_path = Path(os.environ["META_DATA_PATH"])
model_path = Path(os.environ["MODEL_PATH"])
work_dir = Path(os.environ["WORK_DIR"])
tokenizer_cache_dir = os.environ["TOKENIZER_CACHE_DIR"]
chat_template_name = os.getenv("CHAT_TEMPLATE_NAME", "qwen3.5-vl")

work_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(__file__, work_dir)

# 256k-context SFT defaults for 32 GPUs (4 nodes x 8 GPUs).
sample_max_length = _get_int_env("SAMPLE_MAX_LENGTH", 256 * 1024)
pack_max_length = _get_int_env("PACK_MAX_LENGTH", 256 * 1024)
rand_video_max_frames = _get_int_env("RAND_VIDEO_MAX_FRAMES", 24)
num_workers = _get_int_env("NUM_WORKERS", 4)
global_batch_size = _get_int_env("GLOBAL_BATCH_SIZE", 8)
total_epoch = _get_int_env("TOTAL_EPOCH", 1)
hf_interval = _get_int_env("HF_INTERVAL", 500)
hf_max_keep = _get_int_env("HF_MAX_KEEP", 2)
checkpoint_interval = _get_int_env("CHECKPOINT_INTERVAL", 500)
checkpoint_maxkeep = _get_int_env("CHECKPOINT_MAXKEEP", 2)

lr = _get_float_env("LR", 2e-5)
lr_min = _get_float_env("LR_MIN", 1e-6)
weight_decay = _get_float_env("WEIGHT_DECAY", 0.05)
warmup_ratio = _get_float_env("WARMUP_RATIO", 0.1)
recompute_ratio = _get_float_env("RECOMPUTE_RATIO", 1.0)
loss_reduction = os.getenv("LOSS_REDUCTION", "square")
max_pixels = _get_int_env("MAX_PIXELS", 16_777_216)

sp_size = _get_int_env("SP_SIZE", 4)
ep_size = _get_int_env("EP_SIZE", 1)
tp_size = _get_int_env("TP_SIZE", 1)
torch_compile = _get_bool_env("TORCH_COMPILE", True)

# Qwen3.5-35B-A3B model settings.
model_cfg = Qwen3_5_VLMoE35BA3Config()

with (model_path / "config.json").open("r", encoding="utf-8") as file:
    model_hf_config: dict[str, Any] = json.load(file)

model_cfg.text_config.vocab_size = model_hf_config["text_config"]["vocab_size"]
# model_cfg.text_config.mtp_config = [
#     MTPConfig(
#         name="normal",
#         mask_type=None,
#         num_layers=4,
#         share_weights=True,
#         loss_scaling_factor=1.0,
#     ),
# ]

if ep_size > 1:
    model_cfg.text_config.ep_size = ep_size
    model_cfg.text_config.dispatcher = "deepep"

# Dataset recipe: META_DATA_PATH points to a metadata JSON file.
oss_loader_cfg = (
    OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
    if ceph_config
    else None
)

ds_collections: dict[str, Any] = json.loads(
    meta_data_path.read_text(encoding="utf-8")
)
has_pretrain = any(
    data.get("text_pretrain", False) for data in ds_collections.values()
)
dataset_config: list[dict[str, Any]] = []

for name, data in ds_collections.items():
    is_pretrain = data.get("text_pretrain", False)
    if is_pretrain:
        tokenize_fn = PretrainTokenizeFunctionConfig(hash=data.get("hash"))
    else:
        tokenize_fn = Qwen3VLTokenizeFnConfig(
            chat_template=chat_template_name,
            llm_pack_weight=-3.2,
            visual_pack_weight=5.0,
            max_length=sample_max_length,
            processor_path=str(model_path),
            rand_video_max_frames=rand_video_max_frames,
            oss_loader_cfg=oss_loader_cfg,
            max_pixels=max_pixels,
            debug=True,
        )

    dataset_config.append(
        {
            "dataset": DatasetConfig(
                name=name,
                anno_path=data["annotation"],
                media_root=data.get("media_root") or "",
                sample_ratio=data.get("sample_ratio", 1.0),
                class_name="JsonlDataset" if is_pretrain else "VLMJsonlDataset",
                enable_sequential_sampler=True,
                cache_tag="xtuner_train_v2",
                cache_dir=tokenizer_cache_dir,
            ),
            "tokenize_fn": tokenize_fn,
        }
    )

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level="mllm_hybrid" if has_pretrain else "soft",
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=_get_int_env("PACK_EXTRA_BUFFER_SIZE", 20),
)

optim_cfg = MuonConfig(lr=lr, weight_decay=weight_decay)
lr_cfg = LRConfig(
    lr_type="cosine",
    warmup_ratio=warmup_ratio,
    lr_min=lr_min,
)
fsdp_cfg = FSDPConfig(
    tp_size=tp_size,
    ep_size=ep_size,
    recompute_ratio=recompute_ratio,
    torch_compile=torch_compile,
    checkpoint_preserve_rng_state=False,
)

trainer = TrainerConfig(
    sp_size=sp_size,
    load_from=str(model_path),
    resume_cfg=ResumeConfig(auto_resume=True),
    tokenizer_path=str(model_path),
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(
        mode="chunk",
        chunk_size=1024,
        loss_reduction=loss_reduction,
    ),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
)
