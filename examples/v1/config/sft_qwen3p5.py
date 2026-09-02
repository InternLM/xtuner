import os

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig, Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.moe.moe import MTPConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig


def _get_bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


WORK_DIR = os.environ.get("WORK_DIR", "work_dirs/qwen3p5_sft")
QWEN3_5_MOE_PATH = os.environ["QWEN3_5_MOE_PATH"]
ALPACA_PATH = os.environ.get("ALPACA_PATH")
DATA_PATH = os.environ.get("DATA_PATH", ALPACA_PATH)
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "")
if DATA_PATH is None:
    raise RuntimeError("DATA_PATH or ALPACA_PATH is required")

sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", str(128 * 1024)))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", str(128 * 1024)))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))
ep_size = int(os.environ.get("EP_SIZE", "1"))
model_compile = _get_bool_env("MODEL_COMPILE", True)
total_step = int(os.environ["TOTAL_STEP"]) if "TOTAL_STEP" in os.environ else None
is_multimodal = bool(MEDIA_ROOT)

model_cfg = Qwen3_5_VLMoE35BA3Config(only_llm_forward=not is_multimodal, compile_cfg=model_compile)
model_cfg.text_config.ep_size = ep_size
if ep_size > 1:
    model_cfg.text_config.dispatcher = os.environ.get("DISPATCHER", "deepep")

if _get_bool_env("FP8"):
    model_cfg.text_config.float8_cfg = Float8Config(
        scaling_granularity_gemm=ScalingGranularity.TILEWISE,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )

# Shared physical weights retain the reference config's four normal MTP prediction depths.
normal_mtp_layers = int(os.environ.get("NORMAL_MTP_LAYERS", "4"))
model_cfg.text_config.mtp_config = MTPConfig(
    num_layers=normal_mtp_layers,
    share_weights=normal_mtp_layers > 1,
    loss_scaling_factor=float(os.environ.get("NORMAL_MTP_FACTOR", "1.0")),
)

# Qwen3VLTokenizeFnConfig is for multimodal data. Pure-text data such as
# Alpaca can use OpenaiTokenizeFunctionConfig.
if is_multimodal:
    tokenize_fn = Qwen3VLTokenizeFnConfig(
        chat_template="qwen3.5-vl",
        llm_pack_weight=-3.2,
        visual_pack_weight=5.0,
        max_length=sample_max_length,
        processor_path=QWEN3_5_MOE_PATH,
        rand_video_max_frames=int(os.environ.get("RAND_VIDEO_MAX_FRAMES", "24")),
        max_pixels=int(os.environ.get("MAX_PIXELS", str(16384 * 32 * 32))),
    )
else:
    tokenize_fn = OpenaiTokenizeFunctionConfig(
        chat_template="qwen3.5-vl",
        max_length=sample_max_length,
    )

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="multimodal" if is_multimodal else "alpaca",
            anno_path=DATA_PATH,
            class_name="VLMJsonlDataset" if is_multimodal else "JsonlDataset",
            media_root=MEDIA_ROOT,
            sample_ratio=float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0")),
            cache_dir=os.path.join(WORK_DIR, "jsonl_cache"),
            cache_tag=os.environ.get(
                "CACHE_TAG",
                f"qwen3p5_{'vl' if is_multimodal else 'text'}_{sample_max_length}",
            ),
        ),
        "tokenize_fn": tokenize_fn,
    }
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level="soft",
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=_get_bool_env("GLOBAL_PACK", True),
    group_by_length=_get_bool_env("GROUP_BY_LENGTH", True),
    collator="qwen3_vl_sft_collator" if is_multimodal else "sft_llm_collator",
    pack_extra_buffer_size=int(os.environ.get("PACK_EXTRA_BUFFER_SIZE", "20")),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

optim_cfg = MuonConfig(
    lr=float(os.environ.get("LR", "2e-5")),
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.05")),
    use_gram_newton_schulz=_get_bool_env("USE_GRAM_NEWTON_SCHULZ", False),
)
lr_cfg = LRConfig(
    lr_type="cosine",
    warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.1")),
    lr_min=float(os.environ.get("LR_MIN", "1e-6")),
)
fsdp_cfg = FSDPConfig(
    recompute_ratio=float(os.environ.get("RECOMPUTE_RATIO", "1.0")),
    torch_compile=model_compile,
    ep_size=ep_size,
    checkpoint_preserve_rng_state=False,
)
loss_cfg = CELossConfig(
    mode="chunk",
    chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")),
    loss_reduction=os.environ.get("LOSS_REDUCTION", "square"),
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=QWEN3_5_MOE_PATH,
    tokenizer_path=QWEN3_5_MOE_PATH,
    resume_cfg=ResumeConfig(auto_resume=True),
    fsdp_cfg=fsdp_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    global_batch_size=global_batch_size,
    total_step=total_step,
    total_epoch=None if total_step is not None else int(os.environ.get("TOTAL_EPOCH", "1")),
    sp_size=int(os.environ.get("SP_SIZE", "4")),
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "500")),
    checkpoint_maxkeep=int(os.environ.get("CHECKPOINT_MAX_KEEP", "2")),
    hf_interval=int(os.environ.get("HF_INTERVAL", "500")),
    hf_max_keep=int(os.environ.get("HF_MAX_KEEP", "2")),
    work_dir=WORK_DIR,
    debug_skip_save=_get_bool_env("DEBUG_SKIP_SAVE"),
)
