"""GLM-5.3-Flash VL SFT config (image/video), the multimodal counterpart of sft_glm53.py.

The only structural differences from the text-only config are the three that make it a VLM:
``Glm53BaseConfig.from_hf`` (vision tower + projector + language model, instead of the text half
alone), ``Glm53VLTokenizeFnConfig`` over a ``VLMJsonlDataset``, and ``glm53_vl_sft_collator``,
which needs ``image_token_id``/``merge_unit`` bound through ``collator_kwargs`` to re-check its
placeholder<->patch-count invariant after pack-level truncation.
"""

import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import Glm53VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig
from xtuner.v1.train import TrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig


def _get_bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _get_dispatcher():
    dispatcher = os.environ.get("DISPATCHER", "all2all").lower()
    if dispatcher in ("", "none", "null"):
        return None
    return dispatcher


def _get_float8_config() -> Float8Config | None:
    if not _get_bool_env("FP8", False):
        return None
    return Float8Config(
        scaling_granularity_gemm=ScalingGranularity.TILEWISE,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )


GLM5_3_MODEL_PATH = os.environ["GLM5_3_MODEL_PATH"]
DATA_PATH = os.environ["VL_DATA_PATH"]
MEDIA_ROOT = os.environ.get("VL_MEDIA_ROOT", DATA_PATH)

work_dir = os.environ.get("WORK_DIR", "work_dirs/glm53_vl_sft")
ep_size = int(os.environ.get("EP_SIZE", "1"))
sp_size = int(os.environ.get("SP_SIZE", "1"))
intra_layer_micro_batch = int(os.environ.get("INTRA_LAYER_MICRO_BATCH", "1"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", os.environ.get("WORLD_SIZE", "8")))
sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", "4096"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", "16384"))
total_step = int(os.environ.get("TOTAL_STEP", "10"))

loss_cfg = CELossConfig(
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")),
)

model_cfg = Glm53BaseConfig.from_hf(GLM5_3_MODEL_PATH)
model_cfg.text_config.dispatcher = _get_dispatcher()
model_cfg.text_config.ep_size = ep_size
model_cfg.text_config.lm_loss_cfg = loss_cfg
model_cfg.text_config.attention.sparse_mla_backend = (
    os.environ.get("SPARSE_MLA_BACKEND", "flash_mla_cudnn").strip().lower()
)
if "INDEXER_BACKEND" in os.environ:
    model_cfg.text_config.attention.indexer_backend = os.environ["INDEXER_BACKEND"].strip().lower()
# FP8 quantizes the language tower's large projections only; the vision half stays bf16, which
# is also how the published checkpoint stores it.
model_cfg.text_config.float8_cfg = _get_float8_config()
model_cfg.vision_config.attn_impl = os.environ.get("VISION_ATTN_IMPL", "flash_attention")
model_cfg.compile_cfg = _get_bool_env("MODEL_COMPILE", False)
model_cfg.text_config.compile_cfg = _get_bool_env("MODEL_COMPILE", False)

cache_dir = os.path.join(work_dir, "jsonl_cache")
dataset_config = [
    {
        "dataset": DatasetConfig(
            name="glm53_vl",
            anno_path=DATA_PATH,
            class_name="VLMJsonlDataset",
            media_root=MEDIA_ROOT,
            sample_ratio=float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0")),
            cache_dir=cache_dir,
            cache_tag=os.environ.get("CACHE_TAG", f"glm53_vl_{sample_max_length}"),
        ),
        "tokenize_fn": Glm53VLTokenizeFnConfig(
            processor_path=GLM5_3_MODEL_PATH,
            max_length=sample_max_length,
            max_pixels=int(os.environ["VL_MAX_PIXELS"]) if "VL_MAX_PIXELS" in os.environ else None,
            min_pixels=int(os.environ["VL_MIN_PIXELS"]) if "VL_MIN_PIXELS" in os.environ else None,
        ),
    },
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    collator="glm53_vl_sft_collator",
    collator_kwargs={
        "image_token_id": model_cfg.image_token_id,
        "merge_unit": model_cfg.vision_config.spatial_merge_size**2,
    },
    pack_level=os.environ.get("PACK_LEVEL", "soft"),
    pack_max_length=pack_max_length,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=_get_bool_env("GLOBAL_PACK", True),
    group_by_length=_get_bool_env("GROUP_BY_LENGTH", True),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

lr = float(os.environ.get("LR", "1e-6"))
optimizer = os.environ.get("OPTIMIZER", "adamw").lower()
if optimizer == "muon":
    optim_cfg = MuonConfig(lr=lr)
elif optimizer == "adamw":
    optim_cfg = AdamWConfig(
        lr=lr,
        foreach=_get_bool_env("ADAMW_FOREACH", False),
        swap_optimizer=_get_bool_env("SWAP_OPTIMIZER", False),
    )
else:
    raise ValueError(f"Unsupported OPTIMIZER={optimizer!r}. Use adamw or muon.")
lr_cfg = LRConfig(lr_type=os.environ.get("LR_TYPE", "cosine"), warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0")))
fsdp_cfg = FSDPConfig(
    cpu_offload=_get_bool_env("CPU_OFFLOAD", False),
    ep_size=ep_size,
    torch_compile=_get_bool_env("TORCH_COMPILE", False),
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=GLM5_3_MODEL_PATH,
    tokenizer_path=GLM5_3_MODEL_PATH,
    strict_load=_get_bool_env("STRICT_LOAD", False),
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    fsdp_cfg=fsdp_cfg,
    global_batch_size=global_batch_size,
    total_step=total_step,
    intra_layer_micro_batch=intra_layer_micro_batch,
    sp_size=sp_size,
    load_checkpoint_cfg=LoadCheckpointConfig(checkpoint_path=os.environ.get("LOAD_CHECKPOINT_PATH")),
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "200")),
    checkpoint_maxkeep=int(os.environ.get("CHECKPOINT_MAX_KEEP", "3")),
    hf_interval=int(os.environ.get("HF_INTERVAL", "200")),
    hf_max_keep=int(os.environ.get("HF_MAX_KEEP", "3")),
    work_dir=work_dir,
    profile_memory=_get_bool_env("PROFILE_MEMORY", False),
    profile_time=_get_bool_env("PROFILE_TIME", False),
    profile_step=[int(x) for x in os.environ.get("PROFILE_STEP", "2,3").split(",") if x],
    debug_skip_save=_get_bool_env("DEBUG_SKIP_SAVE", False),
)
