import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.train import TrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig


def _get_bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _get_dispatcher():
    dispatcher = os.environ.get("DISPATCHER", "all2all").lower()
    if dispatcher in ("", "none", "null"):
        return None
    return dispatcher


def _dataset_entry(name: str, anno_path: str | Path, sample_ratio: float, cache_dir: str, cache_tag: str):
    return {
        "dataset": DatasetConfig(
            name=name,
            anno_path=anno_path,
            sample_ratio=sample_ratio,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
        ),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(
            chat_template="glm5.2",
            max_length=sample_max_length,
        ),
    }


GLM5_2_MODEL_PATH = os.environ["GLM5_2_MODEL_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]
ALPACA_LONG_PATH = os.environ["ALPACA_LONG_PATH"]

work_dir = os.environ.get("WORK_DIR", "work_dirs/glm52_sft")
# On single-node 8-GPU SFT, EP=8 leaves FSDP size at 1 and replicates non-expert params.
ep_size = int(os.environ.get("EP_SIZE", "1"))
intra_layer_micro_batch = int(os.environ.get("INTRA_LAYER_MICRO_BATCH", "1"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", os.environ.get("WORLD_SIZE", "8")))
sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", "4096"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", "16384"))
total_step = int(os.environ.get("TOTAL_STEP", "10"))

loss_cfg = CELossConfig(
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")),
)

model_cfg = get_model_config_from_hf(GLM5_2_MODEL_PATH)
model_cfg.dispatcher = _get_dispatcher()
model_cfg.ep_size = ep_size
model_cfg.compile_cfg = _get_bool_env("MODEL_COMPILE", False)
model_cfg.lm_loss_cfg = loss_cfg
if hasattr(model_cfg.attention, "sparse_mla_backend"):
    model_cfg.attention.sparse_mla_backend = os.environ.get("SPARSE_MLA_BACKEND", "tilelang")

cache_dir = os.path.join(work_dir, "jsonl_cache")
cache_tag = os.environ.get("CACHE_TAG", f"glm52_{sample_max_length}")
dataset_type = os.environ.get("DATASET_TYPE", "alpaca").lower()
dataset_paths = {
    "alpaca": ALPACA_PATH,
    "alpaca_long": ALPACA_LONG_PATH,
}
if dataset_type not in dataset_paths:
    raise ValueError(f"Unsupported DATASET_TYPE={dataset_type!r}. Use alpaca or alpaca_long.")
dataset_config = [
    _dataset_entry(
        dataset_type,
        dataset_paths[dataset_type],
        float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0")),
        cache_dir,
        f"{cache_tag}_{dataset_type}",
    )
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level=os.environ.get("PACK_LEVEL", "soft"),
    pack_max_length=pack_max_length,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=_get_bool_env("GLOBAL_PACK", True),
    group_by_length=_get_bool_env("GROUP_BY_LENGTH", True),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

optim_cfg = AdamWConfig(
    lr=float(os.environ.get("LR", "1e-6")),
    foreach=_get_bool_env("ADAMW_FOREACH", False),
    swap_optimizer=_get_bool_env("SWAP_OPTIMIZER", False),
)
lr_cfg = LRConfig(lr_type=os.environ.get("LR_TYPE", "cosine"), warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0")))
fsdp_cfg = FSDPConfig(
    cpu_offload=_get_bool_env("CPU_OFFLOAD", False),
    ep_size=ep_size,
    torch_compile=_get_bool_env("TORCH_COMPILE", False),
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=GLM5_2_MODEL_PATH,
    tokenizer_path=GLM5_2_MODEL_PATH,
    strict_load=_get_bool_env("STRICT_LOAD", True),
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    fsdp_cfg=fsdp_cfg,
    global_batch_size=global_batch_size,
    total_step=total_step,
    intra_layer_micro_batch=intra_layer_micro_batch,
    sp_size=int(os.environ.get("SP_SIZE", "1")),
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
