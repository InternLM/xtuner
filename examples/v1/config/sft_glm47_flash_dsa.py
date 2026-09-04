"""Two-stage GLM-4.7 Flash main/MTP indexer training."""

import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import LongTextPretrainTokenizeFunctionConfig, OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Glm47FlashDSAConfig
from xtuner.v1.module.attention import DSAIndexerTrainingConfig
from xtuner.v1.train import TrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig


def _bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


stage = os.environ.get("STAGE", "dense_warmup").lower()
if stage not in ("dense_warmup", "sparse"):
    raise ValueError(f"Unsupported STAGE={stage!r}.")

model_path = os.environ["GLM47_FLASH_MODEL_PATH"]
work_dir = os.environ.get("WORK_DIR", f"work_dirs/glm47_flash_dsa_{stage}")
world_size = int(os.environ.get("WORLD_SIZE", "8"))
ep_size = int(os.environ.get("EP_SIZE", str(world_size)))
sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", "4096"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", str(sample_max_length)))
total_step = int(os.environ.get("TOTAL_STEP", "350" if stage == "dense_warmup" else "3000"))

model_cfg = Glm47FlashDSAConfig.from_hf(model_path)
model_cfg.ep_size = ep_size
model_cfg.dispatcher = os.environ.get("DISPATCHER", "all2all")
model_cfg.compile_cfg = False
model_cfg.float8_cfg = None
model_cfg.attention.index_topk = int(os.environ.get("INDEX_TOPK", "512"))
model_cfg.attention.sparse_mla_backend = (
    "torch" if stage == "dense_warmup" else os.environ.get("SPARSE_MLA_BACKEND", "cudnn_dsa")
)
model_cfg.attention.indexer_training = DSAIndexerTrainingConfig(
    stage=stage,
    loss_coeff=float(os.environ.get("INDEXER_LOSS_COEFF", "1.0")),
    train_mtp_indexer=True,
    indexer_only=True,
    dense_query_block_size=int(os.environ.get("DENSE_QUERY_BLOCK_SIZE", "128")),
    debug_interval=int(os.environ.get("INDEXER_DEBUG_INTERVAL", "0")),
)
model_cfg.freeze_routers = True

if stage == "sparse" and model_cfg.attention.sparse_mla_backend != "cudnn_dsa":
    raise ValueError("Sparse indexer training requires SPARSE_MLA_BACKEND=cudnn_dsa.")
if stage == "sparse" and model_cfg.attention.index_topk % 128 != 0:
    raise ValueError("Sparse cuDNN DSA requires INDEX_TOPK divisible by 128.")
if int(os.environ.get("SP_SIZE", "1")) != 1:
    raise ValueError("Indexer training requires SP_SIZE=1.")
if float(os.environ.get("RECOMPUTE_RATIO", "0")) != 0:
    raise ValueError("Indexer training requires RECOMPUTE_RATIO=0.")

loss_cfg = CELossConfig(mode="chunk", chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")))
model_cfg.lm_loss_cfg = loss_cfg

pretrain_tokenize_cfg = LongTextPretrainTokenizeFunctionConfig(
    chunk_size=sample_max_length,
    tokenizer_chunk_chars=int(os.environ.get("TOKENIZER_CHUNK_CHARS", "32768")),
    overlap_chars=int(os.environ.get("TOKENIZER_OVERLAP_CHARS", "512")),
    min_chunk_tokens=0,
    max_length=sample_max_length,
    add_bos_token=False,
    add_eos_token=True,
)
cache_tag = f"glm47_flash_{sample_max_length}"
if stage == "dense_warmup":
    dataset_config = [
        {
            "dataset": DatasetConfig(
                name="warmup_pretrain_4k",
                anno_path=os.environ["WARMUP_DATASET_PATH"],
                sample_ratio=float(os.environ.get("WARMUP_DATASET_SAMPLE_RATIO", "1.0")),
                cache_dir=os.path.join(work_dir, "jsonl_cache"),
                cache_tag=f"{cache_tag}_warmup",
            ),
            "tokenize_fn": pretrain_tokenize_cfg,
        }
    ]
else:
    dataset_config = [
        {
            "dataset": DatasetConfig(
                name="sft_4k",
                anno_path=os.environ["SFT_DATASET_PATH"],
                sample_ratio=float(os.environ.get("SFT_DATASET_SAMPLE_RATIO", "1.0")),
                cache_dir=os.path.join(work_dir, "jsonl_cache"),
                cache_tag=f"{cache_tag}_sft",
            ),
            "tokenize_fn": OpenaiTokenizeFunctionConfig(
                chat_template=os.environ.get("CHAT_TEMPLATE", "glm5.2"),
                max_length=sample_max_length,
            ),
        },
        {
            "dataset": DatasetConfig(
                name="pretrain_4k",
                anno_path=os.environ["PRETRAIN_DATASET_PATH"],
                sample_ratio=float(os.environ.get("PRETRAIN_DATASET_SAMPLE_RATIO", "1.0")),
                cache_dir=os.path.join(work_dir, "jsonl_cache"),
                cache_tag=f"{cache_tag}_pretrain",
            ),
            "tokenize_fn": pretrain_tokenize_cfg,
        },
    ]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level="soft",
    pack_max_length=pack_max_length,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=True,
    group_by_length=True,
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

optim_cfg = AdamWConfig(
    lr=float(os.environ.get("LR", "1e-3" if stage == "dense_warmup" else "7.3e-6")),
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.0" if stage == "dense_warmup" else "0.01")),
    foreach=False,
)
lr_cfg = LRConfig(
    lr_type="constant" if stage == "dense_warmup" else "cosine",
    warmup_ratio=0,
)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    ep_size=ep_size,
    torch_compile=False,
    recompute_ratio=0,
)

indexer_checkpoint_path = os.environ.get("INDEXER_CHECKPOINT_PATH")
# A full warm-up HF export already contains base, MTP, and indexer weights, so
# it must go through the normal ``load_from`` path rather than the DCP loader.
if indexer_checkpoint_path and os.path.isfile(os.path.join(indexer_checkpoint_path, "model.safetensors.index.json")):
    indexer_checkpoint_path = None
load_checkpoint_cfg = LoadCheckpointConfig(
    checkpoint_path=indexer_checkpoint_path,
    load_optimizer_states=False,
    load_optimizer_args=False,
    load_dataset=False,
    load_scheduler=False,
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    tokenizer_path=model_path,
    strict_load=True,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    fsdp_cfg=fsdp_cfg,
    global_batch_size=int(os.environ.get("GLOBAL_BATCH_SIZE", str(world_size))),
    total_step=total_step,
    intra_layer_micro_batch=1,
    sp_size=1,
    auto_resume=_bool_env("AUTO_RESUME", True),
    load_checkpoint_cfg=load_checkpoint_cfg,
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", str(total_step))),
    checkpoint_maxkeep=int(os.environ.get("CHECKPOINT_MAX_KEEP", "2")),
    hf_interval=int(os.environ.get("HF_INTERVAL", "0")) or None,
    work_dir=work_dir,
    profile_memory=_bool_env("PROFILE_MEMORY", False),
    profile_time=_bool_env("PROFILE_TIME", False),
    profile_step=[2, 3],
    debug_skip_save=False,
    do_clip=True,
)
