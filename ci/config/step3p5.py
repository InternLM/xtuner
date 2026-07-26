import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.step3p5 import Step3p5FlashConfig
from xtuner.v1.train import TrainerConfig


# Point STEP3P5_PATH at the split / per-expert checkpoint produced by
# `.dev_scripts/convert_step3p5_to_split.py` (the released fused-expert layout cannot be sharded).
STEP3P5_PATH = os.environ["STEP3P5_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


# Step-3.5-Flash is a ~200B MoE; real training needs expert parallelism (and a multi-node cluster).
moe_cfg = Step3p5FlashConfig(ep_size=8, dispatcher="all2all", num_hidden_layers=4)
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    # torch.compile for the hybrid per-layer-RoPE decoder layers is a §8 optimization; keep eager here.
    torch_compile=False,
    cpu_offload=False,
    ep_size=moe_cfg.ep_size,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=16386),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=16384)

# Chunked cross-entropy keeps the logits->loss peak memory bounded (never materializes the full
# (seq, vocab) logits) — important for Step-3.5's 128896-token vocab.
loss_cfg = CELossConfig(mode="chunk")


trainer = TrainerConfig(
    load_from=STEP3P5_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=STEP3P5_PATH,
    global_batch_size=16,
    total_step=1000000,
    work_dir="/tmp/step3p5",
    seed=0,
    strict_load=False,
)
