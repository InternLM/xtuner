"""Qwen3-MoE-30BA3 with region-level selective checkpointing switched on.

Copied from `qwen3_moe_30BA3.py`; the only substantive difference is `recompute_cfg` on the model
config. Everything below `recompute_ratio` (1.0 by default) is still wrapped in a checkpoint -- a
unit named here is kept resident *inside* those layers instead of being recomputed.

To sweep arms, copy this file and change RECOMPUTE_CFG. Do not pass it on the command line.

    RECOMPUTE_CFG = None                      # recompute everything (baseline)
    RECOMPUTE_CFG = True                      # keep every unit the model supports
    RECOMPUTE_CFG = [RecomputeUnit.SAVE_ATTN] # keep exactly these

Available units for an MoE stack (it supports all of them):
    SAVE_ATTN           attention call and its projections
    SAVE_MOE_GATE       router: gating projection, top-k, routing weights
    SAVE_MOE_DISPATCH   permutation/padding around the dispatch and combine all-to-alls
    SAVE_MLP            shared-expert MLP, plus the dense MLP of the first_k_dense_replace layers

Needs torch >= 2.10: regions are carried as fx annotations, which older torch cannot record. With
RECOMPUTE_CFG = None this config runs on 2.9 as well.

Run (8xH200):
    export QWEN3_MOE_PATH=... ALPACA_PATH=...   # see ci/scripts/CI_ENV.sh
    torchrun --nproc-per-node 8 -m xtuner.v1.train.cli.sft --config ci/config/qwen3_moe_30BA3_sac.py
"""

import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train import TrainerConfig
from xtuner.v1.utils.selective_checkpointing import RecomputeUnit


QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]

RECOMPUTE_CFG = [RecomputeUnit.SAVE_ATTN]

moe_cfg = Qwen3MoE30BA3Config(recompute_cfg=RECOMPUTE_CFG)
# moe_cfg = Qwen3MoE30BA3Config(recompute_cfg=RECOMPUTE_CFG)
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=True,
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

loss_cfg = CELossConfig()


trainer = TrainerConfig(
    load_from=QWEN3_MOE_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_MOE_PATH,
    global_batch_size=16,
    # total_epoch=1,
    total_step=15,
    # A 30B HF checkpoint per arm fills the disk fast, and nothing here needs the weights back.
    debug_skip_save=True,
    work_dir="/tmp/qwen3_moe_30BA3_sac",
    seed=0,
)
