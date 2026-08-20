"""SAC 验证配置（none）。配套 tools/verify_sac.sh，勿手动单独运行。

eager + all2all：关掉 compile 才能把 loss 差异归因到 SAC 本身而不是编译核。
"""

import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train import TrainerConfig


moe_cfg = Qwen3MoE30BA3Config(ep_size=4, dispatcher="all2all")
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=moe_cfg.ep_size)

trainer = TrainerConfig(
    load_from=os.environ["QWEN3_MOE_PATH"],
    model_cfg=moe_cfg,
    optim_cfg=AdamWConfig(lr=6e-05),
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=[
        {
            "dataset": DatasetConfig(name="alpaca", anno_path=os.environ["ALPACA_PATH"], sample_ratio=1.0),
            "tokenize_fn": FTDPTokenizeFnConfig(max_length=8194),
        },
    ],
    dataloader_cfg=DataloaderConfig(pack_max_length=8192),
    lr_cfg=LRConfig(lr_type="cosine", lr_min=1e-6),
    loss_cfg=CELossConfig(),
    tokenizer_path=os.environ["QWEN3_MOE_PATH"],
    global_batch_size=16,
    total_step=3,
    debug_skip_save=True,
    work_dir="/tmp/sac_verify_none",
    seed=0,
)
