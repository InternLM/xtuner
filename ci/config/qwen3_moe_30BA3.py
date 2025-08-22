import os
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig


QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


moe_cfg = Qwen3MoE30BA3Config()
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

dataloader_config = DataloaderConfig(
    pack_max_length=16384
)


trainer = TrainerConfig(
    load_from=QWEN3_MOE_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=QWEN3_MOE_PATH,
    global_batch_size=16,
    epoch_num=1,
    work_dir="/tmp/qwen3_moe_30BA3",
    chunked_loss=True,
    seed=0,
)
