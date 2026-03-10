import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.model.moe.gpt_oss import GptOss21BA3P6Config
from xtuner.v1.train import TrainerConfig


GPTOSS_21B_PATH = os.environ["GPTOSS_21B_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


gptoss_cfg = GptOss21BA3P6Config(rope_scaling_cfg=RopeScalingConfig(type="yarn", beta_fast=16.0, beta_slow=1.05, factor=16.0, original_max_position_embeddings=4096, truncate=True))
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=gptoss_cfg.ep_size,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=16384),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=16384)

loss_cfg = CELossConfig()


trainer = TrainerConfig(
    load_from=GPTOSS_21B_PATH,
    model_cfg=gptoss_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=GPTOSS_21B_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
)
