import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.train import TrainerConfig


MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]


moe_cfg = Qwen3_5_VLMoE35BA3Config()
moe_cfg.text_config.ep_size = 1
moe_cfg.text_config.mtp_config = MTPConfig(num_layers=1, loss_scaling_factor=1.0)

optim_cfg = AdamWConfig(lr=6e-05, foreach=False, swap_optimizer=True)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=True,
    cpu_offload=False,
    ep_size=1,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca_pretrain", anno_path=DATA_PATH, sample_ratio=1.0),
        "tokenize_fn": PretrainTokenizeFunctionConfig(
            add_bos_token=False,
            add_eos_token=True,
        ),
    },
]

dataloader_config = DataloaderConfig(
    pack_max_length=64 * 1024,
    pack_level="hard",
)

loss_cfg = CELossConfig(mode="chunk", chunk_size=2048, loss_reduction="square")

trainer = TrainerConfig(
    total_step=20,
    load_from=MODEL_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=MODEL_PATH,
    global_batch_size=32,
    # sp_size=2,
    work_dir=os.environ["WORK_DIR"],
    seed=0,
    dist_backend="npu:hccl",
)
