import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.train import TrainerConfig
from pathlib import Path

MODEL_PATH = os.environ["MODEL_PATH"]
ALPACA_PATH = os.environ["DATA_PATH"]


moe_cfg = Qwen3_5_VLMoE35BA3Config()
moe_cfg.text_config.mtp_config = MTPConfig(num_layers=1, loss_scaling_factor=0.1 )

optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(chat_template='qwen3', max_length=16384),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=16384)

loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction = "square")

trainer = TrainerConfig(
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
    total_epoch=1,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
)
