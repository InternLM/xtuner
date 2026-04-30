import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.train import ResumeConfig, TrainerConfig


QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]
CACHE_DIR = os.environ["CACHE_DIR"]


moe_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=False)
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=True,
    cpu_offload=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0, cache_dir=CACHE_DIR),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(chat_template='qwen3', max_length=16384),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=16384)

loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square")


trainer = TrainerConfig(
    load_from=QWEN3_MOE_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=8,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_MOE_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
    resume_cfg=ResumeConfig(auto_resume=True),
)
