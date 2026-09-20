import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.train import TrainerConfig


MODEL_PATH = os.environ["MODEL_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]

ep_size = 8

moe_cfg = get_model_config_from_hf(MODEL_PATH)
moe_cfg.dispatcher = "all2all"
moe_cfg.ep_size = ep_size
moe_cfg.compile_cfg = False
if hasattr(moe_cfg.attention, "sparse_mla_backend"):
    moe_cfg.attention.sparse_mla_backend = "tilelang"

optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    ep_size=ep_size,
    tp_size=2,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(chat_template="glm5.2", max_length=16384),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=16384)

loss_cfg = CELossConfig(mode="chunk", chunk_size=1024)
moe_cfg.lm_loss_cfg = loss_cfg

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
    strict_load=True,
    global_batch_size=8,
    intra_layer_micro_batch=2,
    sp_size=2,
    total_step=20,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
)
