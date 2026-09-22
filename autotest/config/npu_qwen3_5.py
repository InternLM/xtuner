import os
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.loss.ce_loss import CELossConfig
try:
    from xtuner.v1.loss.moe_loss import ZLossConfig
except:
    from xtuner.v1.model.moe.moe import ZLossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig

QWEN3_MOE_PATH = "/mnt/hwfile/llmrazor/qa-llm-cicd/qa_test_models/Qwen/Qwen3.5-35B-A3B"
#ALPACA_PATH = "/mnt/intern-delivery-shared/lintianyang/share/sampled_jsonls"
ALPACA_PATH = "/mnt/hwfile/llmrazor/qa-llm-cicd/xtuner_resource/datasets/alpaca"
#ALPACA_PATH = "/mnt/hwfile/llmrazor/qa-llm-cicd/xtuner_resource/datasets/tmpdata"

moe_cfg = Qwen3_5_VLMoE35BA3Config()
moe_cfg.text_config.ep_size = 1
if MTPConfig is not None:
    moe_cfg.text_config.mtp_config = [MTPConfig(name="normal", num_layers=1, loss_scaling_factor=1.0, mask_type=None)]
optim_cfg = AdamWConfig(lr=6e-05, foreach=False, swap_optimizer=True)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=True,
    cpu_offload=False,
    ep_size=1,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        #"tokenize_fn": PretrainTokenizeFunctionConfig(),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=16*1024),
    },
]

dataloader_config = DataloaderConfig(
    pack_max_length=64 * 1024,
    pack_level="hard",
)

loss_cfg = CELossConfig(mode="chunk", chunk_size=2048, loss_reduction="square")

trainer = TrainerConfig(
    total_step=50,
    load_from=QWEN3_MOE_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_MOE_PATH,
    global_batch_size=32,
    sp_size=2,
)
