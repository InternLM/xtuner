import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.qwen3_5_text_split import Qwen3_5_VLTextMoE397BA17BSplitConfig
from xtuner.v1.module.router import GreedyRouterConfig
from xtuner.v1.module.router.greedy import GreedyGroupedRouter
from xtuner.v1.train import TrainerConfig


from xtuner.v1.float8.config import Float8Config, ScalingGranularity

float8_cfg = Float8Config(
    scaling_granularity_gemm=ScalingGranularity.TILEWISE,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)


# QWEN3_5_MOE_SPLIT_PATH = "/mnt/shared-storage-user/llmrazor-share/yehaochen/model/Qwen3.5-35B-A3B-fused"
QWEN3_5_MOE_SPLIT_PATH = "/mnt/shared-storage-user/llmrazor-share/model/models--Qwen--Qwen3.5-397B-A17B/98d1a504ba52e88924b3a3a008447cf2fdbd518c"
ALPACA_PATH = os.environ["ALPACA_PATH"]


# model_cfg = Qwen3_5_VLMoE35BA3SplitConfig(only_llm_forward=True)
# model_cfg.text_config.ep_size = 2
model_cfg = Qwen3_5_VLTextMoE397BA17BSplitConfig(
    ep_size=8, dispatcher="deepep",
    hf_key_mapping={r"^model\.": "model.language_model."},
    float8_cfg=float8_cfg,
    num_hidden_layers=4,
)
ep_size = model_cfg.ep_size
optim_cfg = AdamWConfig(lr=6e-05)

lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=True,
    cpu_offload=False,
    ep_size=ep_size,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=4096),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=8192)

loss_cfg = CELossConfig(mode="chunk")


trainer = TrainerConfig(
    load_from=QWEN3_5_MOE_SPLIT_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_5_MOE_SPLIT_PATH,
    global_batch_size=32,
    work_dir="/mnt/shared-storage-user/llmrazor-share/profiles/qwen3.5-35BA3/fullgraph-ep1-bf16-32k",
    seed=0,
    strict_load=False,
    total_step=1000000,
    profile_step=10,
    intra_layer_micro_batch=4,
)
