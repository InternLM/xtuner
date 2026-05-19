import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.deepseek_v4 import DeepSeekV4Config
from xtuner.v1.train import TrainerConfig


# DEEPSEEK_V4_PATH should point at a directory that holds the BF16 DeepSeek-V4-Flash
# release plus its tokenizer files. The BF16 dequant of the 46-shard FP4/FP8 release
# lives at /mnt/shared-storage-user/llmrazor-share/yehaochen/model/DeepSeek-V4-Flash
# on the shared storage (109 safetensor shards, 542 GB). `from_hf` reads the local
# `config.json` to recover the 44-entry `compress_ratios` list and all V4 hyper-params.
DEEPSEEK_V4_PATH = os.environ["DEEPSEEK_V4_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


# Use `from_hf` rather than the default-arg constructor so the per-layer
# `compress_ratios` (length = num_hidden_layers + 1) and other release-specific
# fields (num_hash_layers, swiglu_limit, attn_sink dims) are picked up from the
# checkpoint instead of relying on the Config defaults.
moe_cfg = DeepSeekV4Config.from_hf(DEEPSEEK_V4_PATH)

# Sparse-attention training reference path: the current sparse_attn implementation
# is the pure-PyTorch reference shipped in PR4 (no Triton kernel with backward yet),
# so torch.compile + EP layout is on the conservative side. Switch back on once the
# environment ships a flash_attn build with `sinks=` and a TileLang backward.
moe_cfg.ep_size = 8
moe_cfg.dispatcher = "all2all"
moe_cfg.compile_cfg = False

optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
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
    load_from=DEEPSEEK_V4_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=DEEPSEEK_V4_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir="/tmp/deepseek_v4_flash",
    seed=0,
)
