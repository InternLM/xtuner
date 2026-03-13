
from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
    FSDPConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoEFoPEConfig, Qwen3MoEMTPConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.float8 import Float8Config, ScalingGranularity
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.utils.internal_metrics import InternalMetricsConfig
from xtuner.v1.model.moe.moe import ZLossConfig, BalancingLossConfig

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from xtuner_mtp_wrapper import Qwen3MoeMTPXTunerConfig

import os
from pathlib import Path


EP_SIZE = 1
SP_SIZE = 1

INTRA_LAYER_MICRO_BATCH = 1
SEED = 58
LR = 1e-5
LR_MIN = 1e-6
SEQ_LEN = 256
MICRO_BATCH = 1
GLOBAL_BS = 32

CHECKPOINT_INTERVAL = 200
SNAPSHOT_INTERVAL = 1000
CHECK_INTERVAL = 200
INTERNAL_METRIC_INTERVAL = 100


HF_MODEL_PATH = "xxx"  # noqa: E501
CACHE_DIR = (
    "xxx"
)

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="data1",
            anno_path="xxx",  # noqa: E501
            sample_ratio=1.0,
            cache_dir=CACHE_DIR,
        ),
        "tokenize_fn": FTDPTokenizeFnConfig(chat_template="qwen2", max_length=SEQ_LEN),
    },
]

dataloader_config = DataloaderConfig(
    pack_max_length=SEQ_LEN * MICRO_BATCH,
    num_workers=1,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=LR, weight_decay=0.1)
lr_cfg = LRConfig(lr_type="constant", lr_min=LR_MIN, warmup_ratio=0.02)

fsdp_cfg = FSDPConfig(
    torch_compile=True,
)

float8_cfg = Float8Config(
    scaling_granularity_gemm=ScalingGranularity.TILEWISE,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
model_cfg = Qwen3MoeMTPXTunerConfig(model_path=HF_MODEL_PATH)  # type: ignore[arg-type]
model_cfg.float8_cfg = float8_cfg
model_cfg.ep_size = EP_SIZE
model_cfg.z_loss_cfg = ZLossConfig(z_loss_alpha=0)
model_cfg.balancing_loss_cfg = BalancingLossConfig(balancing_loss_alpha=0)

resume_cfg = ResumeConfig(auto_resume=True)

script_path = os.path.abspath(__file__)
script_name = Path(script_path).stem
work_dir = f"xxx"

internal_metrics_cfg = None
# internal_metrics_cfg = InternalMetricsConfig(
#     internal_metrics_interval=INTERNAL_METRIC_INTERVAL,
#     monitor_weights_rms_norm=True,
#     monitor_attn_logits_stats=True,
#     monitor_moe_router_logits_stats=True,  # only applies to MoE models
#     monitor_moe_load_balance_stats=True,
# )

trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,  # type: ignore[arg-type]
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=GLOBAL_BS,
    sp_size=SP_SIZE,
    intra_layer_micro_batch=INTRA_LAYER_MICRO_BATCH,
    total_epoch=20,
    load_from=HF_MODEL_PATH,
    seed=42,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    hf_interval=CHECKPOINT_INTERVAL,
    snapshot_interval=SNAPSHOT_INTERVAL,
    resume_cfg=resume_cfg,
    work_dir=work_dir,
    tokenizer_path=HF_MODEL_PATH,
    strict_load=False,
    exp_tracker="tensorboard",
    skip_checkpoint_validation=True,
    check_health_interval=CHECK_INTERVAL,
    internal_metrics_cfg=internal_metrics_cfg,
)


    
