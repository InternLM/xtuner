"""Verification config for async checkpoint with ~8B parameter model.

Uses Qwen3-8B architecture (36 layers, hidden=4096) to produce ~87 GB
checkpoint (model + optimizer), large enough to demonstrate async benefit.

Requires >= 128 GB host memory per rank for CPU staging.

Environment variables:
  ASYNC_CKPT   - "1" (default) for async, "0" for sync
  CKPT_INTERVAL - checkpoint interval in steps (default: 10)
  TOTAL_STEP    - total training steps (default: 100)
  WORK_DIR      - override work directory path (optional)

Usage:
  # ---- Experiment 1: High-frequency checkpoint, local SSD ----
  # Async
  torchrun --nproc_per_node=8 xtuner/v1/train/cli/sft.py \
    --config examples/v1/config/sft_qwen3_8b_async_verify.py \
    2>&1 | tee logs/sft_async_highfreq_qwen3_8b_$(date +%Y%m%d_%H%M%S).log

  # Sync baseline
  ASYNC_CKPT=0 torchrun --nproc_per_node=8 xtuner/v1/train/cli/sft.py \
    --config examples/v1/config/sft_qwen3_8b_async_verify.py \
    2>&1 | tee logs/sft_sync_highfreq_qwen3_8b_$(date +%Y%m%d_%H%M%S).log

  # ---- Experiment 2: Slow storage (NFS/HDFS mount) ----
  # Point WORK_DIR to a network mount to amplify I/O gap
  WORK_DIR=/mnt/nfs/ckpt_bench torchrun --nproc_per_node=8 \
    xtuner/v1/train/cli/sft.py \
    --config examples/v1/config/sft_qwen3_8b_async_verify.py \
    2>&1 | tee logs/sft_async_nfs_qwen3_8b_$(date +%Y%m%d_%H%M%S).log

  ASYNC_CKPT=0 WORK_DIR=/mnt/nfs/ckpt_bench torchrun --nproc_per_node=8 \
    xtuner/v1/train/cli/sft.py \
    --config examples/v1/config/sft_qwen3_8b_async_verify.py \
    2>&1 | tee logs/sft_sync_nfs_qwen3_8b_$(date +%Y%m%d_%H%M%S).log

Analysis:
  Compare logs by grepping for these key timing markers:
    [Checkpoint Breakdown]      - per-checkpoint blocking time breakdown
    [Checkpoint Total Blocking] - total wall-clock blocking per save
    [Async Checkpoint] Staging  - GPU->CPU staging wait (async only)
    [Async Checkpoint] Upload   - disk I/O wait (async only)
    [DCP Collect Model/Optimizer State Dict] - state_dict collection time
    [DCP save/async_save]       - actual save/async_save API time
    Training finished in        - total training wall-clock

Verification checklist:
  - [ ] Async run exits cleanly (no hang after last step)
  - [ ] Checkpoint files are complete: ls <work_dir>/checkpoints/ckpt-step-*/
  - [ ] Loss/grad_norm matches between async and sync runs (diff < 1e-4)
  - [ ] Compare total [Checkpoint Total Blocking] sums between runs
  - [ ] Check if [Async Checkpoint] Staging shows non-zero waits
        (means checkpoint_interval is tight enough to stress the pipeline)
"""

import os

from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.pt_tokenize_fn import PretrainTokenizeFunctionConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3Dense8BConfig
from xtuner.v1.train import TrainerConfig

# Toggle via environment variable for easy A/B comparison
async_checkpoint = os.environ.get("ASYNC_CKPT", "1") != "0"

# Configurable checkpoint frequency and total steps for different experiments:
#   - checkpoint_interval=10 with step_time~2s means ~20s between saves.
#     Since each save takes ~11s (async) or ~14s (sync), the pipeline is
#     stressed enough that wait_prev > 0 will appear in async mode,
#     revealing the true overlap benefit.
#   - total_step=100 gives 10 checkpoint events, enough to average out
#     cold-start effects while keeping wall-clock under 10 minutes.
checkpoint_interval = int(os.environ.get("CKPT_INTERVAL", "10"))
total_step = int(os.environ.get("TOTAL_STEP", "100"))

# Optional: override work_dir to test on slow storage (NFS/HDFS)
work_dir = os.environ.get("WORK_DIR", None)

# 36 layers with full hidden dimensions (~8.7B params)
model_cfg = Qwen3Dense8BConfig(
    num_hidden_layers=36,
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=151936,
)

# Reuse existing test data
sample_max_length = 4096
pack_max_length = 4096

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="pretrain_text",
            anno_path="tests/resource/pretrain_example_data.jsonl",
            sample_ratio=1.0,
        ),
        "tokenize_fn": PretrainTokenizeFunctionConfig(
            add_bos_token=False,
            add_eos_token=True,
        ),
    },
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
)

optim_cfg = AdamWConfig(lr=2e-5, foreach=True)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0.05)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=32,
    total_step=total_step,
    checkpoint_interval=checkpoint_interval,
    async_checkpoint=async_checkpoint,
    work_dir=work_dir,
    hf_interval=0,
)
