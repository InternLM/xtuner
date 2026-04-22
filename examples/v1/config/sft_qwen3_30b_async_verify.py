"""Verification config for async checkpoint with ~30B parameter model.

Uses Qwen3-32B Dense architecture (64 layers, hidden=5120) to produce ~237 GB
checkpoint (model ~79GB + optimizer ~158GB), large enough to stress async I/O.

Hardware: 8x H200 (141GB each)
Memory estimate per GPU (FSDP 8-way):
  - Model weights (bf16):     ~60GB / 8 =  ~7.5GB
  - Optimizer states (fp32): ~120GB / 8 = ~15.0GB
  - Gradients (bf16):         ~60GB / 8 =  ~7.5GB
  - Activations (grad ckpt):              ~10-20GB
  Total: ~40-50GB per GPU  (well within 141GB)

Requires >= 256 GB host memory per node for CPU staging.

Environment variables:
  ASYNC_CKPT    - "1" (default) for async, "0" for sync
  CKPT_INTERVAL - checkpoint interval in steps (default: 10)
  TOTAL_STEP    - total training steps (default: 100)
  WORK_DIR      - override work directory path (optional)

Usage:
  # Async (default)
  torchrun --nproc_per_node=8 xtuner/v1/train/cli/sft.py \
      --config examples/v1/config/sft_qwen3_30b_async_verify.py \
    2>&1 | tee logs/sft_async_qwen3_30b_$(date +%Y%m%d_%H%M%S).log

  # Sync baseline
  ASYNC_CKPT=0 torchrun --nproc_per_node=8 xtuner/v1/train/cli/sft.py \
      --config examples/v1/config/sft_qwen3_30b_async_verify.py \
    2>&1 | tee logs/sft_sync_qwen3_30b_$(date +%Y%m%d_%H%M%S).log

  # Test on slow storage
  WORK_DIR=/mnt/nfs/ckpt_bench torchrun --nproc_per_node=8 \
      xtuner/v1/train/cli/sft.py \
      --config examples/v1/config/sft_qwen3_30b_async_verify.py \
    2>&1 | tee logs/sft_async_nfs_qwen3_30b_$(date +%Y%m%d_%H%M%S).log

Analysis:
  Compare logs by grepping for these key timing markers:
    [Checkpoint Breakdown]      - per-checkpoint blocking time breakdown
    [Async Checkpoint] Staging  - GPU->CPU staging wait (async only)
    [Async Checkpoint] Upload   - disk I/O wait (async only)
    Training finished in        - total training wall-clock
"""

import os

from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.pt_tokenize_fn import PretrainTokenizeFunctionConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.dense.qwen3 import Qwen3DenseConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.train import TrainerConfig

# ---------------------------------------------------------------------------
# Environment switches
# ---------------------------------------------------------------------------
async_checkpoint = os.environ.get("ASYNC_CKPT", "1") != "0"
checkpoint_save_optimizer = os.environ.get("SAVE_OPTIM", "1") != "0"
checkpoint_interval = int(os.environ.get("CKPT_INTERVAL", "50"))
total_step = int(os.environ.get("TOTAL_STEP", "500"))
work_dir = os.environ.get("WORK_DIR", None)

# ---------------------------------------------------------------------------
# Model — Qwen3-32B Dense architecture (no pretrained weights, pure verify)
# ---------------------------------------------------------------------------
model_cfg = Qwen3DenseConfig(
    vocab_size=151936,
    max_position_embeddings=32768,
    bos_token_id=151643,
    eos_token_id=151645,
    num_hidden_layers=64,
    max_window_layers=64,
    hidden_size=5120,
    intermediate_size=17408,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    hidden_act="silu",
    attention=MHAConfig(
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        qk_norm=True,
        sliding_window=None,
    ),
    tie_word_embeddings=False,
)

# ---------------------------------------------------------------------------
# Data — reuse existing test data
# ---------------------------------------------------------------------------
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
    pack_max_length=4096,
)

# ---------------------------------------------------------------------------
# Optimizer & LR
# ---------------------------------------------------------------------------
optim_cfg = AdamWConfig(lr=2e-5, foreach=True)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0.05)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
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
    checkpoint_save_optimizer=checkpoint_save_optimizer,
    work_dir=work_dir,
)
