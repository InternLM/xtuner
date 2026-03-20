"""
DPO (Direct Preference Optimization) Configuration for Qwen3-VL-8B with Video Data

This configuration is for video preference learning using LongVA-TPO dataset.
Based on qwen_mla_mpo.py but adapted for video data.

Usage:
    export WORK_DIR=/path/to/work_dir
    export MODEL_PATH=/path/to/model

    torchrun --nproc_per_node=8 xtuner/v1/train/cli/dpo.py --config qwen_mla_mpo_video.py
"""

import json

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLDPOTokenizeFnConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.model import Qwen3VLDense8BMLAConfig
from xtuner.v1.rl.dpo import DPOLossConfig
from xtuner.v1.train.dpo_trainer import DPOTrainerConfig


# ============================================================================
# 路径配置 (Path Configuration)
# ============================================================================
ceph_config = "/mnt/shared-storage-user/lisongze/iv3/xtuner/config/petreloss.conf"
meta_data_path = "/mnt/shared-storage-user/lisongze/iv3/xtuner/dpo_config/LongVA_TPO.json"
model_path = "/mnt/shared-storage-user/lisongze/iv3/20260202-hf-2605"
work_dir = "/mnt/shared-storage-user/iv3/mpo/xtuner_saved_model/qwen3vl/qwen3vl-8B-mpo-mla-sp8-longva-tpo-fix-bug-nosft-lr-1e-6"
tokenizer_cache_dir = "/mnt/shared-storage-user/iv3/mpo/xtuner_tokenizer_cache/qwen3vl/qwen3vl-8B-mpo-mla-sp8-longva-tpo-fix-bug-nosft-lr-1e-6"

# ============================================================================
# Training Settings (训练超参数)
# ============================================================================
total_epochs = 1
global_batch_size = 32  # Reduced for video (larger memory footprint)
per_device_batch_size = 1
gradient_accumulation_steps = 16
max_length = 4096 * 16  # 16384 tokens max
pack_max_length = 4096 * 16
num_workers = 8
save_interval = 2000
log_interval = 500

# Learning rate settings
lr = 1e-6
lr_min = 0
warmup_ratio = 0.05
weight_decay = 0.05

# ============================================================================
# Video Settings (视频参数)
# ============================================================================
video_max_frames = 1024  # Maximum frames per video
video_min_frames = 8   # Minimum frames per video
fps = 2                 # Sampling fps
rand_video_max_frames = 64  # Random sampling max frames

# ============================================================================
# 1. Model Configuration
# ============================================================================
model_cfg = Qwen3VLDense8BMLAConfig()

# ============================================================================
# 2. DPO Loss Configuration - MPO (Mixed Preference Optimization)
# ============================================================================
loss_cfg = DPOLossConfig(
    loss_types=["sigmoid", "bco_pair"],
    loss_weights=[0.8, 0.2],
    beta=0.1,
    label_smoothing=0.0,
    reference_free=False,
    use_average_log_prob=False,
    mode="chunk",
    
    chunk_size=512,
    ignore_idx=-100,
)

# ============================================================================
# 3. Dataset Configuration - Video DPO Data
# ============================================================================
oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []

for name, _data in ds_collections.items():
    _data_cfg = {
        "dataset": DatasetConfig(
            name=name,
            anno_path=_data['annotation'],
            media_root=_data.get('media_root', ''),
            sample_ratio=_data.get('sample_ratio', 1.0),
            class_name='VLMPreferenceJsonlDataset',
            enable_sequential_sampler=True,
            cache_tag='cache_tags_dpo_video_v1',
            cache_dir=tokenizer_cache_dir,
        ),
        "tokenize_fn": Qwen3VLDPOTokenizeFnConfig(
            processor_path=model_path,
            max_length=max_length,
            min_pixels=_data.get('min_pixels', None),
            max_pixels=_data.get('max_pixels', None),
            oss_loader_cfg=oss_loader_cfg,
            # DPO keys for video data
            prompt_key="prompt",
            chosen_key="chosen",
            rejected_key="rejected",
            images_key="images",  # Not used for video, but required by config
            # Video settings
            video_max_frames=_data.get('video_max_frames', video_max_frames),
            video_min_frames=video_min_frames,
            fps=_data.get('fps', fps),
            rand_video_max_frames=rand_video_max_frames,
            add_eos_token=True,
            system_message=_data.get('system_message', None),
            hash=_data.get('hash', None),
        ),
    }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    pack_level="none",  # DPO doesn't need packing
    collator="qwen3_vl_dpo_collator",
    num_workers=num_workers,
    group_by_length=False,
)

# ============================================================================
# 4. Optimizer and Learning Rate
# ============================================================================
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)

# ============================================================================
# 5. FSDP Configuration
# ============================================================================
fsdp_cfg = FSDPConfig(
    recompute_ratio=1.0,
    vision_recompute_ratio=1.0,
    reshard_after_forward=True,
    checkpoint_preserve_rng_state=False,
    torch_compile=True,
)

# ============================================================================
# 6. DPO Trainer Configuration
# ============================================================================
trainer = DPOTrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    load_from=model_path,
    ref_load_from=None,
    tokenizer_path=model_path,
    work_dir=work_dir,
    sp_size=8,
    total_epochs=total_epochs,
    global_batch_size=global_batch_size,
    per_device_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_length=max_length,
    save_interval=save_interval,
    log_interval=log_interval,
    seed=42,
    freeze_ref_model=True,
    use_vlm_collator=True,
    num_workers=num_workers,
    dataloader_cfg=dataloader_config,
)
