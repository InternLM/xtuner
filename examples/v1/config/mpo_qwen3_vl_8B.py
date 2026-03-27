"""
DPO (Direct Preference Optimization) Configuration for Qwen3-VL-8B

This configuration demonstrates how to use DPO/MPO for offline preference learning
in xtuner v1 framework, following the same pattern as RL configs.

Supported loss types:
- sigmoid: Standard DPO loss for preference learning
- bco_pair: Binary Classifier Optimization for absolute quality
- sft: Supervised Fine-Tuning loss to maintain generation quality

For MPO (Mixed Preference Optimization), use:(as used in the MPO paper)
    loss_types=["sigmoid", "bco_pair", "sft"]
    loss_weights=[0.8, 0.2, 1.0]

Usage:
    # Set environment variables
    export WORK_DIR=/path/to/work_dir
    export MODEL_PATH=/path/to/model
    export META_DATA_PATH=/path/to/meta.json
    export CEPH_CONFIG=/path/to/ceph.conf
    export TOKENIZER_CACHE_DIR=/path/to/tokenizer_cache_dir
    # Run with torchrun
    torchrun --nproc_per_node=8 xtuner/v1/train/cli/dpo.py --config mpo_qwen3_vl_8B.py
"""

import json

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLDPOTokenizeFnConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.model import Qwen3VLDense8BConfig
from xtuner.v1.rl.dpo import DPOLossConfig
from xtuner.v1.train.dpo_trainer import DPOTrainerConfig
import os


ceph_config = os.environ["CEPH_CONFIG"]
meta_data_path = os.environ["META_DATA_PATH"]
model_path = os.environ["MODEL_PATH"]
work_dir = os.environ["WORK_DIR"]
tokenizer_cache_dir = os.environ["TOKENIZER_CACHE_DIR"]

# basic settings
# global_batch_size = num_gpus × per_device_batch_size × gradient_accumulation_steps x sp_size
total_epochs = 1
global_batch_size = 64 # suppose 256
per_device_batch_size = 1
gradient_accumulation_steps = 4
max_length = 4096 * 2
pack_max_length = 4096 * 2
num_workers = 8
save_interval = 5000
log_interval = 1000

# Learning rate settings
lr = 5e-6  # Lower LR for DPO
# Paper: cosine decay with minimum learning rate 0
lr_min = 0
# Paper: linear warmup for first 5% of total training steps
warmup_ratio = 0.05
weight_decay = 0.05

model_cfg = Qwen3VLDense8BConfig()

# DPO Loss Configuratio
# Option 1: Standard DPO (sigmoid only)
# loss_cfg = DPOLossConfig(
#     loss_types=["sigmoid"],
#     loss_weights=[1.0],
#     beta=0.1,
# )

# MPO (Mixed Preference Optimization) - combines DPO, BCO, and SFT losses
loss_cfg = DPOLossConfig(
    loss_types=["sigmoid", "bco_pair", "sft"],
    loss_weights=[0.8, 0.2, 1.0],
    beta=0.1,
    label_smoothing=0.0,
    reference_free=False,
    use_average_log_prob=False,
    mode="chunk",
    chunk_size=512,
    ignore_idx=-100,
)

# Dataset Configuration - refer to sft_internvl3.5_8B_config_tiny.py)
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
            class_name='VLMPreferenceJsonlDataset',  # use preference dataset class
            enable_sequential_sampler=True,
            cache_tag='cache_tags_dpo_v1',
            cache_dir=tokenizer_cache_dir,
        ),
        "tokenize_fn": Qwen3VLDPOTokenizeFnConfig(
            processor_path=model_path,
            max_length=max_length,
            min_pixels=_data.get('min_pixels', None),
            max_pixels=_data.get('max_pixels', None),
            oss_loader_cfg=oss_loader_cfg,
            prompt_key="prompt",
            chosen_key="chosen",
            rejected_key="rejected",
            images_key="images",
            add_eos_token=True,  # use parent class defined field names
            system_message=_data.get('system_message', None),
            hash=_data.get('hash', None),
        ),
    }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,# must set to True if using sp_size>1
    pack_level="none",
    collator="qwen3_vl_dpo_collator",  # use DPO collator
    num_workers=num_workers,
    group_by_length=False,  # must be False if pack_level=none
)

# Optimizer and Learning Rate
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)

# FSDP Configuration
fsdp_cfg = FSDPConfig(
    recompute_ratio=1.0,
    vision_recompute_ratio=1.0,
    reshard_after_forward=True,
    checkpoint_preserve_rng_state=False,
    torch_compile=True,
)

# DPO Trainer Configuration
trainer = DPOTrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    load_from=model_path,
    ref_load_from=None,  # Use same model as reference
    tokenizer_path=model_path,
    work_dir=work_dir,
    sp_size=1,
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