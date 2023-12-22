# Copyright (c) OpenMMLab. All rights reserved.
from datasets import load_dataset
# from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import process_intern_repo_dataset
from xtuner.dataset.collate_fns import intern_repo_collate_fn, default_collate_fn
from xtuner.dataset.map_fns import template_map_fn_factory, wizardlm_map_fn
from xtuner.engine import DatasetInfoHook, ThroughputHook, EvaluateChatHook
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE
import torch
from deepspeed.ops.adam import FusedAdam
from xtuner.dataset.intern_repo_packed_dataset import build_packed_dataset, StaticBatchSampler, packed_collate_fn
from xtuner.dataset.intern_repo_packed_dataset import DefaultSampler
from torch.utils.data import BatchSampler
from xtuner.engine.runner.loops import EpochBasedTrainLoop

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/mnt/petrelfs/share_data/gaojianfei/7B_kaoshi_7_5_hf'
use_local_attn = True

# Data
dataset_folder = '/mnt/petrelfs/share_data/gaojianfei/llamav7_8k/train'  # yudong sft data
prompt_template = PROMPT_TEMPLATE.internlm_chat
max_length = 8192
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 4  # 2bs * 16acc * 4gpu = 128 batchsize
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.95)
weight_decay = 0.01
max_norm = 1  # grad clip
total_iters = 3669
warm_up_ratio = 0.025

# Evaluate the generation performance during the training
evaluation_freq = 2000
SYSTEM = ''
evaluation_inputs = [
    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_local_attn=use_local_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_packed_dataset,
    folder=dataset_folder,
    packed_length=max_length,
    min_length=0)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True, seed=1024),
    batch_sampler=dict(type=BatchSampler, drop_last=True, batch_size=accumulative_counts),
    collate_fn=dict(type=packed_collate_fn, packed_length=max_length, accumulative_counts=accumulative_counts))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    )

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1/40, by_epoch=False, begin=0, end=total_iters * warm_up_ratio),
    dict(
        type=CosineAnnealingLR,
        eta_min=lr * 0.15,
        by_epoch=True,
        T_max=max_epochs,
        convert_to_iter_based=True)
]


# train, val, test setting
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    # dict(
    #     type=EvaluateChatHook,
    #     tokenizer=tokenizer,
    #     every_n_iters=evaluation_freq,
    #     evaluation_inputs=evaluation_inputs,
    #     system=SYSTEM,
    #     prompt_template=prompt_template),
    dict(
        type=ThroughputHook
    )
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

log_processor = dict(mean_pattern=r'.*(loss|time|data_time|grad_norm|tflops).*')
