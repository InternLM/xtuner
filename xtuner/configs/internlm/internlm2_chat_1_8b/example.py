# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset.train import TextTrainDataset
from xtuner.engine.hooks import ChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import AutoModelForCausalLM, TextFinetune
from xtuner.types import ChatTemplate

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/models--internlm--internlm2-chat-1_8b/snapshots/aa8a7450c2227a3b6733b3c6fe33fefbb2ca54f9/'
use_varlen_attn = False

# Data
alpaca_en_path = 'tatsu-lab/alpaca'
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

chat_template = dict(
    type=ChatTemplate,
    system='<|im_start|>system\n{system}<|im_end|>\n',
    user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
    assistant='{assistant}<|im_end|>\n',
    stop_words=['<|im_end|>'])
#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
model = dict(
    type=TextFinetune,
    tokenizer=tokenizer,
    chat_template=chat_template,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

dataset = dict(
    type=TextTrainDataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
    # data_dir='converted_alpaca',
    data_files=[
        'converted_alpaca/202404121913-shard-2-of-3.json',
        'converted_alpaca/202404121913-shard-3-of-3.json',
        'converted_alpaca/202404121913-shard-1-of-3.json'
    ],
    sample_ratio=[2, 3, 1],
    # cache_dir='cached_alpaca',
    max_length=max_length,
    pack_to_max_length=pack_to_max_length)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=TextFinetune.dataloader_collate_fn)

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
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=ChatHook, prompts=['请介绍五个上海的景点'], every_n_iters=1000)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
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

# set log processor
log_processor = dict(by_epoch=False)
