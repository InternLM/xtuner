# Copyright (c) OpenMMLab. All rights reserved.
# Different from InternLM-Reward, POLAR requires an additional reference trajectory for preference modeling.
# Please refer to https://github.com/InternLM/POLAR for more details.

from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.visualization import TensorboardVisBackend, Visualizer
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from xtuner.dataset.collate_fns.preference_collate_fn import preference_collate_fn
from xtuner.dataset.preference_dataset import build_preference_dataset
from xtuner.engine.hooks import VarlenAttnArgsToMessageHubHook
from xtuner.engine.runner import TrainLoop
from xtuner.model.reward import RewardModel
from xtuner.parallel.sequence import SequenceParallelSampler

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = "internlm/POLAR-7B-Base"
use_varlen_attn = True
reward_token_id = 92527  # use [UNUSED_TOKEN_130] as reward token
loss_type = "ranking"
penalty_type = "none"

# Data
max_length = 16384
max_response_length = 4096
max_packed_length = max_length * 2

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 2
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 1  # reward model should not be trained for more than 1 epoch to avoid overfitting  # noqa: E501
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.95)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
# TODO: eval
# evaluation_freq = 500

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="left",
)

model = dict(
    type=RewardModel,
    use_varlen_attn=use_varlen_attn,
    loss_type=loss_type,
    penalty_type=penalty_type,
    llm=dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
sampler = SequenceParallelSampler if sequence_parallel_size > 1 else DefaultSampler

# preference data format example:
# {
#     "prompt": [{"role": "user", "content": "What is the capital of France?"}],
#     "reference": [{"role": "assistant", "content": "The capital of France is Paris."}],
#     "chosen": [{"role": "assistant", "content": "Paris."}],
#     "rejected": [{"role": "assistant", "content": "I don't know."}],
# }
# Please refer to https://github.com/InternLM/POLAR for more details of data format.

train_dataset = dict(
    type=build_preference_dataset,
    dataset=dict(
        type=load_dataset,
        path="/your/custom/path/here",
    ),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    is_dpo=False,
    is_reward=True,
    reward_token_id=reward_token_id,
    num_proc=32,
    use_varlen_attn=use_varlen_attn,
    max_packed_length=max_packed_length,
    shuffle_before_pack=True,
    max_response_length=max_response_length,
    is_reference=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=preference_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=lr * 0.1,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=lr * 0.1,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = []

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

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
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
