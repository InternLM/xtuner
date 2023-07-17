from mmengine.optim import OptimWrapper
from mmengine._strategy import FSDPStrategy
from mmengine.model.wrappers import MMFullyShardedDataParallel
from bitsandbytes.optim import PagedAdamW32bit
from mmchat.models.utils import ignored_uint8_params
# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=PagedAdamW32bit, lr=0.0002, weight_decay=0.0),
    clip_grad=dict(max_norm=0.3, error_if_nonfinite=True),
)

model_wrapper = dict(type=MMFullyShardedDataParallel,
                     use_orig_params=True,
                     auto_wrap_policy=ignored_uint8_params)

# training strategy
strategy = dict(type=FSDPStrategy, model_wrapper=model_wrapper)


# runner which supports strategies
runner_type = 'FlexibleRunner'

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)