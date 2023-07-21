from bitsandbytes.optim import PagedAdamW32bit
from mmengine.optim import AmpOptimWrapper, ConstantLR, LinearLR

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=PagedAdamW32bit, lr=0.0002, weight_decay=0.0),
    clip_grad=dict(max_norm=0.3, error_if_nonfinite=True),
    accumulative_counts=1,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    # main learning rate scheduler
    dict(
        type=ConstantLR,
        by_epoch=False,
        factor=1.0,
        begin=500,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1)
