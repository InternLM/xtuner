from bitsandbytes.optim import PagedAdamW32bit
from mmengine.optim import AmpOptimWrapper, ConstantLR, LinearLR

lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1
accumulative_counts = 16

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=PagedAdamW32bit, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

max_epochs = 1
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
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1)
