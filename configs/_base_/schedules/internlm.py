from bitsandbytes.optim import PagedAdamW32bit
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR

lr = 2e-4
betas = (0.9, 0.95)
weight_decay = 0.01
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=PagedAdamW32bit, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=1, error_if_nonfinite=False),
    accumulative_counts=16,
    loss_scale='dynamic',
    dtype='float16')

max_epochs = 3
# learning policy
param_scheduler = dict(
    type=CosineAnnealingLR,
    eta_min=lr * 0.1,
    by_epoch=True,
    T_max=max_epochs,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1)
