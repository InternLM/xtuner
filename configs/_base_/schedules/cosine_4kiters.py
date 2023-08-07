from bitsandbytes.optim import PagedAdamW32bit
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR

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

max_iters = 4000
# learning policy
param_scheduler = dict(
    type=CosineAnnealingLR, eta_min=lr * 0.1, by_epoch=False, T_max=max_iters)

# train, val, test setting
train_cfg = dict(by_epoch=False, max_iters=max_iters, val_interval=max_iters)
