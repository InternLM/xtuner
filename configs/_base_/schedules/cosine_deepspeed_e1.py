from mmengine._strategy import DeepSpeedStrategy
from mmengine.optim import CosineAnnealingLR, DeepSpeedOptimWrapper
from torch.optim import AdamW

lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1
accumulative_counts = 16

# optimizer
optim_wrapper = dict(
    type=DeepSpeedOptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, betas=betas, weight_decay=weight_decay))

# training strategy
strategy = dict(
    type=DeepSpeedStrategy,
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=['inputs'],
    gradient_accumulation_steps=accumulative_counts,
    gradient_clipping=max_norm,
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False,
    ))

# runner which supports strategies
runner_type = 'FlexibleRunner'

max_epochs = 1
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
