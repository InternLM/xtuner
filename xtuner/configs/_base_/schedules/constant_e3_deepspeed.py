from mmengine._strategy import DeepSpeedStrategy
from mmengine.optim import ConstantLR, DeepSpeedOptimWrapper, LinearLR
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

max_epochs = 3
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
