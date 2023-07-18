from mmengine._strategy import DeepSpeedStrategy
from mmengine.optim import DeepSpeedOptimWrapper
from torch.optim import AdamW

# optimizer
optim_wrapper = dict(
    type=DeepSpeedOptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0002, weight_decay=0.0))

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
    gradient_accumulation_steps=16,
    gradient_clipping=0.3,
    zero_optimization=dict(
        stage=1,
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
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)
