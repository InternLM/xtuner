from mmengine.optim import DeepSpeedOptimWrapper
from mmengine._strategy import DeepSpeedStrategy
from torch.optim import AdamW
# optimizer
optim_wrapper = dict(
    type=DeepSpeedOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0016, weight_decay=0.0))


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
    gradient_accumulation_steps=1,
    gradient_clipping=0.3,
    zero_optimization=dict(
        stage=3,
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
        type="LinearLR",
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=5,
    ),
    # main learning rate scheduler
    dict(
        type="ConstantLR",
        by_epoch=False,
        factor=1.0,
        begin=5,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=16, enable=False)