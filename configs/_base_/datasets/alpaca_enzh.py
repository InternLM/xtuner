from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from mmchat.datasets import ConcatDataset
from mmchat.datasets.collate_fns import default_collate_fn

with read_base():
    from .alpaca import alpaca
    from .alpaca_zh import alpaca_zh

train_dataset = dict(
    type=ConcatDataset,
    tokenizer=None,
    datasets_cfg=dict(alpaca=alpaca, alpaca_zh=alpaca_zh))

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
