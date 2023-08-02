"""
------------ Dataset Meta Info (after `load_dataset`) ------------

DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 9846
    })
    test: Dataset({
        features: ['text'],
        num_rows: 518
    })
})

------------ Dataset Meta Info (after `process_hf_dataset`) ------------

DatasetDict({
    train: Dataset({
        features: ['text', 'input', 'output'],
        num_rows: 9846
    })
    test: Dataset({
        features: ['text', 'input', 'output'],
        num_rows: 518
    })
})

"""

from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import default_collate_fn
from mmchat.datasets.map_fns import oasst1_map_fn

oasst1 = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='timdettmers/openassistant-guanaco'),
    tokenizer=None,
    max_length=2048,
    map_fn=oasst1_map_fn,
    concat_to_max_length=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=oasst1,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
