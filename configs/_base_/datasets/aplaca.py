from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collators import CollatorWithPadding
from mmchat.datasets.utils import aplaca_map_fn

_alpaca = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='tatsu-lab/alpaca'),
    map_fn=aplaca_map_fn,
    remove_columns=['instruction', 'text'])

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=_alpaca,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(
        type=CollatorWithPadding,
        tokenizer=None,
        source_max_len=1024,
        target_max_len=1024,
        train_on_source=False,
        predict_with_generate=False))
