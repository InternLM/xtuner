from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import default_collate_fn
from mmchat.datasets.map_fns import openorca_map_fn

orca = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='Open-Orca/OpenOrca'),
    tokenizer=None,
    max_length=2048,
    map_fn=openorca_map_fn,
    remove_columns=['id', 'system_prompt', 'question', 'response'],
    concat_to_max_length=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=orca,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
