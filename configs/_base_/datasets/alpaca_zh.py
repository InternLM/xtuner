from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import default_collate_fn
from mmchat.datasets.map_fns import alpaca_zh_map_fn

alpaca_zh = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='silk-road/alpaca-data-gpt4-chinese'),
    tokenizer=None,
    max_length=2048,
    map_fn=alpaca_zh_map_fn,
    remove_columns=['instruction', 'instruction_zh', 'input_zh', 'output_zh'],
    concat_to_max_length=False)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=alpaca_zh,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
