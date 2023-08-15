from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import alpaca_zh_map_fn

alpaca_zh = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='silk-road/alpaca-data-gpt4-chinese'),
    tokenizer=None,
    max_length=2048,
    map_fn=alpaca_zh_map_fn,
    remove_columns=['instruction', 'instruction_zh', 'input_zh', 'output_zh'],
    pack_to_max_length=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=alpaca_zh,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
