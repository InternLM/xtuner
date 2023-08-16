from functools import partial

from datasets import load_dataset
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.registry import DATASETS
from mmengine.runner import Runner

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import alpaca_map_fn


def alpaca_dataloader(tokenizer,
                      batch_size=1,
                      num_workers=0,
                      path=None,
                      max_length=2048,
                      concat_to_max_length=True):
    if path is None:
        path = 'tatsu-lab/alpaca'
    ds = alpaca_dataset(
        tokenizer,
        path=path,
        max_length=max_length,
        concat_to_max_length=concat_to_max_length)
    dl_cfg = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=ds,
        sampler=dict(type=DefaultSampler, shuffle=True),
        collate_fn=dict(type=default_collate_fn))
    dl_cfg = Config(dl_cfg)
    dl = Runner.build_dataloader(dl_cfg)
    return dl


def alpaca_dataset(tokenizer,
                   path=None,
                   max_length=2048,
                   concat_to_max_length=True):
    if path is None:
        path = 'tatsu-lab/alpaca'
    ds_cfg = dict(
        type=process_hf_dataset,
        dataset=dict(type=load_dataset, path=path),
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=alpaca_map_fn,
        remove_columns=['instruction', 'text'],
        concat_to_max_length=concat_to_max_length)
    ds_cfg = Config(ds_cfg)
    ds = DATASETS.build(ds_cfg)
    return ds


def alpaca_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
