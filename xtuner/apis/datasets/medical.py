from functools import partial

from datasets import load_dataset
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.runner import Runner

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import medical_map_fn
from xtuner.registry import BUILDER


def medical_dataloader(tokenizer,
                       batch_size=1,
                       num_workers=0,
                       path=None,
                       data_config_name='finetune',
                       max_length=2048,
                       pack_to_max_length=True):
    if path is None:
        path = 'shibing624/medical'
    ds = medical_dataset(
        tokenizer,
        max_length=max_length,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)
    dl_cfg = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=ds,
        sampler=dict(type=DefaultSampler, shuffle=True),
        collate_fn=dict(type=default_collate_fn))

    dl_cfg = Config(dl_cfg)
    dl = Runner.build_dataloader(dl_cfg)
    return dl


def medical_dataset(tokenizer,
                    path=None,
                    data_config_name='finetune',
                    max_length=2048,
                    pack_to_max_length=True):
    if path is None:
        path = 'shibing624/medical'
    ds_cfg = dict(
        type=process_hf_dataset,
        dataset=dict(type=load_dataset, path=path, name=data_config_name),
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=medical_map_fn,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    ds_cfg = Config(ds_cfg)
    ds = BUILDER.build(ds_cfg)
    return ds


def medical_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
