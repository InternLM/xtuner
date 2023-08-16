from datasets import load_dataset
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.registry import DATASETS
from mmengine.runner import Runner
from functools import partial

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import cmd_map_fn


def cmd_dataloader(tokenizer,
                   batch_size=1,
                   num_workers=0,
                   max_length=2048,
                   concat_to_max_length=True):
    ds = cmd_dataset(
        tokenizer,
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


def cmd_dataset(tokenizer, max_length=2048, concat_to_max_length=True):

    data_url = 'https://github.com/Toyhom/Chinese-medical-dialogue-data/raw/master/Data_数据/'  # noqa: E501

    all_csv = [
        'Andriatria_男科/男科5-13000.csv', 'IM_内科/内科5000-33000.csv',
        'OAGD_妇产科/妇产科6-28000.csv', 'Oncology_肿瘤科/肿瘤科5-10000.csv',
        'Pediatric_儿科/儿科5-14000.csv', 'Surgical_外科/外科5-14000.csv'
    ]

    all_csv = [data_url + csv for csv in all_csv]

    ds_cfg = dict(
        type=process_hf_dataset,
        dataset=dict(
            type=load_dataset,
            path='csv',
            data_files=dict(train=all_csv),
            encoding='GB18030'),
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=cmd_map_fn,
        concat_to_max_length=concat_to_max_length)

    ds_cfg = Config(ds_cfg)
    ds = DATASETS.build(ds_cfg)
    return ds


def cmd_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
