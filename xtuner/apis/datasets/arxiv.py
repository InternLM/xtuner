from functools import partial

from datasets import load_dataset
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.runner import Runner

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import arxiv_map_fn
from xtuner.registry import BUILDER


def arxiv_dataloader(tokenizer,
                     batch_size=1,
                     num_workers=0,
                     data_file=None,
                     max_length=2048,
                     pack_to_max_length=True):
    ds = arxiv_dataset(
        tokenizer,
        data_file=data_file,
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


def arxiv_dataset(tokenizer,
                  data_file=None,
                  max_length=2048,
                  pack_to_max_length=True):
    # 1. Download data from https://kaggle.com/datasets/Cornell-University/arxiv  # noqa: E501
    # 2. Process data with `./tools/data_preprocess/arxiv.py`
    if data_file is None:
        data_file = './data/arxiv_postprocess_csAIcsCLcsCV_20200101.json'

    ds_cfg = dict(
        type=process_hf_dataset,
        dataset=dict(
            type=load_dataset, path='json', data_files=dict(train=data_file)),
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=arxiv_map_fn,
        remove_columns=[
            'id', 'submitter', 'authors', 'title', 'comments', 'journal-ref',
            'doi', 'report-no', 'categories', 'license', 'abstract',
            'versions', 'update_date', 'authors_parsed'
        ],
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    ds_cfg = Config(ds_cfg)
    ds = BUILDER.build(ds_cfg)
    return ds


def arxiv_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
