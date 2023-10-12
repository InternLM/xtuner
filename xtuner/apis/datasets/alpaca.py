# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from datasets import load_dataset
from torch.utils.data import ConcatDataset

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import (alpaca_map_fn, alpaca_zh_map_fn,
                                    template_map_fn_factory)
from xtuner.utils import PROMPT_TEMPLATE


def alpaca_enzh_dataset(tokenizer,
                        path_en='tatsu-lab/alpaca',
                        path_zh='silk-road/alpaca-data-gpt4-chinese',
                        max_length=2048,
                        prompt_template=PROMPT_TEMPLATE.default,
                        remove_unused_columns=True,
                        pack_to_max_length=True):
    alpaca = alpaca_dataset(
        tokenizer,
        path=path_en,
        max_length=max_length,
        prompt_template=prompt_template,
        shuffle_before_pack=True,
        remove_unused_columns=remove_unused_columns,
        pack_to_max_length=pack_to_max_length)
    alpaca_zh = alpaca_zh_dataset(
        tokenizer,
        path=path_zh,
        max_length=max_length,
        prompt_template=prompt_template,
        shuffle_before_pack=True,
        remove_unused_columns=remove_unused_columns,
        pack_to_max_length=pack_to_max_length)
    dataset = ConcatDataset([alpaca, alpaca_zh])
    return dataset


def alpaca_enzh_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)


def alpaca_zh_dataset(tokenizer,
                      path='silk-road/alpaca-data-gpt4-chinese',
                      max_length=2048,
                      prompt_template=PROMPT_TEMPLATE.default,
                      remove_unused_columns=True,
                      pack_to_max_length=True):
    template_map_fn = template_map_fn_factory(template=prompt_template)
    dataset_org = load_dataset(path)
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=alpaca_zh_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=remove_unused_columns,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def alpaca_zh_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)


def alpaca_dataset(tokenizer,
                   path='tatsu-lab/alpaca',
                   max_length=2048,
                   prompt_template=PROMPT_TEMPLATE.default,
                   remove_unused_columns=True,
                   pack_to_max_length=True):
    template_map_fn = template_map_fn_factory(template=prompt_template)
    dataset_org = load_dataset(path)
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=alpaca_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=remove_unused_columns,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def alpaca_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
