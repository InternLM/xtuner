# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from datasets import load_dataset
from torch.utils.data import ConcatDataset

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import (crime_kg_assitant_map_fn,
                                    law_reference_map_fn,
                                    template_map_fn_factory)
from xtuner.utils import PROMPT_TEMPLATE


def lawyer_dataset(tokenizer,
                   crime_data_file=None,
                   reference_data_file=None,
                   max_length=2048,
                   prompt_template=PROMPT_TEMPLATE.default,
                   remove_unused_columns=True,
                   pack_to_max_length=True):
    crime_dataset = lawyer_crime_dataset(
        tokenizer,
        data_file=crime_data_file,
        max_length=max_length,
        prompt_template=prompt_template,
        remove_unused_columns=remove_unused_columns,
        pack_to_max_length=pack_to_max_length)
    reference_dataset = lawyer_reference_dataset(
        tokenizer,
        data_file=reference_data_file,
        max_length=max_length,
        prompt_template=prompt_template,
        remove_unused_columns=remove_unused_columns,
        pack_to_max_length=pack_to_max_length)
    dataset = ConcatDataset([crime_dataset, reference_dataset])
    return dataset


def lawyer_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)


def lawyer_crime_dataset(tokenizer,
                         data_file=None,
                         max_length=2048,
                         prompt_template=PROMPT_TEMPLATE.default,
                         remove_unused_columns=True,
                         pack_to_max_length=True):
    template_map_fn = template_map_fn_factory(template=prompt_template)
    # Download data from https://github.com/LiuHC0428/LAW-GPT  # noqa: E501
    if data_file is None:
        data_file = './data/law/CrimeKgAssitant清洗后_52k.json'
    dataset_org = load_dataset(path='json', data_files=dict(train=data_file))
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=crime_kg_assitant_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=remove_unused_columns,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def lawyer_crime_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)


def lawyer_reference_dataset(tokenizer,
                             data_file=None,
                             max_length=2048,
                             prompt_template=PROMPT_TEMPLATE.default,
                             remove_unused_columns=True,
                             pack_to_max_length=True):
    template_map_fn = template_map_fn_factory(template=prompt_template)
    # Download data from https://github.com/LiuHC0428/LAW-GPT  # noqa: E501
    if data_file is None:
        data_file = './data/law/训练数据_带法律依据_92k.json'
    dataset_org = load_dataset(path='json', data_files=dict(train=data_file))
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=law_reference_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=remove_unused_columns,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def lawyer_reference_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
