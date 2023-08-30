# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from torch.utils.data import ConcatDataset

from xtuner.dataset import MOSSSFTDataset
from xtuner.dataset.collate_fns import default_collate_fn


def moss_003_sft_dataset(tokenizer,
                         plugins_data_file=None,
                         no_plugins_data_file=None,
                         bot_name=None,
                         max_length=2048):
    plugins = moss_003_sft_plugins_dataset(
        tokenizer,
        data_file=plugins_data_file,
        bot_name=bot_name,
        max_length=max_length)
    no_plugins = moss_003_sft_no_plugins_dataset(
        tokenizer,
        data_file=no_plugins_data_file,
        bot_name=bot_name,
        max_length=max_length)
    dataset = ConcatDataset([plugins, no_plugins])
    return dataset


def moss_003_sft_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)


def moss_003_sft_no_plugins_dataset(tokenizer,
                                    data_file=None,
                                    bot_name=None,
                                    max_length=2048):

    # Download data from https://huggingface.co/datasets/fnlp/moss-003-sft-data
    if data_file is None:
        data_file = './data/moss-003-sft-no-tools.jsonl'
    dataset = MOSSSFTDataset(
        data_file=data_file,
        bot_name=bot_name,
        tokenizer=tokenizer,
        max_length=max_length)

    return dataset


def moss_003_sft_no_plugins_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)


def moss_003_sft_plugins_dataset(tokenizer,
                                 data_file=None,
                                 bot_name=None,
                                 max_length=2048):

    # Download data from https://huggingface.co/datasets/fnlp/moss-003-sft-data
    if data_file is None:
        data_file = './data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl'  # noqa: E501
    dataset = MOSSSFTDataset(
        data_file=data_file,
        bot_name=bot_name,
        tokenizer=tokenizer,
        max_length=max_length)

    return dataset


def moss_003_sft_plugins_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
