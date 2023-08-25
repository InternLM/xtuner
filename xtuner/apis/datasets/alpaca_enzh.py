from functools import partial

from torch.utils.data import ConcatDataset

from xtuner.dataset.collate_fns import default_collate_fn
from .alpaca import alpaca_dataset
from .alpaca_zh import alpaca_zh_dataset


def alpaca_enzh_dataset(tokenizer,
                        path_en='tatsu-lab/alpaca',
                        path_zh='silk-road/alpaca-data-gpt4-chinese',
                        max_length=2048,
                        pack_to_max_length=True):
    alpaca = alpaca_dataset(
        tokenizer,
        path=path_en,
        max_length=max_length,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)
    alpaca_zh = alpaca_zh_dataset(
        tokenizer,
        path=path_zh,
        max_length=max_length,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)
    dataset = ConcatDataset([alpaca, alpaca_zh])
    return dataset


def alpaca_enzh_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
