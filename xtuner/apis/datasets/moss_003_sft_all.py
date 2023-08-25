from functools import partial

from torch.utils.data import ConcatDataset

from xtuner.dataset.collate_fns import default_collate_fn
from .moss_003_sft_no_plugins import moss_003_sft_no_plugins_dataset
from .moss_003_sft_plugins import moss_003_sft_plugins_dataset


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
