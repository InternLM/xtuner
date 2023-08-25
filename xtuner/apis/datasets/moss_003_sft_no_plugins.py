from functools import partial

from xtuner.dataset import MOSSSFTDataset
from xtuner.dataset.collate_fns import default_collate_fn


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
