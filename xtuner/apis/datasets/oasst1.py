from functools import partial

from datasets import load_dataset

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import oasst1_map_fn


def oasst1_dataset(tokenizer,
                   path='timdettmers/openassistant-guanaco',
                   max_length=2048,
                   pack_to_max_length=True):

    dataset_org = load_dataset(path)
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=oasst1_map_fn,
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def oasst1_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
