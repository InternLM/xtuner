from functools import partial

from datasets import load_dataset

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn


def alpaca_dataset(tokenizer,
                   path='tatsu-lab/alpaca',
                   max_length=2048,
                   pack_to_max_length=True):
    dataset_org = load_dataset(path)
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=alpaca_map_fn,
        remove_columns=['instruction', 'text'],
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def alpaca_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
