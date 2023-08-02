from functools import partial

from datasets import DatasetDict
from mmengine.config.lazy import LazyObject

from mmchat.registry import DATASETS, TOKENIZER
from .utils import Concatenator, encode_fn


def process_hf_dataset(dataset,
                       tokenizer,
                       max_length,
                       split='train',
                       map_fn=None,
                       remove_columns=[],
                       rename_maps=[],
                       concat_to_max_length=True,
                       input_with_labels=True):

    dataset = DATASETS.build(dataset)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]

    if isinstance(map_fn, str):
        map_fn = eval(map_fn)
    if isinstance(map_fn, list):
        assert all(
            [callable(fn) and isinstance(fn, LazyObject) for fn in map_fn])
        for fn in map_fn[:-1]:
            fn = fn.build()
            dataset = dataset.map(fn)
        dataset = dataset.map(
            map_fn[-1].build(), remove_columns=remove_columns)
    elif map_fn is not None:
        dataset = dataset.map(map_fn, remove_columns=remove_columns)
    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)
    tokenizer = TOKENIZER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            input_with_labels=input_with_labels))
    if concat_to_max_length and split == 'train':
        column_names = list(dataset.column_names)
        dataset = dataset.map(
            Concatenator(max_length),
            batched=True,
            remove_columns=column_names)
    return dataset
