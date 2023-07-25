from functools import partial

from mmchat.registry import DATASETS, TOKENIZER
from .utils import Concatenator, encode_fn


def process_hf_dataset(dataset,
                       tokenizer,
                       max_length,
                       mode='train',
                       map_fn=None,
                       remove_columns=[],
                       rename_maps=[],
                       concat_to_max_length=True,
                       predict_with_generation=False):

    dataset = DATASETS.build(dataset)
    if isinstance(map_fn, str):
        map_fn = eval(map_fn)
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)
    tokenizer = TOKENIZER.build(tokenizer)
    column_names = list(dataset[mode].column_names)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            with_output=predict_with_generation is False))
    if concat_to_max_length and mode == 'train':
        dataset = dataset.map(
            Concatenator(max_length),
            batched=True,
            remove_columns=column_names)
    return dataset[mode]
