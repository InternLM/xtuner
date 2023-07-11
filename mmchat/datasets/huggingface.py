
from typing import Sequence
from mmchat.registry import DATASETS, TOKENIZER
from torch.utils.data import Dataset, DataLoader, ChainDataset
import transformers
import torch
from dataclasses import dataclass
import copy
from torch.nn.utils.rnn import pad_sequence
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
def process_hf_dataset(dataset, mode='train', 
                       prompt_input_format=None,
                       prompt_no_input_format=None,
                       map_fn=lambda x:x,remove_columns=[], rename_maps=[]):

    dataset = DATASETS.build(dataset)
    
    def _prompt_format(example):
        if example.get("input", "") != "":
            prompt_format = prompt_input_format
        else:
            prompt_format = prompt_no_input_format
        return {'input': prompt_format.format(**example)}

    if prompt_input_format and prompt_no_input_format:
        dataset = dataset.map(_prompt_format)

    if isinstance(map_fn, str):
        map_fn = eval(map_fn)
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # Remove unused columns.
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
    )
    return dataset[mode]


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, map_fn, remove_columns=[], rename_maps=[]) -> None:
        super().__init__()

        dataset = DATASETS.build(dataset)
        dataset = dataset.map(map_fn, remove_columns=remove_columns)
        for old, new in rename_maps:
            dataset = dataset.rename_column(old, new)
        self.dataset = dataset

        self.tokenizer = TOKENIZER.build(tokenizer)

    


