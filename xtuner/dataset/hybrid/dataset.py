import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_from_disk
from mmengine import print_log
from torch import distributed as dist
from tqdm import tqdm

from xtuner.dataset.hybrid._pack import _PackDataset
from xtuner.dataset.hybrid.mappings import map_protocol, map_sequential
from xtuner.registry import BUILDER
from xtuner.types import ChatTemplate
from xtuner.utils import build_tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@map_protocol(
    input_keys=dict(input_ids=list),
    added_keys=dict(tokens=int),
)
def _register_tokens(data, tokenizer=None, chat_template=None):
    data['tokens'] = len(data['input_ids'])
    return data


@map_protocol(
    input_keys=dict(input_ids=list),
    added_keys=dict(cumulative_len=list),
)
def _register_cumulative_len(data, tokenizer=None, chat_template=None):
    data['cumulative_len'] = [0, len(data['input_ids'])]
    return data


@map_protocol(
    input_keys=dict(input_ids=list),
    added_keys=dict(position_ids=list),
)
def _register_position_ids(data, tokenizer=None, chat_template=None):
    data['position_ids'] = [i for i in range(len(data['input_ids']))]
    return data


@map_protocol(
    input_keys=dict(
        input_ids=list,
        labels=list,
        tokens=int,
        position_ids=list,
        cumulative_len=list),
    output_keys=dict(
        input_ids=list,
        labels=list,
        tokens=int,
        position_ids=list,
        cumulative_len=list))
def _check_mapped_data(item, tokenizer=None, chat_template=None):
    assert isinstance(item['input_ids'][0], int)
    assert isinstance(item['labels'][0], int)
    return item


class ChatDataset(torch.utils.data.Dataset):
    """"""

    def __init__(self,
                 tokenizer,
                 chat_template: Union[Dict, ChatTemplate],
                 sample_ratio: int = 1.0,
                 max_length: int = 2048,
                 pack_to_max_length: bool = False,
                 num_workers: int = 8,
                 mappings: Union[Callable, List[Callable]] = [],
                 data_dir: Optional[str] = None,
                 data_files: Optional[Union[str, List[str]]] = None,
                 data_cached: Optional[str] = None):
        super().__init__()

        assert data_dir or data_files or data_cached

        self.tokenizer = build_tokenizer(tokenizer)

        if isinstance(chat_template, ChatTemplate):
            self.chat_template = chat_template
        elif isinstance(chat_template, dict):
            self.chat_template = BUILDER.build(chat_template)
        else:
            raise TypeError

        self.sample_ratio = sample_ratio
        self.max_length = max_length
        self.pack_to_max_length = pack_to_max_length

        mappings.append(_register_cumulative_len)
        mappings.append(_register_position_ids)
        mappings.append(_register_tokens)
        mappings.append(_check_mapped_data)
        map_fn = map_sequential(mappings)
        self.map_fn = partial(
            map_fn, tokenizer=self.tokenizer, chat_template=self.chat_template)

        self.num_workers = num_workers
        if data_cached:
            self.data_dir = data_dir
            self.data_files = data_files
            self.data_cached = data_cached
        else:
            data_dir = Path(data_dir)
            if data_files is None:
                data_files = [str(f) for f in data_dir.rglob('*.json')]
            elif isinstance(data_files, list):
                data_files = [str(data_dir / Path(f)) for f in data_files]
            elif isinstance(data_files, str):
                data_files = [str(data_dir / data_files)]
            else:
                raise TypeError

            self.data_dir = str(data_dir)
            self.data_files = data_files
            self.data_cached = data_cached

        self.dataset = self.build_dataset()

    def build_dataset(self):

        if not (dist.is_available() and dist.is_initialized()):
            return self._build_dataset()

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=30)))
        print_log(f'xtuner_dataset_timeout = {timeout}', logger='current')

        gloo_group = dist.new_group(backend='gloo', timeout=timeout)

        if dist.get_rank() == 0:
            dataset = self._build_dataset()
            objects = [dataset]
        else:
            objects = [None]

        dist.monitored_barrier(group=gloo_group, timeout=timeout)
        dist.broadcast_object_list(objects, src=0)

        return objects[0]

    def _build_dataset(self):

        if self.data_cached:
            dataset = load_from_disk(self.data_cached)
            if self.pack_to_max_length:
                dataset = self._pack_dataset(dataset)
            return dataset

        dataset = []
        for file in self.data_files:
            dataset.extend(json.load(open(file)))
            print_log(f'Loaded json data from {file}', logger='current')

        if self.sample_ratio < 1:
            num_samples = int(self.sample_ratio * len(dataset))
            dataset = random.sample(dataset, num_samples)
            print_log(
                f'Randomly selected {num_samples} samples', logger='current')

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            dataset = list(
                tqdm(
                    executor.map(self.map_fn, dataset),
                    desc='Map Dataset',
                    total=len(dataset)))

        dataset = self.filter_non_labels_data(dataset)

        self.analysis_tokens_labels(dataset)

        dataset = Dataset.from_list(dataset)

        if self.pack_to_max_length:
            dataset = self._pack_dataset(dataset)

        return dataset

    def _pack_dataset(self, dataset):

        unpacked_samples = len(dataset)
        dataset = _PackDataset(dataset, self.max_length)
        packed_samples = len(dataset)
        print_log(
            'Before pack multi samples to max length: '
            f'{unpacked_samples} samples',
            logger='current')
        print_log(
            'After pack multi samples to max length: '
            f'{packed_samples} samples',
            logger='current')
        return dataset

    def filter_non_labels_data(self, dataset):

        def filter_fn(item):
            return any(item['labels'][i] >= 0 for i in range(self.max_length))

        ori_samples = len(dataset)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(filter_fn, dataset),
                    desc='Filter Dataset',
                    total=len(dataset)))

        new_dataset = [x for x, passed in zip(dataset, results) if passed]

        new_samples = len(new_dataset)
        print_log(f'Before filter: {ori_samples} samples', logger='current')
        print_log(f'After filter: {new_samples} samples', logger='current')
        print_log(
            f'Filtered {ori_samples - new_samples} samples '
            '(all labels are ignore)',
            logger='current')
        return new_dataset

    def analysis_tokens_labels(self, dataset):

        def label_counter(item):
            return sum([1 for i in item['labels'] if i >= 0])

        def token_counter(item):
            return len(item['input_ids'])

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            tokens = list(
                tqdm(
                    executor.map(token_counter, dataset),
                    desc='Count Tokens',
                    total=len(dataset)))

            labels = list(
                tqdm(
                    executor.map(label_counter, dataset),
                    desc='Count Labels',
                    total=len(dataset)))

        num_tokens = sum(tokens)
        num_labels = sum(labels)
        print_log(
            f'There are a total of {num_tokens} tokens, '
            f'of which {num_labels} tokens need loss calculation.',
            logger='current')

    def cache(self, cache_dir: str):
        cache_dir = Path(cache_dir)

        if self.pack_to_max_length:
            hf_dataset = Dataset.from_list(self.dataset.dataset)
        else:
            hf_dataset = Dataset.from_list(self.dataset)

        hf_dataset.save_to_disk(cache_dir)

        dset_conf = {
            'data_files': self.data_files,
            'max_length': self.max_length,
            'chat_template': self.chat_template.model_dump(),
            'pack_to_max_length': self.pack_to_max_length,
            'tokenizer': type(self.tokenizer).__name__,
        }

        with open(cache_dir / 'dataset_configuration.json', 'w') as f:
            json.dump(dset_conf, f)

        self.tokenizer.save_pretrained(cache_dir / 'tokenizer')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, List]:

        data = self.dataset[item]

        return data


if __name__ == '__main__':

    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'],
    )

    from xtuner.dataset.hybrid.mappings import openai_to_raw_training

    data_dir = './llava_data/LLaVA-Instruct-150K/'
    image_dir = './llava_data/llava_images/'
    data_files = 'llava_v1_5_mix665k.json'

    dataset = ChatDataset(
        'internlm/internlm2-chat-1_8b',
        chat_template,
        sample_ratio=1,
        max_length=32 * 1024,
        data_dir=data_dir,
        data_files=data_files,
        pack_to_max_length=True,
        mappings=[openai_to_raw_training],
        num_workers=4)

    print(dataset[0])

    dataset.cache('cached_llava')
    dataset = ChatDataset(
        'internlm/internlm2-chat-1_8b',
        chat_template,
        sample_ratio=1,
        max_length=32 * 1024,
        data_cached='cached_llava',
        pack_to_max_length=True,
        mappings=[
            openai_to_raw_training,
        ],
        num_workers=4)
    print(dataset[0])

    from mmengine.dataset import DefaultSampler
    from torch.utils.data import DataLoader

    from xtuner.dataset.hybrid.collate import chat_collate_fn
    loader = DataLoader(
        dataset,
        4,
        num_workers=0,
        collate_fn=chat_collate_fn,
        sampler=DefaultSampler(dataset, shuffle=True))

    for data in tqdm(loader):
        continue
