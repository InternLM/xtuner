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
from PIL import Image
from torch import distributed as dist
from torch import nn
from tqdm import tqdm

from xtuner.dataset.hybrid._pack import _PackDataset
from xtuner.dataset.hybrid.mappings import map_protocol, map_sequential
from xtuner.dataset.utils import expand2square
from xtuner.registry import BUILDER
from xtuner.types import HybridChatTemplate
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
    added_keys=dict(image_ranges=list), )
def _register_empty_img_ranges(data, tokenizer=None, chat_template=None):
    if 'image_ranges' not in data:
        data['image_ranges'] = []
    return data


@map_protocol(
    input_keys=dict(
        input_ids=list,
        labels=list,
        tokens=int,
        image_urls=list,
        image_ranges=list,
        position_ids=list,
        cumulative_len=list),
    output_keys=dict(
        input_ids=list,
        labels=list,
        tokens=int,
        image_urls=list,
        image_ranges=list,
        position_ids=list,
        cumulative_len=list))
def _check_mapped_data(item, tokenizer=None, chat_template=None):
    assert isinstance(item['input_ids'][0], int)
    assert isinstance(item['labels'][0], int)

    if len(item['image_urls']) > 0:
        assert isinstance(item['image_urls'][0], str)

    if len(item['image_ranges']) > 0:
        assert isinstance(item['image_ranges'][0], list)
        assert isinstance(item['image_ranges'][0][0], int)

    return item


class HybridDataset(torch.utils.data.Dataset):
    """
    Args:
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding.
        max_length: Max length of the sequence.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        use_varlen_attn: If use_varlen_attn is True, we calculate attention
            the actual length of the sequence rather than the actual length
            of the sequence
    """

    def __init__(self,
                 tokenizer,
                 chat_template: Union[Dict, HybridChatTemplate],
                 sample_ratio: int = 1.0,
                 max_length: int = 2048,
                 pack_to_max_length: bool = False,
                 num_workers: int = 8,
                 mappings: Union[Callable, List[Callable]] = [],
                 data_dir: Optional[str] = None,
                 data_files: Optional[Union[str, List[str]]] = None,
                 data_cached: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 image_processor: Optional[nn.Module] = None,
                 pad_img_to_squared: bool = True):
        super().__init__()

        assert data_dir or data_files or data_cached

        self.tokenizer = build_tokenizer(tokenizer)

        if isinstance(chat_template, HybridChatTemplate):
            self.chat_template = chat_template
        elif isinstance(chat_template, dict):
            self.chat_template = BUILDER.build(chat_template)
        else:
            raise TypeError

        if isinstance(image_processor, dict):
            image_processor = BUILDER.build(image_processor)
        self.image_processor = image_processor

        if image_dir:
            self.image_dir = Path(image_dir)
        else:
            self.image_dir = Path('')

        self.pad_img_to_squared = pad_img_to_squared

        self.sample_ratio = sample_ratio
        self.max_length = max_length
        self.pack_to_max_length = pack_to_max_length

        mappings.append(_register_cumulative_len)
        mappings.append(_register_position_ids)
        mappings.append(_register_tokens)
        mappings.append(_register_empty_img_ranges)
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
        self.analysis_image_samples(dataset)

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

        filter_fn = lambda item: any(item['labels'][i] >= 0 for i in range(
            self.max_length))  # noqa: E501, E731

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

    def analysis_image_samples(self, dataset):

        img_sample_counter = lambda item: len(item['image_urls']
                                              ) > 0  # noqa: E501, E731
        img_counter = lambda item: len(item['image_urls'])  # noqa: E501, E731

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(
                tqdm(
                    executor.map(img_counter, dataset),
                    desc='Count Images',
                    total=len(dataset)))

            samples = list(
                tqdm(
                    executor.map(img_sample_counter, dataset),
                    desc='Count Contain Image Samples',
                    total=len(dataset)))

        num_images = sum(images)
        num_samples = sum(samples)
        print_log(
            f'There are a total of {num_samples} samples with images, '
            f'amounting to {num_images} images.',
            logger='current')

    def analysis_tokens_labels(self, dataset):

        label_counter = lambda item: sum([1 for i in item['labels']
                                          if i >= 0])  # noqa: E501, E731
        token_counter = lambda item: len(item['input_ids'])

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
            'image_dir': str(self.image_dir),
            'data_files': self.data_files,
            'max_length': self.max_length,
            'chat_template': self.chat_template.model_dump(),
            'pack_to_max_length': self.pack_to_max_length,
            'tokenizer': type(self.tokenizer).__name__,
        }

        with open(cache_dir / 'dataset_configuration.json', 'w') as f:
            json.dump(dset_conf, f)

        self.tokenizer.save_pretrained(cache_dir / 'tokenizer')
        self.image_processor.save_pretrained(cache_dir / 'image_processor')

    def load_image(self, url):
        image_file = self.image_dir / url
        image = Image.open(image_file).convert('RGB')

        if self.pad_img_to_squared:
            background = tuple(
                int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, background)

        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        return image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, List]:

        data = self.dataset[item]

        pixel_values = []
        for url in data['image_urls']:
            image = self.load_image(url)

            pixel_values.append(image)

        data['pixel_values'] = pixel_values

        return data


if __name__ == '__main__':

    from transformers import CLIPImageProcessor

    chat_template = HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'],
        image_token='<image>',
        function_call=
        '{assistant}<|action_start|><|plugin|>\n{function_call}<|action_end|><|im_end|>\n',  # noqa: E501
        function_result=
        '<|im_start|>environment name=<|plugin|>\n{function_result}<|im_end|>\n<|im_start|>assistant\n',  # noqa: E501
        functions='<|im_start|>system name=<|plugin|>\n{functions}<|im_end|>\n'
    )

    processor = CLIPImageProcessor.from_pretrained(
        'openai/clip-vit-large-patch14-336',
        trust_remote_code=True,
    )

    from xtuner.dataset.hybrid.mappings import (
        insert_img_pad_tokens, llava_to_openai, openai_to_raw_training)

    data_dir = './llava_data/LLaVA-Instruct-150K/'
    image_dir = './llava_data/llava_images/'
    data_files = 'llava_v1_5_mix665k.json'

    dataset = HybridDataset(
        'internlm/internlm2-chat-1_8b',
        chat_template,
        sample_ratio=1,
        max_length=32*1024,
        data_dir=data_dir,
        data_files=data_files,
        image_dir=image_dir,
        image_processor=processor,
        pack_to_max_length=True,
        mappings=[
            llava_to_openai, openai_to_raw_training, insert_img_pad_tokens,
        ],
        num_workers=4)

    print(dataset[0])

    dataset.cache('cached_llava')
    dataset = HybridDataset(
        'internlm/internlm2-chat-1_8b',
        chat_template,
        sample_ratio=1,
        max_length=32*1024,
        data_cached='cached_llava',
        image_dir=image_dir,
        image_processor=processor,
        pack_to_max_length=True,
        mappings=[
            llava_to_openai, openai_to_raw_training, insert_img_pad_tokens,
        ],
        num_workers=4)
    print(dataset[0])

    from mmengine.dataset import DefaultSampler
    from torch.utils.data import DataLoader

    from xtuner.dataset.hybrid.collate import hybrid_collate_fn
    loader = DataLoader(
        dataset,
        4,
        num_workers=0,
        collate_fn=hybrid_collate_fn,
        sampler=DefaultSampler(dataset, shuffle=True))

    for data in tqdm(loader):
        continue
