# Copyright (c) OpenMMLab. All rights reserved.
import functools
import logging
import os
import random
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, load_from_disk
from mmengine import dump, load, print_log
from mmengine.dist import master_only
from torch import distributed as dist
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from xtuner.dataset.train._pack import _PackDataset
from xtuner.types import ChatMessages, ChatTemplate
from xtuner.utils.config import build_from_cfg_or_obj

_TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# HACK If not set to true, multithreaded tokenization cannot be carried out,
# but transformers do not recommend setting it to true.
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def master_only_load(load_fn):

    @functools.wraps(load_fn)
    def wrapper(*args, **kwargs):

        if not (dist.is_available() and dist.is_initialized()):
            return load_fn(*args, **kwargs)

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=30)))
        print_log(f'xtuner_dataset_timeout = {timeout}', logger='current')

        gloo_group = dist.new_group(backend='gloo', timeout=timeout)

        if dist.get_rank() == 0:
            dataset = load_fn(*args, **kwargs)
            objects = [dataset]
        else:
            objects = [None]

        dist.monitored_barrier(group=gloo_group, timeout=timeout)
        dist.broadcast_object_list(objects, src=0)

        return objects[0]

    return wrapper


class BaseTrainDataset(torch.utils.data.Dataset):
    """A high-performance dataset designed for LLM instruction finetuning.

    Capable of supporting ultra-large-scale training, and can enhance training
    efficiency.

    The following features are provided for extremely large datasets:

        1. Distributed Load and Multi-thread Process the Dataset

            In the traditional data-parallel training process, each rank loads
            and processes the same dataset independently, which involves a
            large amount of redundant operations. Moreover, ranks compete for
            resources with each other, resulting in very low efficiency of
            data loading and processing.

            To address this situation, `TextDataset` loads and processes the
            dataset only on rank 0 and then broadcasts the processed dataset
            to other ranks, avoiding repeated loading on all ranks. When
            loading and processing data, multi-threading is also enabled to
            speed up, significantly improving the efficiency of dataset
            loading.

            This feature is default, developers need no special settings.

        2. Cached Dataset

            Even though the loading and processing of datasets can be speed up
            via distributed and multi-threaded methods, the process might
            still be time-consuming. For cases where the same dataset is used
            for training multiple times, it is possible to cache the processed
            dataset during the first training. This way, there's no need to
            reprocess the data for subsequent training.

            This feature requires developers to set `cache_dir`. During the
            first training, the processed dataset will be automatically cached
            to the directory specified by `cache_dir`. Subsequent training
            will automatically load data from the `cache_dir` path.


        3. Packed Dataset

            In the batch training, the length of each piece of data varies.
            The traditional method pads shorter data within a batch to match
            the length of the longest data. This leads to a wastage of
            computing resources due to pad tokens. Additionally, the maximum
            length of each batch tends to fluctuate, making it challenging to
            maintain a high level of GPU utilization.

            To address this issue, `TextDataset` takes multiple pieces of
            unevenly sized raw data and concatenates them up to `max_length`.
            The last concatenated piece of raw data, if it exceeds the max
            length with its cumulative length, is cut off. The portion
            exceeding the maximum length is used as the beginning of the next
            piece of packed data.

            When using this feature, developer need to set `pack_to_max_length`
            to True.

            Each packed data can compute attention within each original data,
            or treat it as a whole data to compute attention, which can be
            controlled in the `model` by `use_varlen_attn`.

    Args:
        tokenizer (_TokenizerType):
            The original data is a dict composed of strings, which needs
            to be tokenized before it can be trained. The tokenizer must be
            consistent with the one used in the model being trained.
        chat_template (dict | ChatTemplate):
            Different models correspond to different chat template. If it is
            a dict, it will be considered in the style of MMEngine config and
            will be built accordingly. Ultimately, a ChatTemplate instance
            will be obtained.
        sample_ratio (float | list):
            Control over-sampling or under-sampling of the dataset, 2 means
            over-sampling the dataset to double its original size,
            0.5 means under-sampling the dataset to half its original size. If
            you want to set different sampling ratios for data in `data_files`,
            you can pass in a list that is equal in length to `data_files`.
            Default is 1.
        max_length (int):
            The maximum length of a single data in the dataset. When
            `pack_to_max_length` is False, data exceeding the `max_length`
            will be truncated, only preserving the first `max_length` tokens.
            Default is 2048.
        pack_to_max_length (bool):
            If True, multiple data will be packed to form a new dataset. Each
            data in the packed dataset has a length of `max_length`. Default
            is True.
        num_proc (int):
            The number of threads used when processing dataset. Default is 8.
        cache_dir (str):
            The cache path of the dataset. When the path points to an existing
            cached dataset, it will directly load the cached dataset, and the
            settings of `data_dir` and `data_files` will become invalid. When
            the path does not point to a cached dataset, the processed data
            will be cached to this path. Default is None.
        data_files (List[str]):
            The paths of several json files. When `data_dir` is None, it will
            directly load files according to the addresses in `data_files`;
            when `data_dir` is not None, the file paths in `data_files` should
            be relative paths under `data_dir`. When `cache_dir` is not None,
            and `cache_dir` points to a cached dataset, the settings of
            `data_files` become invalid. Default is None.
        data_dir (str):
            When `data_files` is None, it will load all json files in
            `data_dir` and its several tiers of sub-directories; when
            `data_files` is not None, only json files in `data_files` are
            loaded. When `cache_dir` is not None, and `cache_dir` points
            to a  cached dataset, the setting of `data_dir` become invalid.
            Default is None.
    """

    def __init__(self,
                 tokenizer: _TokenizerType,
                 chat_template: Union[Dict, ChatTemplate],
                 sample_ratio: Union[float, List[float]] = 1.0,
                 max_length: int = 2048,
                 pack_to_max_length: bool = True,
                 num_proc: int = 8,
                 data_dir: Optional[str] = None,
                 data_files: Optional[Union[str, List[str]]] = None,
                 cache_dir: Optional[str] = None):
        super().__init__()

        self.tokenizer = build_from_cfg_or_obj(
            tokenizer, accept=(PreTrainedTokenizer, PreTrainedTokenizerFast))

        self.chat_template = build_from_cfg_or_obj(
            chat_template, accept=ChatTemplate)

        if isinstance(sample_ratio, (list, tuple)):
            if len(sample_ratio) != len(data_files):
                raise ValueError('The length of `sample_ratio`'
                                 f'({len(sample_ratio)}) should be the same '
                                 'as the length of `data_files`'
                                 f'({len(data_files)})')
        self.sample_ratio = sample_ratio

        self.max_length = max_length
        self.pack_to_max_length = pack_to_max_length

        self.num_workers = num_proc
        self.dataset = self.load_dataset(data_dir, data_files, cache_dir)

        # When the cache_dir is set and there isn't a cached dataset in it,
        # the tokenized dataset will be cached to the cache_dir.
        if cache_dir and not self.is_cached(cache_dir):
            self.cache_dataset(cache_dir)

    def load_dataset(
            self,
            data_dir: Optional[str] = None,
            data_files: Optional[List[str]] = None,
            cache_dir: Optional[str] = None) -> Union[Dataset, _PackDataset]:
        """Load multiple JSON files, or the cached tokenized dataset.

        Args:
            data_dir (str):
                When `data_files` is None, it will load all json files in
                `data_dir` and its several tiers of sub-directories; when
                `data_files` is not None, only json files in `data_files` are
                loaded. When `cache_dir` is not None, and `cache_dir` points
                to a  cached dataset, the setting of `data_dir` become invalid.
                Default is None.
            data_files (List[str]):
                The paths of several json files. When `data_dir` is None, it
                will directly load files according to the addresses in
                `data_files`; when `data_dir` is not None, the file paths in
                `data_files` should be relative paths under `data_dir`. When
                `cache_dir` is not None, and `cache_dir` points to a cached
                dataset, the settings of `data_files` become invalid. Default
                is None.
            cache_dir (str):
                The cache path of the dataset. When the path points to an
                existing cached dataset, it will directly load the cached
                dataset, and the settings of `data_dir` and `data_files` will
                become invalid. When the path does not point to a cached
                dataset, the processed data will be cached to this path.
                Default is None.

        Raises:
            RuntimeError:
                When the dataset is not cached, `data_files` and `data_dir`
                cannot be None at the same time.
            TypeError:
                If `data_files` is not None, it should be a str or a list
                of str.

        Returns:
            datasets.Dataset or _PackDataset:
                If `pack_to_max_length` is True, the returned will be the
                processed(tokenized) dataset, using `datasets.Dataset` for
                easy caching and data packing.
                If `pack_to_max_length` is False, the returned will be the
                packed dataset.
        """
        if self.is_cached(cache_dir):
            print_log(
                f'{cache_dir} is a cached dataset that will be loaded '
                'directly; `data_files` and `data_dir` will become '
                'invalid.',
                logger='current')

            return self._load_cached_dataset(cache_dir=cache_dir)
        elif data_dir is not None:

            data_dir = Path(data_dir)
            if data_files is None:
                # TODO support other format
                data_files = [str(f) for f in data_dir.rglob('*.json')]
            elif isinstance(data_files, list):
                data_files = [str(data_dir / Path(f)) for f in data_files]
            elif isinstance(data_files, str):
                data_files = [str(data_dir / data_files)]
            else:
                raise TypeError('`data_files` should be a str or a list of '
                                f'str, not {type(data_files)}.')

            self.data_files = data_files
            return self._load_json_dataset(files=data_files)
        elif data_dir is None:
            if data_files is None:
                raise RuntimeError('When the dataset is not cached, '
                                   '`data_files` and `data_dir` cannot be '
                                   'None at the same time.')

            if isinstance(data_files, list):
                for file in data_files:
                    if not os.path.exists(file):
                        raise FileNotFoundError(f'{file} does not exist, '
                                                'please check `data_files`.')
            elif isinstance(data_files, str):
                if not os.path.exists(file):
                    raise FileNotFoundError(f'{file} does not exist, please '
                                            'check `data_files`.')
                data_files = [data_files]
            else:
                raise TypeError('`data_files` should be a str or a list of '
                                f'str, not {type(data_files)}.')

            self.data_files = data_files
            return self._load_json_dataset(files=data_files)
        else:
            raise NotImplementedError

    @master_only_load
    def _load_cached_dataset(self,
                             cache_dir: str) -> Union[Dataset, _PackDataset]:
        """Load the cached dataset."""

        dataset = load_from_disk(cache_dir)
        if self.pack_to_max_length:
            dataset = self._pack_dataset(dataset)
        return dataset

    @master_only_load
    def _load_json_dataset(self,
                           files: List[str]) -> Union[Dataset, _PackDataset]:
        """Load several json files and map them into a trainable format."""
        dataset = []

        if isinstance(self.sample_ratio, float):
            ratios = [self.sample_ratio] * len(files)
        else:
            ratios = self.sample_ratio

        for ratio, file in zip(ratios, files):
            shard_dataset = load(file)
            ori_samples = len(shard_dataset)
            target_samples = int(ratio * ori_samples)
            dataset.extend(random.choices(shard_dataset, k=target_samples))
            print_log(
                f'Loaded json data from {file}, '
                f'originally {ori_samples} samples, '
                f'random sampled to {target_samples} samples, '
                f'sample ratio {ratio}',
                logger='current')

        dataset = self.tokenize_dataset(dataset)

        dataset = self.filter_non_labels_data(dataset)

        self.analysis_tokens_labels(dataset)

        dataset = Dataset.from_list(dataset)

        if self.pack_to_max_length:
            dataset = self._pack_dataset(dataset)

        return dataset

    @abstractmethod
    def tokenize_dataset(self, dataset: List[dict]) -> List[dict]:
        """Tokenize the dataset and convert it into a trainable format.

        In the `tokenize_dataset` method, you need to define how to convert
        the raw data to the correct prompts (taking note of chat templates)
        and tokenize them; you need to define the label for each token, with
        the labels of the part that doesn't need to calculate loss set to -100.

        The labels don't need to be offset, it will be offset when the model
        calculates loss, meaning the label of each token should be itself or
        -100.


         Args:
             dataset (List[dict]):  The untokenized dataset.

         Note:
             The input must be a native Python list of dict, not
             `datasets.Dataset`, otherwise multithreaded data filtering will be
             slow.

         Returns:
             List[dict]:
                 Each dict in the list must contain three keys: `input_ids`,
                 `labels` and `num_tokens`.
                 `input_ids` and `labels` are lists of int, and they should
                 have equal lengths.
                 `num_tokens` is an integer, the length of `input_ids`.
        """

        def openai_to_training(item: dict) -> Dict:

            data = ChatMessages.from_dict(item)
            data = data.tokenize(self.tokenizer, self.chat_template)

            return data

        dataset = self.multi_thread_map(openai_to_training, dataset,
                                        'Tokenize Dataset')

        return dataset

    def _pack_dataset(self, dataset) -> _PackDataset:
        """Pack the processed(tokenized) dataset.

        Args:
            dataset (datasets.Dataset): The processed(tokenized) dataset.

        Returns:
            _PackDataset: Pack multiple data until reaching the max length.
        """
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

    def filter_non_labels_data(self, dataset: List[dict]) -> List[dict]:
        """Filter the data which all labels are ignore.

        Note:
            If all the labels for a data are ignore, it will result in a NAN
            loss value during training.

        Args:
            dataset (List[dict]):  The processed(tokenized) dataset.

        Note:
            The input must be a native Python list of dict, not
            `datasets.Dataset`, otherwise multithreaded data filtering will be
            slow.

        Returns:
            List[dict]: Filtered Dataset
        """

        def filter_fn(item):
            return any(label >= 0 for label in item['labels'])

        ori_samples = len(dataset)

        results = self.multi_thread_map(filter_fn, dataset, 'Filter Dataset')
        new_dataset = [x for x, passed in zip(dataset, results) if passed]

        new_samples = len(new_dataset)
        print_log(f'Before filter: {ori_samples} samples', logger='current')
        print_log(f'After filter: {new_samples} samples', logger='current')
        print_log(
            f'Filtered {ori_samples - new_samples} samples '
            '(all labels are ignore)',
            logger='current')
        return new_dataset

    def analysis_tokens_labels(self, dataset: List[dict]) -> List[str]:
        """Count the total number of tokens in the dataset, and the number of
        tokens for which loss needs to be calculated.

        Args:
            dataset (List[dict]):  The processed(tokenized) dataset.

        Note:
            The input must be a native Python list of dict, not
            `datasets.Dataset`, otherwise multithreaded counting will be slow.
        """

        def label_counter(item):
            return sum([1 for i in item['labels'] if i >= 0])

        def token_counter(item):
            return len(item['input_ids'])

        tokens = self.multi_thread_map(token_counter, dataset, 'Count Tokens')
        labels = self.multi_thread_map(label_counter, dataset, 'Count Labels')

        num_tokens = sum(tokens)
        num_labels = sum(labels)
        print_log(
            f'There are a total of {num_tokens} tokens, '
            f'of which {num_labels} tokens need loss calculation.',
            logger='current')

    def multi_thread_map(self, map_fn, dataset, desc):

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(map_fn, dataset),
                    desc=desc,
                    total=len(dataset)))

        return results

    @master_only
    def cache_dataset(self, cache_dir: str):
        """Cache the processed(tokenized) dataset."""

        if self.pack_to_max_length:
            # The packed dataset can't be cached, only the dataset before
            # #packing can be cached.
            # The cached dataset will be re-packed after loading.
            hf_dataset = self.dataset.dataset
        else:
            hf_dataset = self.dataset

        hf_dataset.save_to_disk(cache_dir)

        dset_conf = {
            'data_files': self.data_files,
            'max_length': self.max_length,
            'chat_template': self.chat_template.model_dump(),
            'pack_to_max_length': self.pack_to_max_length,
            'tokenizer': type(self.tokenizer).__name__,
        }

        dump(dset_conf, os.path.join(cache_dir, 'xtuner_dataset_conf.json'))

        print_log(
            f'The processed dataset is cached in the {cache_dir}.',
            logger='current')

    def is_cached(self, cache_dir: Optional[str]) -> bool:
        """Determine whether the path is a cached dataset path."""

        if cache_dir is None:
            return False

        if os.path.isdir(cache_dir):
            dset_config = os.path.join(cache_dir, 'xtuner_dataset_conf.json')
            if os.path.exists(dset_config):
                conf = load(dset_config)
                cache_tok = conf['tokenizer']
                cur_tok = type(self.tokenizer).__name__
                if cache_tok != cur_tok:
                    print_log(
                        f'The tokenizer({cache_tok}) used by the cached '
                        'dataset is different from the tokenizer'
                        f'({cur_tok}) you set, which may lead to '
                        'training errors.',
                        logger='current',
                        level=logging.WARNING)
                return True
            else:
                return False
        else:
            return False

    def __len__(self) -> int:
        """Get the length of the dataset.

        Note:
           If pack_to_max_length is True, the length might be much less than
           the original dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, List]:
        """Return the corresponding data according to the index.

        Note:
            If the developer changes the returned data format, the
            `collate_fn` of the dataloader also needs to be modified
            accordingly. XTuner defines `collate_fn` in the model, because the
            data format returned by `collate_fn` is heavily dependent on model
            training.

        Returns:
            Dict[str, List]:
                The returned should be a dict, and must include two keys:
                `input_ids` and `labels`. Both are lists of int, and they
                should have equal lengths.

                If `pack_to_max_length` is True, there should also be a key
                named `cumulative_len`, which records the cumulative length of
                each data being packed. For example, if three pieces of data
                are packed into one, with the respective lengths of 2, 4, and
                6, then the corresponding `cumulative_len` is [0, 2, 6, 12].

                The length of `cumulative_len` should be the number of packed
                data plus 1.
        """
        data = self.dataset[item]
        return data


if __name__ == '__main__':

    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'],
    )

    dataset = BaseTrainDataset(
        'internlm/internlm2-chat-1_8b',
        chat_template,
        sample_ratio=1,
        max_length=32 * 1024,
        data_dir='converted_alpaca',
        cache_dir='cached_alpaca',
        pack_to_max_length=True,
        num_proc=4)

    print(dataset[0])

    from mmengine.dataset import DefaultSampler
    from torch.utils.data import DataLoader

    from xtuner.model import TextFinetune
    loader = DataLoader(
        dataset,
        4,
        num_workers=0,
        collate_fn=TextFinetune.dataloader_collate_fn,
        sampler=DefaultSampler(dataset, shuffle=True))

    for data in tqdm(loader):
        continue
