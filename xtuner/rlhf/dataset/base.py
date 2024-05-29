"""Basic datasets implement."""

import gzip
import json
import random
from contextlib import contextmanager

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, IterableDataset, Subset


@contextmanager
def open_file(filename):
    """Construct a file handler.

    The handler can read a normal file or a file compressed by `gzip`.
    """
    if filename.endswith('.gz'):
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, encoding='utf-8')
    yield fp
    fp.close()


class InfiniteDataset(IterableDataset):
    """Load infinite data from original dataset with shuffle."""

    def __init__(self, dataset, rng=None):
        self.data = list(iter(dataset))
        self.indices = list(range(len(self.data)))
        if rng is None:
            rng = random.Random()
        self.rng = rng

    def __iter__(self):
        while True:
            self.rng.shuffle(self.indices)
            for i in self.indices:
                yield self.data[i]


class FileDataset(IterableDataset):
    """Single json file dataset."""

    def __init__(self,
                 filename,
                 tokenizer,
                 sys_meta='default',
                 rm_meta='default'):
        self._filename = filename
        self.tokenizer = tokenizer
        self.data_list = []
        self.sys_meta = sys_meta
        self.rm_meta = rm_meta
        with open_file(self._filename) as fin:
            for lineno, line in enumerate(fin):
                data = json.loads(line)
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        try:
            self.tokenizer.apply_chat_template(data, tokenize=True)
            return {
                'data': data,
                'sys_meta': self.sys_meta,
                'rm_meta': self.rm_meta
            }
        except Exception:
            print(f'[data tokenize check] skip dirty data: {data}')
            return None

    def __iter__(self):
        with open_file(self._filename) as fin:
            for lineno, line in enumerate(fin):
                data = json.loads(line)
                try:
                    self.tokenizer.apply_chat_template(data, tokenize=True)
                except Exception:
                    print(f'[data tokenize check] skip dirty data: {data}')
                    continue
                if data is None:
                    continue
                yield {
                    'data': data,
                    'sys_meta': self.sys_meta,
                    'rm_meta': self.rm_meta
                }


class OpensourceDataset(IterableDataset):
    """Opensource dataset."""

    def __init__(self,
                 filename,
                 tokenizer,
                 sys_meta='default',
                 rm_meta='default'):
        self._filename = filename
        self.tokenizer = tokenizer
        self.sys_meta = sys_meta
        self.rm_meta = rm_meta
        assert 'Anthropic' in filename or 'openai' in filename, '[Coming soon] currently only support loading Anthropic and openai opensource datasets...'  # noqa: E501
        if 'Anthropic' in filename:
            from .open_datasets.Anthropic_hh_rlhf import AnthropicHhrlhf
            self.data_list = AnthropicHhrlhf(path=filename)
        elif 'openai' in filename:
            pass
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        try:
            self.tokenizer.apply_chat_template(data, tokenize=True)
            return {
                'data': data,
                'sys_meta': self.sys_meta,
                'rm_meta': self.rm_meta
            }
        except Exception:
            print(f'[data tokenize check] skip dirty data: {data}')
            return None

    def __iter__(self):
        for lineno, data in enumerate(self.data_list):
            if data is None:
                continue
            try:
                self.tokenizer.apply_chat_template(data, tokenize=True)
            except Exception:
                print(f'[data tokenize check] skip dirty data: {data}')
                continue
            yield {
                'data': data,
                'sys_meta': self.sys_meta,
                'rm_meta': self.rm_meta
            }


class MultiSourceDatset(IterableDataset):
    """Multiple source dataset."""

    def __init__(self,
                 task_groups,
                 sub_dataset_type='file',
                 tokenizer=None,
                 random_seed=1024,
                 ratio_within_datas=True):
        self._task_group = []
        for _task in task_groups:
            file_path, extra_info = _task.split('::')[0], _task.split('::')[1]
            prob = float(extra_info.split('[')[0])
            sys_meta = 'default'
            rm_meta = 'default'
            if '[META]:' in extra_info:
                sys_meta = extra_info.split('[META]:')[-1].split('[')[0]
            if '[REWARD_META]:' in extra_info:
                rm_meta = extra_info.split('[REWARD_META]:')[-1].split('[')[0]
            if prob > 0:
                self._task_group.append({
                    'prob': prob,
                    'filepath': file_path,
                    'sys_meta': sys_meta,
                    'rm_meta': rm_meta
                })
                print(
                    f'[DataLoader] Load {_task} with prob:{prob}, sys_meta type: {sys_meta}, reward meta: {rm_meta}'  # noqa: E501
                )
            else:
                print(
                    f'[DataLoader] Warning skip file, prob of {file_path} is {prob} ...'  # noqa: E501
                )
        assert len(self._task_group) > 0, 'No data to be trained'
        if sub_dataset_type == 'file':
            for task in self._task_group:
                filepath = task['filepath']
                if '.json' in filepath:
                    task['dataset'] = FileDataset(filepath, tokenizer,
                                                  task['sys_meta'],
                                                  task['rm_meta'])
                else:
                    # loading opensource datasets
                    print(f'Try loading {filepath} from huggingface ...')
                    task['dataset'] = OpensourceDataset(
                        filepath, tokenizer, task['sys_meta'], task['rm_meta'])
        else:
            raise NotImplementedError('Cannot support filelist now.')
        self.random_seed = random_seed
        self.ratio_within_datas = ratio_within_datas

        if self.ratio_within_datas:
            sum_prob = sum([task['prob'] for task in self._task_group])
            for task in self._task_group:
                task['prob'] = task['prob'] / sum_prob
        else:
            datasets = []
            for i, task in enumerate(self._task_group):
                task['dataset'] = self._get_subset_by_ratio(
                    task['dataset'], task['prob'], random_seed)
                datasets.append(task['dataset'])

            self.all_dataset = ConcatDataset(datasets)
            self.iter_all_dataset = iter(self.all_dataset)

    def _get_subset_by_ratio(self, dataset: Dataset, ratio: float, seed: int):
        np_random = np.random.RandomState(seed)
        indices = np.arange(len(dataset))
        np_random.shuffle(indices)
        subset_indices = indices[:int(len(dataset) * ratio)]
        subset_indices = list(subset_indices)
        return Subset(dataset, subset_indices)

    def __iter__(self):
        """sample data one task by probs."""
        if self.ratio_within_datas:
            rng = random.Random(self.random_seed)
            probs = [task['prob'] for task in self._task_group]
            # Initialize task iterator
            for task in self._task_group:
                task['iterator'] = iter(task['dataset'])
            while True:
                task = rng.choices(self._task_group, weights=probs)[0]
                try:
                    yield from task['iterator']
                except StopIteration:
                    task['iterator'] = iter(task['dataset'])
                    yield from task['iterator']
        else:
            yield next(self.iter_all_dataset)
