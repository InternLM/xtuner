"""Basic datasets implement."""

import gzip
import json
import random
from contextlib import contextmanager

import numpy as np
from loguru import logger
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


class IterDataset(IterableDataset):
    """Single json file dataset."""

    def __init__(self,
                 filename=None,
                 data_list=None,
                 tokenizer=None,
                 sys_prompt='default',
                 rm_prompt='default'):
        assert filename is not None or data_list is not None
        self._filename = filename
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.sys_prompt = sys_prompt
        self.rm_prompt = rm_prompt

    def __iter__(self):
        if self.data_list is not None:
            for lineno, data in enumerate(self.data_list):
                try:
                    self.tokenizer.apply_chat_template(data, tokenize=True)
                except Exception:
                    logger.info(
                        f'[data tokenize check] skip dirty data: {data}')
                    continue
                if data is None:
                    continue
                yield dict(
                    data=data,
                    sys_prompt=self.sys_prompt,
                    rm_prompt=self.rm_prompt)
        else:
            with open_file(self._filename) as fin:
                for lineno, line in enumerate(fin):
                    data = json.loads(line)
                    try:
                        self.tokenizer.apply_chat_template(data, tokenize=True)
                    except Exception:
                        logger.info(
                            f'[data tokenize check] skip dirty data: {data}')
                        continue
                    if data is None:
                        continue
                    yield dict(
                        data=data,
                        sys_prompt=self.sys_prompt,
                        rm_prompt=self.rm_prompt)


class MultiSourceInBatchDatset(IterableDataset):
    """Multiple source dataset."""

    def __init__(self, task_groups, tokenizer=None, random_seed=1024):
        self._task_group = []
        for _task in task_groups:
            file_path, extra_info = _task.split('::')[0], _task.split('::')[1]
            prob = float(extra_info.split('[')[0])
            sys_prompt = 'default'
            rm_prompt = 'default'
            if '[SYS_PROMPT]:' in extra_info:
                sys_prompt = extra_info.split('[SYS_PROMPT]:')[-1].split(
                    '[')[0]
            if '[RM_PROMPT]:' in extra_info:
                rm_prompt = extra_info.split('[RM_PROMPT]:')[-1].split('[')[0]
            if prob > 0:
                self._task_group.append(
                    dict(
                        prob=prob,
                        filepath=file_path,
                        sys_prompt=sys_prompt,
                        rm_prompt=rm_prompt))
                logger.info(f'[DataLoader] Load {_task} with prob:{prob}, '
                            f'sys_prompt type: {sys_prompt}, '
                            f'reward prompt type: {rm_prompt}')
            else:
                logger.warning('[DataLoader] skip file, '
                               f'prob of {file_path} is {prob} ...')
        assert len(self._task_group) > 0, 'No data to be trained'

        for task in self._task_group:
            filepath = task['filepath']
            if '[HF]' in filepath:
                from xtuner.rlhf.dataset.utils.from_hf import load_from_hf

                # loading & convert & save opensource datasets
                hf_dir = filepath.split('[HF]')[-1]
                logger.info(f'Loading {hf_dir} from huggingface ...')
                dataset = load_from_hf(hf_dir, tokenizer=tokenizer)
                task['dataset'] = IterDataset(
                    data_list=dataset['conversation'],
                    tokenizer=tokenizer,
                    sys_prompt=task['sys_prompt'],
                    rm_prompt=task['rm_prompt'])

            else:
                task['dataset'] = IterDataset(
                    filename=filepath,
                    tokenizer=tokenizer,
                    sys_prompt=task['sys_prompt'],
                    rm_prompt=task['rm_prompt'])

        sum_prob = sum([task['prob'] for task in self._task_group])
        for task in self._task_group:
            task['prob'] = task['prob'] / sum_prob

        self.random_seed = random_seed

    def __iter__(self):
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


class JsonDataset(Dataset):
    """Single json file dataset."""

    def __init__(self,
                 filename=None,
                 data_list=None,
                 tokenizer=None,
                 sys_prompt='default',
                 rm_prompt='default'):
        assert filename is not None or data_list is not None
        self.tokenizer = tokenizer
        self.sys_prompt = sys_prompt
        self.rm_prompt = rm_prompt

        if filename is not None:
            self.data_list = []
            with open_file(filename) as fin:
                for lineno, line in enumerate(fin):
                    data = json.loads(line)
                    self.data_list.append(data)
        else:
            self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        try:
            self.tokenizer.apply_chat_template(data, tokenize=True)
            return {
                'data': data,
                'sys_prompt': self.sys_prompt,
                'rm_prompt': self.rm_prompt
            }
        except Exception:
            logger.info(f'[data tokenize check] skip dirty data: {data}')
            return None


class MultiSourceInDataDatset(Dataset):
    """Multi source dataset.

    Args:
        task_groups: list of data path.
            e.g. ['PATH_TO_XTUNER/examples/rlhf/demo_datas/prompt_data.json::0.9[SYS_PROMPT]:summarization',  # noqa: E501
                  'PATH_TO_XTUNER/examples/rlhf/demo_datas/pretrain_data.json::0.1',
                  '[HF]Anthropic/hh-rlhf/helpful-base::0.5[RM_PROMPT]:default',
                  '[HF]HuggingFaceH4/summarize_from_feedback::0.5'
                  ]
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding. This argument should not be None. Default to None.
        random_seed:
    """

    def __init__(self, task_groups, tokenizer=None, random_seed=1024):
        self._task_group = []
        for _task in task_groups:
            file_path, extra_info = _task.split('::')[0], _task.split('::')[1]
            prob = float(extra_info.split('[')[0])
            sys_prompt = 'default'
            rm_prompt = 'default'
            if '[SYS_PROMPT]:' in extra_info:
                sys_prompt = extra_info.split('[SYS_PROMPT]:')[-1].split(
                    '[')[0]
            if '[RM_PROMPT]:' in extra_info:
                rm_prompt = extra_info.split('[RM_PROMPT]:')[-1].split('[')[0]
            if prob > 0:
                self._task_group.append(
                    dict(
                        prob=prob,
                        filepath=file_path,
                        sys_prompt=sys_prompt,
                        rm_prompt=rm_prompt))
                logger.info(
                    f'[DataLoader] Load {_task} with prob:{prob}, '
                    f'sys_prompt type: {sys_prompt}, reward meta: {rm_prompt}')
            else:
                logger.warning('[DataLoader] skip file, '
                               f'prob of {file_path} is {prob} ...')
        assert len(self._task_group) > 0, 'No data to be trained'

        datasets = []
        for task in self._task_group:
            filepath = task['filepath']

            if '[HF]' in filepath:
                from xtuner.rlhf.dataset.utils.from_hf import load_from_hf

                # loading & convert & save opensource datasets
                hf_dir = filepath.split('[HF]')[-1]
                logger.info(f'Loading {hf_dir} with huggingface format ...')
                dataset = load_from_hf(hf_dir, tokenizer=tokenizer)
                task['dataset'] = JsonDataset(
                    data_list=dataset['conversation'],
                    tokenizer=tokenizer,
                    sys_prompt=task['sys_prompt'],
                    rm_prompt=task['rm_prompt'])
            else:
                task['dataset'] = JsonDataset(
                    filename=filepath,
                    tokenizer=tokenizer,
                    sys_prompt=task['sys_prompt'],
                    rm_prompt=task['rm_prompt'])
            task['dataset'] = self._get_subset_by_ratio(
                task['dataset'], task['prob'], random_seed)
            datasets.append(task['dataset'])

        self.all_dataset = ConcatDataset(datasets)
        self.iter_all_dataset = iter(self.all_dataset)

        self.random_seed = random_seed

    def _get_subset_by_ratio(self, dataset: Dataset, ratio: float, seed: int):
        np_random = np.random.RandomState(seed)
        indices = np.arange(len(dataset))
        np_random.shuffle(indices)
        subset_indices = indices[:int(len(dataset) * ratio)]
        subset_indices = list(subset_indices)
        return Subset(dataset, subset_indices)

    def __iter__(self):
        yield next(self.iter_all_dataset)
