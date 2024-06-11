import copy
import json
import os
from datetime import timedelta
from functools import partial
from multiprocessing import Process, Queue
from typing import Callable, Dict, List

import numpy as np
import torch.distributed as dist
import tqdm
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils.misc import get_object_from_string
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from xtuner.registry import BUILDER, MAP_FUNC
from .huggingface import build_origin_dataset


def _worker(
    tokenize_fun: Callable,
    data_queue: Queue,
    out_queue: Queue,
):
    while True:
        data_chunk = data_queue.get()

        if data_chunk is None:
            out_queue.put(None)
            break
        chunk_results = []
        for idx, data in data_chunk:
            chunk_results.append([idx, tokenize_fun(data)])
        out_queue.put(chunk_results)


def _chunk_data_to_queue(data_queue: Queue, data: List[Dict], chunk_size: int,
                         nproc):
    data_iter = iter(data)
    chunk_data = []
    while True:
        try:
            item = next(data_iter)
        except StopIteration:
            break
        chunk_data.append(item)
        if len(chunk_data) == chunk_size:
            data_queue.put(chunk_data)
            chunk_data = []
    if chunk_data:
        data_queue.put(chunk_data)

    for _ in range(nproc):
        data_queue.put(None)


def _multi_progress(tokenize_fun_p, dataset, nproc, task_num, chunksize,
                    description):
    processes = []
    data_queue = Queue()
    output_queue = Queue()
    bar = tqdm.tqdm(total=task_num, desc=description)
    # task_id = bar.add_task(total=task_num, description=description)
    dataset = enumerate(dataset)
    _chunk_data_to_queue(data_queue, dataset, chunksize, nproc)
    for _ in range(nproc):
        process = Process(
            target=_worker, args=(tokenize_fun_p, data_queue, output_queue))
        process.start()
        processes.append(process)

    results = []
    finished_process = 0
    while finished_process < nproc:
        chunk_results = output_queue.get()
        if chunk_results is None:
            finished_process += 1
            continue
        results.extend(chunk_results)
        bar.update(len(chunk_results))
        bar.refresh()
    results = map(lambda x: x[1], sorted(results, key=lambda x: x[0]))
    return results


def load_jsonl_dataset(data_files=None, data_dir=None, suffix=None):
    assert (data_files is not None) != (data_dir is not None)
    if data_dir is not None:
        data_files = os.listdir(data_dir)
        data_files = [os.path.join(data_dir, fn) for fn in data_files]
        if suffix is not None:
            data_files = [fp for fp in data_files if fp.endswith(suffix)]
    elif isinstance(data_files, str):
        data_files = [data_files]

    dataset_list = []
    for fp in data_files:
        with open(fp, encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        ds = HFDataset.from_list(data)
        dataset_list.append(ds)
    dataset = concatenate_datasets(dataset_list)
    return dataset


def tokenize(pair: str,
             tokenizer: AutoTokenizer,
             max_length: int,
             is_reward: bool = False,
             reward_token_id: int = -1):
    prompt = tokenizer.apply_chat_template(
        pair['prompt'], tokenize=False, add_generation_prompt=True)
    chosen = tokenizer.apply_chat_template(
        pair['prompt'] + pair['chosen'],
        tokenize=False,
        add_generation_prompt=False)
    rejected = tokenizer.apply_chat_template(
        pair['prompt'] + pair['rejected'],
        tokenize=False,
        add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)

    if len(chosen_ids) > max_length:
        chosen_ids = chosen_ids[:max_length]
    if len(rejected_ids) > max_length:
        rejected_ids = rejected_ids[:max_length]

    if is_reward:
        # reward label
        chosen_ids = chosen_ids + [reward_token_id]
        rejected_ids = rejected_ids + [reward_token_id]
        chosen_labels = [-100] * len(chosen_ids[:-1]) + [0]
        rejected_labels = [-100] * len(rejected_ids[:-1]) + [1]
    else:
        # dpo label
        prompt_len = min(len(prompt_ids), max_length)
        chosen_labels = [-100] * prompt_len + copy.deepcopy(
            chosen_ids[prompt_len:])
        rejected_labels = [-100] * prompt_len + copy.deepcopy(
            rejected_ids[prompt_len:])

    return {
        'chosen_ids': chosen_ids,
        'rejected_ids': rejected_ids,
        'chosen_labels': chosen_labels,
        'rejected_labels': rejected_labels,
    }


class PreferenceDataset(Dataset):

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: AutoTokenizer,
        max_length: int,
        is_dpo: bool = True,
        is_reward: bool = False,
        reward_token_id: int = -1,
        num_proc: int = 32,
    ) -> None:
        self.max_length = max_length
        assert is_dpo != is_reward, \
            'Only one of is_dpo and is_reward can be True'
        if is_reward:
            assert reward_token_id != -1, \
                'reward_token_id should be set if is_reward is True'

        self.is_dpo = is_dpo
        self.is_reward = is_reward
        self.reward_token_id = reward_token_id
        self.tokenized_pairs = []

        for tokenized_pair in _multi_progress(
                partial(
                    tokenize,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    is_reward=is_reward,
                    reward_token_id=reward_token_id),
                dataset,
                nproc=num_proc,
                task_num=len(dataset),
                chunksize=num_proc,
                description='Tokenizing dataset'):
            self.tokenized_pairs.append(tokenized_pair)

    def __len__(self):
        return len(self.tokenized_pairs)

    def __getitem__(self, idx):
        return self.tokenized_pairs[idx]


class PackedDatasetWrapper(Dataset):

    def __init__(self,
                 dataset,
                 max_packed_length=16384,
                 shuffle_before_pack=True) -> None:
        super().__init__()
        self.max_packed_length = max_packed_length
        self.lengths = []
        self.data = []

        indices = np.arange(len(dataset))
        if shuffle_before_pack:
            np.random.shuffle(indices)

        data_bin = []
        bin_seq_len = 0
        removed = 0
        for idx in indices:
            data = dataset[int(idx)]
            cur_len = len(data['chosen_ids']) + len(data['rejected_ids'])
            if cur_len > max_packed_length:
                print_log(
                    f'sequence length {cur_len} is '
                    f'larger than max_packed_length {max_packed_length}',
                    logger='current')
                removed += 1
                continue
            if (bin_seq_len +
                    cur_len) > max_packed_length and len(data_bin) > 0:
                self.data.append(data_bin)
                self.lengths.append(bin_seq_len)
                data_bin = []
                bin_seq_len = 0
            data_bin.append(data)
            bin_seq_len += cur_len

        if len(data_bin) > 0:
            self.data.append(data_bin)
            self.lengths.append(bin_seq_len)
        if removed > 0:
            print_log(
                f'removed {removed} samples because '
                f'of length larger than {max_packed_length}',
                logger='current')
        print_log(
            f'The batch numbers of dataset is changed '
            f'from {len(dataset)} to {len(self)} after'
            ' using var len attention.',
            logger='current')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pairs = self.data[index]
        input_ids, cu_seqlens, position_ids, labels = [], [0], [], []

        for pair in pairs:
            input_ids.extend(pair['chosen_ids'])
            input_ids.extend(pair['rejected_ids'])

            position_ids.extend(list(range(len(pair['chosen_ids']))))
            position_ids.extend(list(range(len(pair['rejected_ids']))))

            labels.extend(pair['chosen_labels'])
            labels.extend(pair['rejected_labels'])

            cu_seqlens.append(cu_seqlens[-1] + len(pair['chosen_ids']))
            cu_seqlens.append(cu_seqlens[-1] + len(pair['rejected_ids']))

        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'cumulative_len': cu_seqlens
        }


def unpack_seq(seq, cu_seqlens):
    """Unpack a packed sequence to a list of sequences with different
    lengths."""
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    subseqs = seq.split(seqlens)
    return subseqs


def broad_cast_dataset(dataset):
    xtuner_dataset_timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=60)))
    print_log(
        f'xtuner_dataset_timeout = {xtuner_dataset_timeout}', logger='current')
    using_dist = dist.is_available() and dist.is_initialized()
    if using_dist:
        # monitored barrier requires gloo process group to perform host-side sync.  # noqa
        group_gloo = dist.new_group(
            backend='gloo', timeout=xtuner_dataset_timeout)
    if not using_dist or dist.get_rank() == 0:
        objects = [dataset]
    else:
        objects = [None]
    if using_dist:
        dist.monitored_barrier(
            group=group_gloo, timeout=xtuner_dataset_timeout)
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


def map_dataset(dataset, dataset_map_fn, map_num_proc):
    if isinstance(dataset_map_fn, str):
        map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(
            dataset_map_fn)
        if map_fn_obj is not None:
            dataset_map_fn = map_fn_obj
        else:
            raise TypeError('dataset_map_fn must be a function or a '
                            "registered function's string in MAP_FUNC, "
                            f"but got a string of '{dataset_map_fn}'")

    dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)
    return dataset


def build_preference_dataset(
    dataset: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    dataset_map_fn: Callable = None,
    is_dpo: bool = True,
    is_reward: bool = False,
    reward_token_id: int = -1,
    num_proc: int = 32,
    use_varlen_attn: bool = False,
    max_packed_length: int = 16384,
    shuffle_before_pack: bool = True,
) -> Dataset:
    using_dist = dist.is_available() and dist.is_initialized()
    tokenized_ds = None
    if not using_dist or dist.get_rank() == 0:
        if isinstance(tokenizer, dict) or isinstance(
                tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            tokenizer = BUILDER.build(tokenizer)

        dataset = build_origin_dataset(dataset, split='train')
        if dataset_map_fn is not None:
            dataset = map_dataset(
                dataset, dataset_map_fn, map_num_proc=num_proc)

        tokenized_ds = PreferenceDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            is_dpo=is_dpo,
            is_reward=is_reward,
            reward_token_id=reward_token_id,
            num_proc=num_proc,
        )
        if use_varlen_attn:
            tokenized_ds = PackedDatasetWrapper(
                dataset=tokenized_ds,
                max_packed_length=max_packed_length,
                shuffle_before_pack=shuffle_before_pack,
            )
    tokenized_ds = broad_cast_dataset(tokenized_ds)
    return tokenized_ds


def intel_orca_dpo_map_fn(example):
    prompt = [{
        'role': 'system',
        'content': example['system']
    }, {
        'role': 'user',
        'content': example['question']
    }]
    chosen = [{'role': 'assistant', 'content': example['chosen']}]
    rejected = [{'role': 'assistant', 'content': example['rejected']}]
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}


def orpo_dpo_mix_40k_map_fn(example):
    assert len(example['chosen']) == len(example['rejected'])
    prompt = example['chosen'][:-1]
    chosen = example['chosen'][-1:]
    rejected = example['rejected'][-1:]
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}
