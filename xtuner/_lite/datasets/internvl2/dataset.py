import os
import torch.distributed as dist
from mmengine.utils import mkdir_or_exist
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import numpy as np
import json
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import copy
import random

from ..json import calculate_json_sha256
from ..jsonl import calculate_jsonl_sha256
from ..pack import SoftPackDataset

from xtuner._lite import get_logger
from xtuner._lite.parallel import get_dp_mesh, VLMLengthGroupedSampler, ParallelSampler

logger = get_logger()


def _load_json_or_jsonl(json_path):
    if json_path.endswith('.json'):
        with open(json_path) as f:
            data = json.load(f)
    elif json_path.endswith('.jsonl'):
        with open(json_path) as f:
            data = f.readlines()
    else:
        raise ValueError(f'Unsupported file format: {json_path}, '
                         f'only support .json and .jsonl.')
    return data


class BaseOrigDataset(Dataset):
    def __init__(self,
                 data_name,
                 data,
                 chat_template,
                 tokenizer,
                 max_length,
                 image_token_str='<image>',
                 group_by_length=False,
                 pack_data=False,
                 pack_data_cache_dir=None,
                 random_sample=False):
        self.data_name = data_name
        self.max_length = max_length
        self.group_by_length = group_by_length
        self.pack_data = pack_data
        self.pack_data_cache_dir = pack_data_cache_dir
        self.chat_template = chat_template
        self.image_token_str = image_token_str
        self.tokenizer = tokenizer
        self.tokenizer_workers = int(os.environ.get('XTUNER_TOKENIZE_WORKERS', 8))

        try:
            self.root = data['media_root']
        except KeyError:
            self.root = data.get('root', '')
        logger.info(f"{dist.get_rank()} ======= Start to process dataset: {os.path.basename(data['annotation'])}")

        self.annotation = data['annotation']
        self._is_jsonl = self.annotation.endswith('.jsonl')
        self.raw_data = _load_json_or_jsonl(self.annotation)

        # -------------------pack---------------------------------------
        self.num_tokens = None
        self.pack_data_cache_dir = pack_data_cache_dir
        if pack_data:
            assert pack_data_cache_dir is not None, 'pack_data_cache_dir must be provided when pack_data is True'
            self.num_tokens = self.calc_packing_info()
            assert len(self.num_tokens) == len(
                self.raw_data), f'===={len(self.num_tokens)} neq {len(self.raw_data)}===='

        repeat_time = data.get('repeat_time', 1)
        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            if random_sample:
                num_samples = int(len(self.raw_data) * repeat_time)
                sampled = random.sample([i for i in range(len(self.raw_data))], num_samples)
                self.raw_data = [self.raw_data[index] for index in sampled]
                if pack_data:
                    self.num_tokens = self.num_tokens[sampled]
            else:
                num_samples = int(len(self.raw_data) * repeat_time)
                self.raw_data = self.raw_data[:num_samples]
                if pack_data:
                    self.num_tokens = self.num_tokens[:num_samples]

        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            # Repeat the list if repeat_time is greater than 1
            self.raw_data = self.raw_data * repeat_time
            if pack_data:
                self.num_tokens = np.tile(self.num_tokens, repeat_time)

        if pack_data:
            assert len(self.num_tokens) == len(self.raw_data), f' {len(self.num_tokens)} neq {len(self.raw_data)}'

        self.group_length = []
        if self.group_by_length and not pack_data:
            self.group_length = self.calc_group_len()

    def __len__(self):
        return len(self.raw_data)

    def calc_group_len(self):
        raise NotImplementedError

    def calc_packing_info(self):
        if os.path.exists(self.pack_data_cache_dir):
            assert os.path.isdir(self.pack_data_cache_dir)
        else:
            mkdir_or_exist(self.pack_data_cache_dir)

        # TODO: more rubost way to calculate the hash
        if self._is_jsonl:
            file_hash = calculate_jsonl_sha256(self.annotation)
        else:
            file_hash = calculate_json_sha256(self.annotation)
        file_cache_dir = os.path.join(self.pack_data_cache_dir, file_hash)
        if not os.path.exists(file_cache_dir):
            mkdir_or_exist(file_cache_dir)

        if 'num_tokens.npy' in os.listdir(file_cache_dir):
            _cached_file = os.path.join(file_cache_dir, 'num_tokens.npy')
            num_tokens = np.load(_cached_file)
            logger.info(f"Load num_tokens from cache: {os.path.basename(self.annotation)}")
        else:
            logger.info(f"Start calculating the cache of num_tokens: {os.path.basename(self.annotation)}")
            num_tokens = self.count_tokens_for_pack(file_cache_dir)
        return num_tokens

    def count_tokens_for_pack(self, cache_dir=None):
        num_samples = len(self.raw_data)

        if dist.is_available():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        dataset_shard = self.raw_data[start:end]

        desc = f'[Rank {rank}] {os.path.basename(self.annotation)}'
        with ProcessPoolExecutor(max_workers=self.tokenizer_workers) as executor:
            tokenized = list(
                tqdm(
                    executor.map(self.pre_tokenize_fn_for_pack, dataset_shard,
                                 chunksize=min(max(1, len(dataset_shard) // self.tokenizer_workers), 500)),
                    desc=desc,
                    total=len(dataset_shard)))

        _num_tokens = [data['num_tokens'] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_available():
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens)
            num_tokens = np.concatenate(num_tokens, axis=0)
        else:
            num_tokens = _num_tokens

        if rank == 0 and cache_dir:
            save_path = os.path.join(cache_dir, 'num_tokens.npy')
            np.save(save_path, num_tokens)

        return num_tokens

    def pre_tokenize_fn_for_pack(self, data):
        raise NotImplementedError

    def process_text(self, conversations, media_type='image', image_grids=None):
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]

        assert len(conversations) % 2 == 0, f'Invalid conversation length: {len(conversations)}'

        input_ = ''
        out_conversation = []
        for msg in conversations:
            if msg['from'] == 'human':
                input_ += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input_,
                    'output': msg['value'].strip()
                })
                input_ = ''
            else:
                raise NotImplementedError(f'Unsupported message type: {msg}')

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input_ = single_turn_conversation.get('input', '')
            if input_ is None:
                input_ = ''
            input_ = self.chat_template['user'].format(user=input_)

            if i == 0:
                input_ = self._process_media_format_first_round(input_, media_type, image_grids)
                # TODO: support system prompt
                # input_ = self.chat_template['system'] + input_
                input_encode = self.tokenizer.encode(input_, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(input_, add_special_tokens=False)

            input_ids += input_encode
            labels += [-100] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            output_encode = self.chat_template['assistant'].format(assistant=output_text)
            output_encode = self.tokenizer.encode(output_encode, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            logger.info(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}')
        return {'input_ids': input_ids, 'labels': labels}

    def _process_media_format_first_round(self, input_, media_type, image_grids):
        raise NotImplementedError

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length


def build_dataset(args, datasets):
    assert len(datasets) > 0, 'No dataset found.'
    if args.dset_pack:
        train_dataset = SoftPackDataset(datasets,
                                        target=args.pack_max_length,
                                        blend=args.concat_before_pack)
    else:
        train_dataset = ConcatDataset(datasets)
        if dist.get_rank() == 0:
            logger.info(f'[Dataset] (Original) {len(train_dataset)} samples.')
    return train_dataset


def build_train_dataloader(args, train_dataset, collate_fn):
    dp_mesh = get_dp_mesh()
    if args.group_by_length:
        if args.dset_pack:
            length_property = 'longest'
        else:
            length_property = 'length'
        sampler = VLMLengthGroupedSampler(train_dataset, dp_mesh,
                                          args.global_batch_size,
                                          seed=args.seed,
                                          length_property=length_property)
    elif args.group_by_modality_length:
        if args.dset_pack:
            raise NotImplementedError
        else:
            sampler = VLMLengthGroupedSampler(train_dataset, dp_mesh,
                                              args.global_batch_size,
                                              seed=args.seed,
                                              length_property='modality_length')
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, seed=args.seed, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0)

    if dist.get_rank() == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')

    dist.barrier()
    return train_dataloader
