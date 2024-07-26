import bisect
import itertools
import os
import random

import torch
from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.nn.utils.rnn import pad_sequence

from xtuner._lite import get_logger
from xtuner._lite.chat import ChatMessages
from xtuner._lite.parallel import get_sp_world_size, pad_for_sequence_parallel
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .cache import CacheDataset
from .format import OPENAI_FORMAT_MAP

logger = get_logger()


def sort_and_return_indices(lst):
    return [i[0] for i in sorted(enumerate(lst), key=lambda x: x[1])]


class TextTokenizeFunction():

    def __init__(self, tokenizer, chat_template, raw_format='openai'):

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.raw_format = raw_format

    def __call__(self, item):

        formatter = OPENAI_FORMAT_MAP[self.raw_format]
        msg = ChatMessages.from_dict(formatter(item))
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)
        return tokenized


class TextTokenizedDataset(CacheDataset):

    def __init__(self, dataset, max_length):
        super().__init__()

        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)

        self.dataset = dataset
        self.num_samples = len(dataset)
        self.total_tokens = sum(dataset['num_tokens'])
        self.max_length = max_length

        self._cached = False
        self._cached_dir = None

    @property
    def cached(self):
        return self._cached

    @property
    def cached_dir(self):
        return self._cached_dir

    def cache(self, cache_dir):
        dset_dir = os.path.join(cache_dir, 'dataset')

        if len(self.dataset.cache_files) == 0 and dist.is_available(
        ) and dist.get_rank() == 0:
            self.dataset.save_to_disk(dset_dir)

        self._cached_dir = cache_dir
        self.dataset = None
        self._cached = True

    def load_cache(self):
        assert self.cached
        dset_dir = os.path.join(self.cached_dir, 'dataset')
        self.dataset = load_from_disk(dset_dir)

    @classmethod
    def from_cache(cls, cache_dir, max_length):
        dataset = load_from_disk(os.path.join(cache_dir, 'dataset'))
        ret = cls(dataset, max_length)
        ret.cache(cache_dir)
        return ret

    def _free(self):
        self.dataset = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """

        if self.cached:
            self.load_cache()

        data = {
            'input_ids': self.dataset[item]['input_ids'],
            'labels': self.dataset[item]['labels'],
            'num_tokens': [self.dataset[item]['num_tokens']]
        }

        if self.cached:
            self._free()

        return data


class TextOnlineTokenizeDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenize_fn):
        super().__init__()

        self.dataset = dataset
        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        raw_data = self.dataset[item]
        tokenized_data = self.tokenize_fn(raw_data)

        data = {
            'input_ids': tokenized_data['input_ids'],
            'labels': tokenized_data['labels'],
            'num_tokens': [tokenized_data['num_tokens']]
        }

        return data


class SoftPackerForText(CacheDataset):

    def __init__(self, dataset, max_length=2048, pack_info=None, seed=None):
        super().__init__()

        self.max_length = max_length

        if isinstance(dataset, list):
            self.dataset = Dataset.from_list(dataset)
        else:
            self.dataset = dataset

        if pack_info is None:
            pack_info = self.get_pack_info(self.dataset, max_length, seed)
        self.pack_info = pack_info

        # The number of data items after packing
        self.num_packed_samples = len(self.pack_info)
        self.num_samples = len(self.dataset)
        self.total_tokens = sum(self.dataset['num_tokens'])

        self._cached = False
        self._cached_dir = None

    @property
    def max_length_per_pack(self):
        if self.cached:
            pack_info = load_from_disk(self._cached_pack_info)
        else:
            pack_info = self.pack_info
        return pack_info['max_length']

    @property
    def cached(self):
        return self._cached

    @property
    def cached_dir(self):
        return self._cached_dir

    def cache(self, cache_dir):
        dset_dir = os.path.join(cache_dir, 'dataset')
        pack_info_dir = os.path.join(cache_dir,
                                     f'pack-info-soft-{self.max_length}')

        if len(self.dataset.cache_files) == 0:
            self.dataset.save_to_disk(dset_dir)
        if len(self.pack_info.cache_files) == 0:
            self.pack_info.save_to_disk(pack_info_dir)

        self._cached = True
        self._cached_dir = cache_dir
        self._cached_dset = dset_dir
        self._cached_pack_info = pack_info_dir

        self._free()

    def load_cache(self):
        assert self._cached
        dset_dir = os.path.join(self.cached_dir, 'dataset')
        pack_info_dir = os.path.join(self.cached_dir,
                                     f'pack-info-soft-{self.max_length}')
        self.dataset = load_from_disk(dset_dir)
        self.pack_info = load_from_disk(pack_info_dir)

    def _free(self):
        self.dataset = None
        self.pack_info = None

    def __len__(self):
        return self.num_packed_samples

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        if self._cached:
            self.load_cache()

        dataset = self.dataset
        pack_info = self.pack_info

        packed_items = pack_info[item]['indices']
        assert len(packed_items) > 0

        input_ids = []
        labels = []
        num_tokens = []
        for i in packed_items:
            input_ids.extend(dataset[i]['input_ids'])
            labels.extend(dataset[i]['labels'])

            _num_tokens = dataset[i]['num_tokens']
            num_tokens.append(_num_tokens)

        if len(input_ids) < self.max_length:
            num_pad_tokens = self.max_length - len(input_ids)
            input_ids.extend([DEFAULT_PAD_TOKEN_INDEX] * num_pad_tokens)
            labels.extend([IGNORE_INDEX] * num_pad_tokens)
            num_tokens.append(num_pad_tokens)
        else:
            num_tokens.append(0)

        packed = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': num_tokens,
        }

        if len(input_ids) != len(labels):
            logger.error(f'[packed_items] {packed_items}')
            logger.error(f'[input_ids] {input_ids}')
            logger.error(f'[labels] {labels}')
            logger.error(f'[num_tokens] {num_tokens}')
            raise RuntimeError('The lengths of input_ids and labels must be '
                               f'equal, but  found {len(input_ids)} and '
                               f'{len(labels)}.')

        if self.cached:
            self._free()

        return packed

    @classmethod
    def get_pack_info(cls, dataset, max_length, seed=None):

        _ori_lens = dataset['num_tokens']
        inds = [i for i in range(len(dataset))]
        if seed is not None:
            random.seed(seed)
        random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        max_length_one_pack = 0

        pack_infos = []
        for shfl_i in inds:
            if _ori_lens[shfl_i] + sum(length_buffer) <= max_length:
                item_buffer.append(shfl_i)
                length_buffer.append(_ori_lens[shfl_i])
                max_length_one_pack = max(max_length_one_pack,
                                          _ori_lens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        'indices': item_buffer,
                        'max_length': max_length_one_pack
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [_ori_lens[shfl_i]]
                max_length_one_pack = _ori_lens[shfl_i]

        if len(item_buffer) > 0:
            info = {'indices': item_buffer, 'max_length': max_length_one_pack}
            pack_infos.append(info)

        pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    @classmethod
    def from_cache(cls, cache_dir, max_length, seed=None):

        dataset = load_from_disk(os.path.join(cache_dir, 'dataset'))

        pack_info_dir = os.path.join(cache_dir, f'pack-info-soft-{max_length}')
        if os.path.exists(pack_info_dir):
            pack_info = load_from_disk(pack_info_dir)
        else:
            pack_info = cls.get_pack_info(dataset, max_length, seed)

        ret = cls(dataset, max_length, pack_info)
        ret.cache(cache_dir)

        return ret


class HardPackerForText(SoftPackerForText):
    """The new dataset obtained by concatenating multiple raw data.

    Args:
        dataset (datasets.Dataset): The tokenized dataset.
        max_length (int): The length of each data after concatenation.
        use_varlen_attn (bool): Determines whether to calculate attention
            based on the seq_len dimension or the actual length of the
            sequence.

    Note:
        The original dataset's type must be `datasets.Dataset`, others will be
        very slow.

    Note:
        The data in the original dataset must have the `num_tokens` key,
        recording the number of tokens for each piece of data.
    """

    def __init__(self, dataset, max_length=2048, pack_info=None):
        super().__init__(dataset, max_length, pack_info)

        self.num_packed_samples = self.total_tokens // max_length

    @classmethod
    def _cal_max_length(cls, begin, end, shfl_item_rngs_left,
                        shfl_item_rngs_right):
        left = bisect.bisect(shfl_item_rngs_right, begin)
        right = bisect.bisect(shfl_item_rngs_left, end)
        max_length = 0
        for i in range(left, right):
            item_begin = shfl_item_rngs_left[i]
            item_end = shfl_item_rngs_right[i]
            inner_l = max(begin, item_begin) - item_begin
            inner_r = min(end, item_end) - item_begin
            trunc_size = inner_r - inner_l
            max_length = max(max_length, trunc_size)
        return max_length

    @classmethod
    def get_pack_info(cls, dataset, max_length, seed=None):

        _ori_lens = dataset['num_tokens']

        # The number of data items after packing
        num_packed_samples = sum(_ori_lens) // max_length

        # Shuffle the order of the original dataset
        # The packing will proceed according to the order after shuffle.
        # Assume the following conditions hold:
        #   (1) shfl_inds = [3, 1, 2, 0]
        #   (2) self._ori_lens[3] + self._ori_lens[1] = max_length
        #   (3) self._ori_lens[2] + self._ori_lens[0] = max_length
        # Ultimately, dataset[3] and dataset[1] will be combined into a new
        # data, and dataset[2] and dataset[0] will be combined into a new data.
        inds = [i for i in range(len(dataset))]
        if seed is not None:
            random.seed(seed)
        random.shuffle(inds)
        shfl_inds = inds

        # shuffled cumulative lengths
        shfl_lens = [_ori_lens[i] for i in shfl_inds]
        shfl_acc_lens = list(itertools.accumulate(shfl_lens))

        shfl_item_rngs_left = [0] + shfl_acc_lens[:-1]
        shfl_item_rngs_right = shfl_acc_lens

        max_length_per_pack = []
        for i in range(num_packed_samples):
            begin = i * max_length
            end = (i + 1) * max_length
            max_length_per_pack.append(
                cls._cal_max_length(begin, end, shfl_item_rngs_left,
                                    shfl_item_rngs_right))

        return {
            'ranges_left': shfl_item_rngs_left,
            'ranges_right': shfl_item_rngs_right,
            'num_packed_samples': num_packed_samples,
            'indices': shfl_inds,
            'max_length_per_pack': max_length_per_pack
        }

    def _pack_ids_and_labels_in_range(self, begin: int, end: int):
        """Packs ids and labels in a given range using bisection method.

        Args:
            begin: Index indicating the beginning of the range.
            end: Index indicating the end of the range.

        Returns:
            A tuple containing packed ids, labels, and cumulative lengths.
        """

        # Use binary search to find dataset positions that fall within begin
        # and end range
        left = bisect.bisect(self._shfl_item_rngs_right, begin)
        right = bisect.bisect(self._shfl_item_rngs_left, end)

        trunc_input_ids = []
        trunc_labels = []
        trunc_sizes = []

        for i in range(left, right):

            # Determine the real range we will cut in current original item
            item_begin = self._shfl_item_rngs_left[i]
            item_end = self._shfl_item_rngs_right[i]

            # Calculate exact positions within current dataset item
            inner_l = max(begin, item_begin) - item_begin
            inner_r = min(end, item_end) - item_begin

            # Get original data and labels
            ori_idx = self.shfl_inds[i]
            ori_input_ids = self.dataset[ori_idx]['input_ids']
            ori_labels = self.dataset[ori_idx]['labels']

            # Add original data and labels from calculated positions
            # to trunc_ids and trunc_labels
            trunc_input_ids.extend(ori_input_ids[inner_l:inner_r])
            trunc_labels.extend(ori_labels[inner_l:inner_r])
            trunc_sizes.append(inner_r - inner_l)

        # return populated lists of truncated ids, labels and their cumulative
        # lengths
        return trunc_input_ids, trunc_labels, trunc_sizes

    def __len__(self):
        return self._num_packed_samples

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        # The cumulative length from the start position of this data
        begin = item * self.max_length
        # The cumulative length from the end position of this data
        end = (item + 1) * self.max_length

        # Extract data within the range from the shuffled original dataset.
        _res = self._pack_ids_and_labels_in_range(begin, end)
        packed_input_ids, packed_labels, num_tokens = _res
        assert self.max_length == len(packed_input_ids) == len(packed_labels)

        packed = {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'num_tokens': num_tokens,
        }

        return packed


class TextCollator():

    def __init__(self, pack_batch=False):
        self.pack_batch = pack_batch

    def __call__(self, instances):

        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        num_tokens = []

        for data in instances:
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))
            num_tokens.extend(data['num_tokens'])

        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        num_tokens = torch.IntTensor(num_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if get_sp_world_size() > 1:
            ori_seq_len = input_ids.shape[1]
            input_ids = pad_for_sequence_parallel(input_ids, pad_index)
            labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)
            pad_seq_len = input_ids.shape[1] - ori_seq_len
            if pad_seq_len > 0:
                pad_num_token = torch.tensor([pad_seq_len]).int()
                num_tokens = torch.cat([num_tokens, pad_num_token])

        if input_ids.shape != labels.shape:
            logger.error(f'[instances] {instances}')
            logger.error(f'[num_tokens] {num_tokens}')
            logger.error(f'[input_ids] {input_ids}')
            logger.error(f'[labels] {labels}')
            raise RuntimeError('The shape of input_ids and labels must be '
                               f'equal, but  found {input_ids.shape} and '
                               f'{labels.shape}.')
        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': num_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict
