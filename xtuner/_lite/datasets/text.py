import bisect
import itertools
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite.chat import ChatMessages
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .format import OPENAI_FORMAT_MAP


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


class SoftPackerForText(torch.utils.data.Dataset):

    def __init__(self, dataset, max_length=2048, use_varlen_attn=True):
        super().__init__()

        if use_varlen_attn and not is_flash_attn_2_available():
            raise NotImplementedError('`use_varlen_attn=True` requires the '
                                      'installation of `flash_attn`')

        self.max_length = max_length
        self.use_varlen_attn = use_varlen_attn
        # unpack dataset
        self.dataset = dataset

        self._ori_lens = dataset['num_tokens']

        inds = [i for i in range(len(self.dataset))]
        random.shuffle(inds)

        _packed_length = 0
        _packed_items = []
        self.pack_lut = []

        for i in inds:
            if self._ori_lens[i] + _packed_length <= max_length:
                _packed_items.append(i)
                _packed_length += self._ori_lens[i]
            else:
                self.pack_lut.append(_packed_items)
                _packed_items = []
                _packed_length = 0

        if len(_packed_items) > 0:
            self.pack_lut.append(_packed_items)

        # The number of data items after packing
        self._num_packed_samples = len(self.pack_lut)

    def __len__(self):
        return self._num_packed_samples

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """

        packed_items = self.pack_lut[item]

        packed_input_ids = []
        packed_labels = []
        num_tokens = []
        for i in packed_items:
            packed_input_ids.extend(self.dataset[i]['input_ids'])
            packed_labels.extend(self.dataset[i]['labels'])

            _num_tokens = self.dataset[i]['num_tokens']
            num_tokens.append(_num_tokens)

        packed = {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'num_tokens': num_tokens,
        }

        return packed


class TextTokenizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """

        data = {
            'input_ids': self.dataset[item]['input_ids'],
            'labels': self.dataset[item]['labels'],
            'num_tokens': [self.dataset[item]['num_tokens']]
        }

        return data


class TextRawDataset(torch.utils.data.Dataset):

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


class HardPackerForText(torch.utils.data.Dataset):
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

    def __init__(self, dataset, max_length=2048, use_varlen_attn=True):
        super().__init__()

        if use_varlen_attn and not is_flash_attn_2_available():
            raise NotImplementedError('`use_varlen_attn=True` requires the '
                                      'installation of `flash_attn`')

        self.max_length = max_length
        self.use_varlen_attn = use_varlen_attn
        # unpack dataset
        self.dataset = dataset

        self._ori_lens = dataset['num_tokens']

        # The number of data items after packing
        self._num_packed_samples = sum(self._ori_lens) // self.max_length

        # Shuffle the order of the original dataset
        # The packing will proceed according to the order after shuffle.
        # Assume the following conditions hold:
        #   (1) shfl_inds = [3, 1, 2, 0]
        #   (2) self._ori_lens[3] + self._ori_lens[1] = max_length
        #   (3) self._ori_lens[2] + self._ori_lens[0] = max_length
        # Ultimately, dataset[3] and dataset[1] will be combined into a new
        # data, and dataset[2] and dataset[0] will be combined into a new data.
        inds = [i for i in range(len(self.dataset))]
        random.shuffle(inds)
        self.shfl_inds = inds

        # shuffled cumulative lengths
        shfl_lens = [self._ori_lens[i] for i in inds]
        shfl_acc_lens = list(itertools.accumulate(shfl_lens))

        self._shfl_item_rngs_left = [0] + shfl_acc_lens[:-1]
        self._shfl_item_rngs_right = shfl_acc_lens

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
        trunc_position_ids = []
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
            trunc_position_ids.extend(range(inner_r - inner_l))
            trunc_sizes.append(inner_r - inner_l)

        # return populated lists of truncated ids, labels and their cumulative
        # lengths
        return trunc_input_ids, trunc_labels, trunc_position_ids, trunc_sizes

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
        packed_input_ids, packed_labels, packed_pos_ids, unpack_sizes = _res
        assert self.max_length == len(packed_input_ids) == len(packed_labels)

        if self.use_varlen_attn:
            position_ids = packed_pos_ids
        else:
            position_ids = [i for i in range(self.max_length)]

        packed = {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'position_ids': position_ids,
            'chunk_sizes': unpack_sizes,
        }

        return packed


class TextCollator():

    def __init__(self, pack_batch=False):
        self.pack_batch = pack_batch

    def __call__(self, instances):

        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        attention_mask = []
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

        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': num_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict
