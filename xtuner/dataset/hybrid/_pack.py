import bisect
import itertools
import random

import torch


class _PackDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, max_length=2048):
        super().__init__()

        self.max_length = max_length

        # unpack dataset
        self.dataset = dataset

        self._ori_lens = dataset['num_tokens']

        self._num_packed_samples = sum(self._ori_lens) // self.max_length

        inds = [i for i in range(len(self.dataset))]
        random.shuffle(inds)
        self.shfl_inds = inds

        shfl_lens = [self._ori_lens[i] for i in inds]
        # shuffled cumulative lengths
        shfl_acc_lens = list(itertools.accumulate(shfl_lens))

        self._shfl_item_rngs_left = [0] + shfl_acc_lens[:-1]
        self._shfl_item_rngs_right = shfl_acc_lens

    def _pack_ids_and_labels_in_range(self, begin, end):

        left = bisect.bisect(self._shfl_item_rngs_right, begin)
        right = bisect.bisect(self._shfl_item_rngs_left, end)

        trunc_ids = []
        trunc_labels = []
        cumulative_len = []
        position_ids = []
        for i in range(left, right):
            cumulative_len.append(len(trunc_ids))

            item_begin = self._shfl_item_rngs_left[i]
            item_end = self._shfl_item_rngs_right[i]

            inner_l = max(begin, item_begin) - item_begin
            inner_r = min(end, item_end) - item_begin
            position_ids.extend([i for i in range(inner_r - inner_l)])

            ori_idx = self.shfl_inds[i]
            ori_input_ids = self.dataset[ori_idx]['input_ids']
            ori_labels = self.dataset[ori_idx]['labels']

            trunc_ids.extend(ori_input_ids[inner_l:inner_r])
            trunc_labels.extend(ori_labels[inner_l:inner_r])

        cumulative_len.append(self.max_length)
        return trunc_ids, trunc_labels, cumulative_len, position_ids

    def __len__(self):
        return self._num_packed_samples

    def __getitem__(self, item):

        begin = item * self.max_length
        end = (item + 1) * self.max_length

        _res = self._pack_ids_and_labels_in_range(begin, end)
        packed_ids, packed_labels, cumulative_len, position_ids = _res
        assert self.max_length == len(packed_ids) == len(packed_labels)

        packed = {
            'input_ids': packed_ids,
            'labels': packed_labels,
            'num_tokens': self.max_length,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids
        }

        return packed
