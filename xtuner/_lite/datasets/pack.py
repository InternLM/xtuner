# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import itertools
import random

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import ConcatDataset


class SoftPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, target=2048, blend=False, sort=False):
        if blend:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.target = target

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _infos = self.get_pack_infos(dataset, i, num_tokens[i])
            pack_infos.append(_infos)
        self.pack_infos = concatenate_datasets(pack_infos)

    @property
    def longest(self):
        return self.pack_infos["longest"]

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        # _ori_lens = dataset['num_tokens']
        inds = [i for i in range(len(dataset))]
        random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        longest = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= self.target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                longest = max(longest, num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                        "longest": int(longest),
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                longest = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                "dataset_id": dataset_id,
                "indices": item_buffer,
                "longest": int(longest),
            }

            pack_infos.append(info)

        pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        indices = self.pack_infos[item]["indices"]
        dataset_id = self.pack_infos[item]["dataset_id"]
        return [self.datasets[dataset_id][i] for i in indices]


class HardPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, target=2048, blend=True, sort=False):
        if blend:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.target = target

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _info = self.get_pack_info(dataset, i, num_tokens[i])
            pack_infos.append(_info)

        _ranges_left = []
        _ranges_right = []
        _num_packed_samples = []
        _indices = []
        _max_length_per_pack = []
        _dataset_id = []
        for info in pack_infos:
            _ranges_left.extend(info["ranges_left"])
            _ranges_right.extend(info["ranges_right"])
            _num_packed_samples.append(info["num_packed_samples"])
            _indices.extend(info["indices"])
            _max_length_per_pack.extend(info["max_length_per_pack"])
            _dataset_id.extend(info["dataset_id"])

        self.pack_infos = {
            "ranges_left": _ranges_left,
            "ranges_right": _ranges_right,
            "num_packed_samples": _num_packed_samples,
            "indices": _indices,
            "max_length_per_pack": _max_length_per_pack,
            "dataset_id": _dataset_id,
        }

    @classmethod
    def _cal_max_length(cls, begin, end, shfl_item_rngs_left, shfl_item_rngs_right):
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

    def get_pack_info(self, dataset, dataset_id, num_tokens):
        # The number of data items after packing
        num_packed_samples = int(num_tokens.sum() / self.target)

        # Shuffle the order of the original dataset
        # The packing will proceed according to the order after shuffle.
        # Assume the following conditions hold:
        #   (1) shfl_inds = [3, 1, 2, 0]
        #   (2) self._ori_lens[3] + self._ori_lens[1] = max_length
        #   (3) self._ori_lens[2] + self._ori_lens[0] = max_length
        # Ultimately, dataset[3] and dataset[1] will be combined into a new
        # data, and dataset[2] and dataset[0] will be combined into a new data.
        inds = [i for i in range(len(dataset))]
        # if seed is not None:
        #     random.seed(seed)
        random.shuffle(inds)
        shfl_inds = inds

        # shuffled cumulative lengths
        shfl_lens = [num_tokens[i] for i in shfl_inds]
        shfl_acc_lens = list(itertools.accumulate(shfl_lens))

        shfl_item_rngs_left = [0] + shfl_acc_lens[:-1]
        shfl_item_rngs_right = shfl_acc_lens

        max_length_per_pack = []
        belong_dataset_ids = []
        for i in range(num_packed_samples):
            begin = i * self.target
            end = (i + 1) * self.target
            max_length_per_pack.append(
                self._cal_max_length(
                    begin, end, shfl_item_rngs_left, shfl_item_rngs_right
                )
            )
            belong_dataset_ids.append(dataset_id)

        pack_infos = {
            "ranges_left": shfl_item_rngs_left,
            "ranges_right": shfl_item_rngs_right,
            "num_packed_samples": num_packed_samples,
            "indices": shfl_inds,
            "dataset_id": belong_dataset_ids,
            "max_length_per_pack": max_length_per_pack,
        }

        # pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

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
        left = bisect.bisect(self.pack_infos["ranges_left"], begin)
        right = bisect.bisect(self.pack_infos["ranges_right"], end)

        trunc_input_ids = []
        trunc_labels = []
        trunc_sizes = []

        for i in range(left, right):
            # Determine the real range we will cut in current original item
            item_begin = self.pack_infos["ranges_left"][i]
            item_end = self.pack_infos["ranges_right"][i]

            # Calculate exact positions within current dataset item
            inner_l = max(begin, item_begin) - item_begin
            inner_r = min(end, item_end) - item_begin

            # Get original data and labels
            ori_idx = self.pack_infos["indices"][i]
            ori_dataset_id = self.pack_infos["dataset_id"][i]
            ori_input_ids = self.datasets[ori_dataset_id][ori_idx]["input_ids"]
            ori_labels = self.datasets[ori_dataset_id][ori_idx]["labels"]

            # Add original data and labels from calculated positions
            # to trunc_ids and trunc_labels
            trunc_input_ids.extend(ori_input_ids[inner_l:inner_r])
            trunc_labels.extend(ori_labels[inner_l:inner_r])
            trunc_sizes.append(inner_r - inner_l)

        # return populated lists of truncated ids, labels and their cumulative
        # lengths
        return trunc_input_ids, trunc_labels, trunc_sizes

    def __len__(self):
        return len(self.pack_infos["indices"])

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        # The cumulative length from the start position of this data
        begin = item * self.target
        # The cumulative length from the end position of this data
        end = (item + 1) * self.target

        # Extract data within the range from the shuffled original dataset.
        _res = self._pack_ids_and_labels_in_range(begin, end)
        packed_input_ids, packed_labels, num_tokens = _res
        assert self.target == len(packed_input_ids) == len(packed_labels)

        packed = {
            "input_ids": packed_input_ids,
            "labels": packed_labels,
            "num_tokens": num_tokens,
        }

        return packed
