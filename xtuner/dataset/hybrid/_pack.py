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

        self._ori_img_urls = dataset['image_urls']
        self._ori_img_rngs = dataset['image_ranges']
        self._ori_lens = dataset['tokens']

        self._num_packed_samples = sum(self._ori_lens) // self.max_length

        inds = [i for i in range(len(self.dataset))]
        random.shuffle(inds)
        self.shfl_inds = inds

        shfl_lens = [self._ori_lens[i] for i in inds]
        # shuffled cumulative lengths
        shfl_acc_lens = list(itertools.accumulate(shfl_lens))

        self._shfl_item_rngs_left = [0] + shfl_acc_lens[:-1]
        self._shfl_item_rngs_right = shfl_acc_lens

        shfl_img_urls = [self._ori_img_urls[i] for i in inds]
        self._flat_shfl_img_urls = list(itertools.chain(*shfl_img_urls))

        flat_shfl_acc_img_rngs = []
        flat_shfl_acc_img_rngs_left = []
        flat_shfl_acc_img_rngs_right = []
        for i in range(len(self.dataset)):
            shfl_i = self.shfl_inds[i]
            img_rngs = self._ori_img_rngs[shfl_i]
            for left, right in img_rngs:
                acc_left = left + self._shfl_item_rngs_left[i]
                acc_right = right + self._shfl_item_rngs_left[i]

                flat_shfl_acc_img_rngs_left.append(acc_left)
                flat_shfl_acc_img_rngs_right.append(acc_right)
                flat_shfl_acc_img_rngs.append([acc_left, acc_right])
        assert len(flat_shfl_acc_img_rngs) == len(self._flat_shfl_img_urls)

        self._flat_shfl_acc_img_rngs = flat_shfl_acc_img_rngs
        self._flat_shfl_acc_img_rngs_left = flat_shfl_acc_img_rngs_left
        self._flat_shfl_acc_img_rngs_right = flat_shfl_acc_img_rngs_right

    def _pack_img_urls_and_rngs_in_range(self, begin, end):

        left = bisect.bisect(self._flat_shfl_acc_img_rngs_right, begin)
        right = bisect.bisect(self._flat_shfl_acc_img_rngs_left, end)

        filter_urls = self._flat_shfl_img_urls[left:right]
        filter_rngs = self._flat_shfl_acc_img_rngs[left:right]

        inner_rngs = []
        for rng in filter_rngs:
            inner_left = max(begin, rng[0]) - begin
            inner_right = min(end, rng[1]) - begin

            if inner_right - inner_left > 0:
                inner_rngs.append([inner_left, inner_right])

        return filter_urls, inner_rngs

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

        return trunc_ids, trunc_labels, cumulative_len, position_ids

    def __len__(self):
        return self._num_packed_samples

    def __getitem__(self, item):

        begin = item * self.max_length
        end = (item + 1) * self.max_length

        _res = self._pack_ids_and_labels_in_range(begin, end)
        packed_ids, packed_labels, cumulative_len, position_ids = _res
        assert self.max_length == len(packed_ids) == len(packed_labels)

        _res = self._pack_img_urls_and_rngs_in_range(begin, end)
        packed_img_urls, packed_img_rngs = _res

        for left, right in packed_img_rngs:
            assert len(set(packed_ids[left:right])) == 1

        packed = {
            'input_ids': packed_ids,
            'labels': packed_labels,
            'tokens': self.max_length,
            'image_urls': packed_img_urls,
            'image_ranges': packed_img_rngs,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids
        }

        return packed
