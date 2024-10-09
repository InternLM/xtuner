import random

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import ConcatDataset


class SoftPackDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, target=2048, blend=False, sort=False):

        if blend:
            num_tokens = [
                np.concatenate([dset.num_tokens for dset in datasets])
            ]
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

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        # _ori_lens = dataset['num_tokens']
        inds = [i for i in range(len(dataset))]
        random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        max_length_one_pack = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= self.target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                max_length_one_pack = max(max_length_one_pack,
                                          num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        'dataset_id': dataset_id,
                        'indices': item_buffer,
                        'max_length': max_length_one_pack
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                max_length_one_pack = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                'dataset_id': dataset_id,
                'indices': item_buffer,
                'max_length': max_length_one_pack
            }

            pack_infos.append(info)

        pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        indices = self.pack_infos[item]['indices']
        dataset_id = self.pack_infos[item]['dataset_id']
        return [self.datasets[dataset_id][i] for i in indices]
