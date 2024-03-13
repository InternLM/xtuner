import logging
import warnings
from typing import Iterator, Optional, Sized

import numpy as np
from mmengine import print_log
from torch.utils.data import Sampler

from xtuner.parallel.sequence import (get_data_parallel_rank,
                                      get_data_parallel_world_size)


class InternRepoSampler(Sampler):

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        if seed is not None and seed != 1024:
            warnings.warn('For alignment accuracy, seed in InternRepoSampler'
                          'must be set to 1024.')
        world_size = get_data_parallel_world_size()
        rank = get_data_parallel_rank()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = 1024
        self.epoch = 0

        self.num_samples = len(self.dataset) // world_size
        self.total_size = self.num_samples * world_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = np.arange(len(self.dataset))
            rng.shuffle(indices)
            indices = indices.tolist()
        else:
            indices = np.arange(len(self.dataset)).tolist()

        self.indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        self.subsample_indices = indices

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class InternlmRepoSampler(InternRepoSampler):

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(dataset, shuffle, seed)
        print_log(('InternlmRepoSampler will be deprecated in the future.'
                   'Please use InternRepoSampler instead.'),
                  logger='current',
                  level=logging.WARNING)
