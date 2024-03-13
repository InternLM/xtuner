# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sized

from mmengine.dataset import DefaultSampler
from mmengine.dist import sync_random_seed

from .setup_distributed import (get_data_parallel_rank,
                                get_data_parallel_world_size)


class SequenceParallelSampler(DefaultSampler):

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank = get_data_parallel_rank()
        world_size = get_data_parallel_world_size()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)
