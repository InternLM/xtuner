# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Iterator, Optional

import torch
from mmengine.dist import sync_random_seed
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Sampler

from xtuner.v1.utils import get_logger

from .jsonl import JsonlDataset
from .packing import _LegacySoftPackDataset


logger = get_logger()


class ParallelSampler(Sampler):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(
        self,
        dataset: TorchConcatDataset[JsonlDataset] | _LegacySoftPackDataset,
        global_batch_size: int,
        dp_mesh: DeviceMesh | None = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        super().__init__()
        if dp_mesh is not None:
            rank = dp_mesh.get_local_rank()
            world_size = dp_mesh.size()
        else:
            rank = 0
            world_size = 1

        assert global_batch_size % world_size == 0
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.step = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / global_batch_size) * global_batch_size // world_size
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]

        # subsample
        indices = indices[self.step + self.rank : self.total_size : self.world_size]

        yield from iter(indices)
        self.step = 0

    def __len__(self) -> int:
        """The number of samples in this rank."""
        # TODO: not same with LengthGroupedSampler?
        return self.num_samples - self.step

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def load_state_dict(self, state_dict) -> None:
        """Load the sampler state.

        Args:
            state_dict (dict): The state of the sampler.
        """
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]

        if self.shuffle != state_dict["shuffle"]:
            raise ValueError(
                f"The shuffle in the state_dict ({state_dict.get('shuffle')}) "
                f"is different from the current shuffle ({self.shuffle})."
            )

    def get_state_dict(self, step: int):
        # Attention! Do not set self.step here, or it will cause the next __iter__ to get less samples.
        # self.step = step % self.total_size
        return {
            "epoch": self.epoch,
            "step": step,
            "world_size": self.world_size,
            "shuffle": self.shuffle,
            "round_up": self.round_up,
            "num_samples": self.num_samples,
            "total_size": self.total_size,
        }


def get_length_grouped_indices(
    max_lengths, group_batch_size, group_size, torch_generator: torch.Generator, random_generator: random.Random
):
    indices = torch.randperm(len(max_lengths), generator=torch_generator)
    megabatches = [indices[i : i + group_batch_size].tolist() for i in range(0, len(max_lengths), group_batch_size)]
    output = []
    for megabatch in megabatches:
        megabatch = sorted(megabatch, key=lambda i: max_lengths[i], reverse=True)
        grouped_megabatch = [megabatch[i : i + group_size] for i in range(0, len(megabatch), group_size)]
        random_generator.shuffle(grouped_megabatch)
        for group in grouped_megabatch:
            output.extend(group)

    return output


class LengthGroupedSampler(Sampler):
    GROUP_BATCH_FACTOR = 4
    MAX_GROUP_BATCH_SIZE = 50

    def __init__(
        self,
        dataset: _LegacySoftPackDataset,
        global_batch_size: int,
        dp_mesh: DeviceMesh | None = None,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        super().__init__()

        if dp_mesh is not None:
            rank = dp_mesh.get_local_rank()
            world_size = dp_mesh.size()
        else:
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size
        self.torch_generator = torch.Generator()
        self.random_generator = random.Random()
        assert global_batch_size % world_size == 0

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.step = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / global_batch_size) * global_batch_size // world_size
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        # Default for mega_batch_mult: 50 or the number to get 4
        # megabatches, whichever is smaller.
        mega_batch_mult = min(
            len(self.dataset) // (global_batch_size * self.GROUP_BATCH_FACTOR), self.MAX_GROUP_BATCH_SIZE
        )
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1
        self.group_batch_size = mega_batch_mult * global_batch_size
        self.group_size = self.world_size

        self.max_lengths = self.dataset.longest
        assert isinstance(self.max_lengths, (list, tuple))

        self.global_batch_size = global_batch_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        if self.seed is not None:
            self.torch_generator.manual_seed(self.seed + self.epoch)
            self.random_generator.seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(
            max_lengths=self.max_lengths,
            group_batch_size=self.group_batch_size,
            group_size=self.group_size,
            torch_generator=self.torch_generator,
            random_generator=self.random_generator,
        )
        assert len(set(indices)) == len(indices)
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]
        # subsample
        assert len(indices) == self.total_size
        indices = indices[self.step + self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples - self.step // self.world_size
        yield from iter(indices)
        self.step = 0

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

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the sampler state.

        Args:
            state_dict (dict): The state of the sampler.
        """
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]

        if self.group_batch_size != (origin_group_batch_size := state_dict["group_batch_size"]):
            logger.warning(
                f"The group_batch_size in the state_dict ({origin_group_batch_size}) "
                f"is different from the current group_batch_size ({self.group_batch_size})."
            )

        if self.group_size != (origin_group_size := state_dict["group_size"]):
            logger.warning(
                f"The group_size in the state_dict ({state_dict.get('group_size')}) "
                f"is different from the current group_size ({self.group_size}). "
                "The balance of grouped sampling may be affected, which will slow down training."
            )
            self.group_size = origin_group_size

    def get_state_dict(self, step: int):
        """Get the sampler state dict.

        Returns:
            dict: The state of the sampler.
        """
        # Attention! Do not set self.step here, or it will cause the next __iter__ to get less samples.
        # self.step = step % self.total_size
        return {
            "epoch": self.epoch,
            "step": self.step,
            "world_size": self.world_size,
            "round_up": self.round_up,
            "num_samples": self.num_samples,
            "total_size": self.total_size,
            "group_batch_size": self.group_batch_size,
            "group_size": self.group_size,
        }
