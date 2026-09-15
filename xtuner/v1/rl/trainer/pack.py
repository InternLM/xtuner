import math
import random
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch

from xtuner.v1.datasets.sampler import get_length_grouped_indices


PackIndices: TypeAlias = list[int]
OptimizerStepPackIndices: TypeAlias = list[PackIndices]
DPRankPackIndices: TypeAlias = list[OptimizerStepPackIndices]
PackedDataIndices: TypeAlias = list[DPRankPackIndices]


def get_soft_pack_infos(data_indices: Sequence[int], num_tokens: Sequence[int], target: int) -> list[PackIndices]:
    """Group sample indices into sequential packs without touching sample data."""
    if len(data_indices) != len(num_tokens):
        raise ValueError("data_indices and num_tokens must have the same length")

    pack_infos: list[PackIndices] = []
    current_indices: PackIndices = []
    current_len = 0

    for data_index, token_len in zip(data_indices, num_tokens):
        if current_len + token_len <= target:
            current_indices.append(int(data_index))
            current_len += token_len
        else:
            if current_indices:
                pack_infos.append(current_indices)
            current_indices = [int(data_index)]
            current_len = token_len

    if current_indices:
        pack_infos.append(current_indices)
    return pack_infos


class RLDataPacker:
    """Build an index-only packing plan for RL training data.

    The packer owns scheduling decisions only. It never creates padding tensors,
    concatenates ``SequenceContext`` objects, or otherwise materializes a pack.
    """

    def __init__(
        self,
        pack_max_length: int,
        world_size: int,
        data_replicate_size: int,
        optimizer_steps: int,
        pack_strategy: str = "greedy",
        seed: int = 42,
    ):
        self.pack_max_length = pack_max_length
        self.world_size = world_size
        self.data_replicate_size = data_replicate_size
        self.optimizer_steps = optimizer_steps
        self.split_size = 1024
        self.dp_size = self.world_size // self.data_replicate_size
        self.seed = seed
        self.strategy_map = {
            "greedy": self.greedy_pack_and_split,
            "balance": self.balance_split_and_pack,
            "native": self.native_split_and_pack,
        }
        if pack_strategy not in self.strategy_map:
            raise ValueError(f"Unknown packing strategy: {pack_strategy}")
        self._impl = self.strategy_map[pack_strategy]

    def pack(self, data_lengths: Sequence[int]) -> tuple[PackedDataIndices, int]:
        """Return ``[dp][optimizer_step][pack][sample_index]`` and padding size."""
        if not data_lengths:
            return [], 0
        for data_index, data_length in enumerate(data_lengths):
            if data_length > self.pack_max_length:
                raise ValueError(
                    f"Single sample {data_index} seq len {data_length} exceeds "
                    f"pack_max_length {self.pack_max_length}"
                )

        data_indices = list(range(len(data_lengths)))
        packed_data_indices = self._impl(data_indices, data_lengths)
        padding_tokens = self._count_padding_tokens(packed_data_indices, data_lengths)
        return packed_data_indices, padding_tokens

    def native_split_and_pack(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        # Use private negative indices as scheduling-only padding slots. They are
        # removed from the returned plan and materialized as padding by workers.
        scheduled_indices = data_indices.copy()
        if len(scheduled_indices) % self.dp_size != 0:
            pad_num = self.dp_size - (len(scheduled_indices) % self.dp_size)
            scheduled_indices.extend(-(index + 1) for index in range(pad_num))

        batches_per_dp_group = np.array_split(scheduled_indices, self.dp_size)
        actual_optimizer_steps = min(len(batches_per_dp_group[0]), self.optimizer_steps)
        packed_data_indices: PackedDataIndices = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(self.dp_size)
        ]
        max_packs_per_step = [0] * actual_optimizer_steps

        for dp_rank, dp_worker_indices in enumerate(batches_per_dp_group):
            indices_for_optim_steps = np.array_split(dp_worker_indices, actual_optimizer_steps)
            for step_idx, step_indices_array in enumerate(indices_for_optim_steps):
                step_indices = [int(index) for index in step_indices_array]
                packed_step_indices = self._pack_indices(step_indices, data_lengths)
                packed_data_indices[dp_rank][step_idx] = packed_step_indices
                max_packs_per_step[step_idx] = max(max_packs_per_step[step_idx], len(packed_step_indices))

        self._align_pack_count(packed_data_indices, max_packs_per_step)
        return packed_data_indices

    def balance_split_and_pack(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        torch_generator = torch.Generator().manual_seed(self.seed)
        random_generator = random.Random(self.seed)
        grouped_indices = get_length_grouped_indices(
            max_lengths=list(data_lengths),
            group_batch_size=len(data_indices),
            group_size=self.dp_size,
            torch_generator=torch_generator,
            random_generator=random_generator,
        )

        partitioned_indices: PackedDataIndices = [
            [[] for _ in range(self.optimizer_steps)] for _ in range(self.dp_size)
        ]
        for i, data_index in enumerate(grouped_indices):
            dp_rank = i % self.dp_size
            step_idx = (i // self.dp_size) % self.optimizer_steps
            partitioned_indices[dp_rank][step_idx].append(int(data_index))

        packed_data_indices: PackedDataIndices = [
            [[] for _ in range(self.optimizer_steps)] for _ in range(self.dp_size)
        ]
        max_packs_per_step = [0] * self.optimizer_steps
        for dp_rank in range(self.dp_size):
            for step_idx in range(self.optimizer_steps):
                packed_step_indices = self._pack_indices(
                    partitioned_indices[dp_rank][step_idx],
                    data_lengths,
                )
                packed_data_indices[dp_rank][step_idx] = packed_step_indices
                max_packs_per_step[step_idx] = max(max_packs_per_step[step_idx], len(packed_step_indices))

        self._align_pack_count(packed_data_indices, max_packs_per_step)
        return packed_data_indices

    def greedy_pack_and_split(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        total_pack_indices = self._pack_indices(data_indices, data_lengths)
        pad_num = math.ceil(len(total_pack_indices) / self.dp_size) * self.dp_size - len(total_pack_indices)
        total_pack_indices.extend([[] for _ in range(pad_num)])

        each_dp_batches_num = len(total_pack_indices) // self.dp_size
        if each_dp_batches_num < self.optimizer_steps:
            iters_per_step = 1
            actual_optimizer_steps = each_dp_batches_num
        else:
            iters_per_step = math.ceil(each_dp_batches_num / self.optimizer_steps)
            actual_optimizer_steps = math.ceil(each_dp_batches_num / iters_per_step)

        packed_data_indices: PackedDataIndices = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(self.dp_size)
        ]
        for dp_rank in range(self.dp_size):
            for step_idx in range(actual_optimizer_steps):
                start_idx = dp_rank * each_dp_batches_num + step_idx * iters_per_step
                end_idx = min(start_idx + iters_per_step, each_dp_batches_num * (dp_rank + 1))
                packed_data_indices[dp_rank][step_idx] = total_pack_indices[start_idx:end_idx]
        return packed_data_indices

    def _pack_indices(self, data_indices: Sequence[int], data_lengths: Sequence[int]) -> OptimizerStepPackIndices:
        scheduled_lengths = [self._get_scheduled_length(data_index, data_lengths) for data_index in data_indices]
        if sum(scheduled_lengths) > self.pack_max_length:
            packs = get_soft_pack_infos(data_indices, scheduled_lengths, self.pack_max_length)
        else:
            packs = [[int(data_index) for data_index in data_indices]]
        return [[data_index for data_index in pack if data_index >= 0] for pack in packs]

    def _get_scheduled_length(self, data_index: int, data_lengths: Sequence[int]) -> int:
        if data_index >= 0:
            return data_lengths[data_index]
        return min(self.split_size, self.pack_max_length)

    def _align_pack_count(self, packed_data_indices: PackedDataIndices, max_packs_per_step: Sequence[int]) -> None:
        for step_idx, max_packs in enumerate(max_packs_per_step):
            for dp_rank in range(self.dp_size):
                missing_packs = max_packs - len(packed_data_indices[dp_rank][step_idx])
                packed_data_indices[dp_rank][step_idx].extend([[] for _ in range(missing_packs)])

    def _count_padding_tokens(
        self,
        packed_data_indices: PackedDataIndices,
        data_lengths: Sequence[int],
    ) -> int:
        total_packs = 0
        scheduled_tokens = 0
        seen_indices: list[int] = []
        for dp_rank_indices in packed_data_indices:
            for step_indices in dp_rank_indices:
                total_packs += len(step_indices)
                for pack_indices in step_indices:
                    seen_indices.extend(pack_indices)
                    scheduled_tokens += sum(data_lengths[data_index] for data_index in pack_indices)

        if sorted(seen_indices) != list(range(len(data_lengths))):
            raise RuntimeError("Packing plan must contain every data index exactly once")
        return total_packs * self.pack_max_length - scheduled_tokens
