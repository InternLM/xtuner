from __future__ import annotations

import heapq
import math
import random
from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias

import numpy as np
import torch

from xtuner.v1.datasets.sampler import get_length_grouped_indices
from xtuner.v1.utils import XTUNER_DETERMINISTIC


PackIndices: TypeAlias = list[int]
OptimizerStepPackIndices: TypeAlias = list[PackIndices]
DPRankPackIndices: TypeAlias = list[OptimizerStepPackIndices]
PackedDataIndices: TypeAlias = list[DPRankPackIndices]

_BALANCE_DEFAULT_SEED = 42


class TokenTensorBatch(NamedTuple):
    """Intermediate type for passing a list of tensors across Ray.

    A list of 1D tensors is sent as one float32 tensor (``data``) plus one
    length per sample (``lengths``). Tensors with more than one dimension are rejected.
    """

    data: torch.Tensor
    lengths: torch.Tensor

    @staticmethod
    def cat_samples(samples: Sequence[torch.Tensor]) -> TokenTensorBatch:
        """``list[torch.Tensor]`` -> ``TokenTensorBatch``.

        Each sample must be a 1D tensor.
        """
        if not samples:
            return TokenTensorBatch(torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.int32))
        for index, sample in enumerate(samples):
            if sample.ndim != 1:
                raise ValueError(
                    f"cat_samples requires a 1D tensor per sample, got shape {tuple(sample.shape)} at index {index}"
                )
        flat = [sample.detach().reshape(-1).to(dtype=torch.float32, device="cpu").contiguous() for sample in samples]
        lengths = torch.tensor([sample.numel() for sample in flat], dtype=torch.int32)
        data = flat[0] if len(flat) == 1 else torch.cat(flat)
        return TokenTensorBatch(data, lengths)

    def split_samples(self) -> list[torch.Tensor]:
        """``TokenTensorBatch`` -> ``list[torch.Tensor]``.

        Split into one 1D tensor per sample.
        """
        if int(self.lengths.numel()) == 0:
            return []
        return list(self.data.split(tuple(int(length) for length in self.lengths)))


def get_greedy_pack_infos(data_indices: Sequence[int], num_tokens: Sequence[int], target: int) -> list[PackIndices]:
    """Group sample indices into sequential greedy packs without touching
    sample data.

    Samples are visited in the given order and appended to the current pack while they
    fit; the first non-fitting sample starts a new pack.

    Args:
        data_indices (Sequence[int]): Sample indices to schedule, in visit order.
        num_tokens (Sequence[int]): Training token length of each sample, aligned with
            ``data_indices``.
        target (int): Maximum token budget of one pack.

    Returns:
        list[PackIndices]: The greedy pack layout; every input index appears exactly
        once.
    """
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


def get_lpt_pack_infos(
    data_indices: Sequence[int],
    num_tokens: Sequence[int],
    pack_max_length: int,
    num_bins: int,
    dp_size: int,
) -> list[PackIndices]:
    """Assign samples with the longest-processing-time heuristic.

    Samples are placed from longest to shortest into the currently least-loaded pack that
    still has room. Equal loads prefer the smaller pack index. When a sample fits in no
    pack, ``num_bins`` grows by ``dp_size`` and the assignment restarts.

    Args:
        data_indices (Sequence[int]): Sample indices to place.
        num_tokens (Sequence[int]): Training token length of each sample, aligned with
            ``data_indices``.
        pack_max_length (int): Maximum token budget of one pack.
        num_bins (int): Initial number of packs. Grown by ``dp_size`` until every sample fits.
        dp_size (int): Data parallel size. The returned pack count is a multiple of this.

    Returns:
        list[PackIndices]: One entry per pack, including empty packs. Indices inside a pack
        follow assignment order.
    """
    if len(data_indices) != len(num_tokens):
        raise ValueError("data_indices and num_tokens must have the same length")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    for position, token_len in enumerate(num_tokens):
        if token_len > pack_max_length:
            raise ValueError(
                f"Single sample {data_indices[position]} seq len {token_len} exceeds pack_max_length {pack_max_length}"
            )

    order = sorted(range(len(data_indices)), key=lambda position: num_tokens[position], reverse=True)
    while True:
        heap = [(0, bin_index) for bin_index in range(num_bins)]
        heapq.heapify(heap)
        bins: list[PackIndices] = [[] for _ in range(num_bins)]
        feasible = True
        for position in order:
            load, bin_index = heap[0]
            token_len = num_tokens[position]
            if load + token_len <= pack_max_length:
                heapq.heapreplace(heap, (load + token_len, bin_index))
                bins[bin_index].append(int(data_indices[position]))
            else:
                feasible = False
                break
        if feasible:
            return bins
        num_bins += dp_size


def _partition_packs(packs: list[PackIndices], max_updates: int) -> list[list[PackIndices]]:
    """Split packs into optimizer steps as evenly as possible.

    Earlier steps receive one extra pack when the count is not divisible by ``max_updates``.
    """
    if not packs:
        return []
    num_updates = min(len(packs), max_updates)
    quotient, remainder = divmod(len(packs), num_updates)
    chunks: list[list[PackIndices]] = []
    offset = 0
    for chunk_index in range(num_updates):
        chunk_size = quotient + int(chunk_index < remainder)
        chunks.append(packs[offset : offset + chunk_size])
        offset += chunk_size
    return chunks


class RLDataPacker:
    """Build an index-only packing plan for RL training data.

    The packer owns scheduling decisions only. It never creates padding tensors,
    concatenates ``SequenceContext`` objects, or otherwise materializes a pack. Empty
    packs (``[]``) are scheduling-only padding slots: the training worker materializes
    them as all-padding packs of ``pack_max_length`` tokens.

    Args:
        pack_max_length (int): Maximum token budget of one pack.
        world_size (int): Total number of training workers.
        data_replicate_size (int): Number of workers holding one data replica.
        optimizer_steps (int): Configured optimizer steps per training iteration.
        pack_strategy (Literal["legacy", "greedy", "balance", "native", "lpt"]): Packing
            strategy. ``legacy`` exactly reproduces the previous greedy pack,
            interleaved DP allocation and optimizer-step grouping; ``greedy`` keeps the
            greedy layout but assigns contiguous pack blocks per DP rank; ``balance``
            length-groups samples before partitioning; ``native`` splits samples evenly
            across DP ranks before packing; ``lpt`` assigns longest sequences first to
            the least-loaded pack that has enough remaining capacity. Defaults to "legacy".
        pack_seed (int | None): Seed for randomized strategies such as ``balance``.
            Defaults to None.

    Ordering is a per-strategy decision: ``legacy`` and ``greedy`` shuffle the visit
    order before packing unless ``XTUNER_DETERMINISTIC`` is set, ``balance`` performs
    its own seeded length grouping, ``native`` keeps the input order, and ``lpt`` shuffles
    finished packs unless ``XTUNER_DETERMINISTIC`` is set.
    """

    def __init__(
        self,
        pack_max_length: int,
        world_size: int,
        data_replicate_size: int,
        optimizer_steps: int,
        pack_strategy: Literal["legacy", "greedy", "balance", "native", "lpt"] = "legacy",
        pack_seed: int | None = None,
    ):
        self.pack_max_length = pack_max_length
        self.world_size = world_size
        self.data_replicate_size = data_replicate_size
        self.optimizer_steps = optimizer_steps
        self.dp_size = self.world_size // self.data_replicate_size
        self.pack_seed = pack_seed
        self.split_size = 1024
        strategy_map = {
            "legacy": self._legacy_pack,
            "greedy": self._greedy_pack,
            "balance": self._balance_pack,
            "native": self._native_pack,
            "lpt": self._lpt_pack,
        }
        if pack_strategy not in strategy_map:
            raise ValueError(f"Unknown packing strategy: {pack_strategy}")
        self._impl = strategy_map[pack_strategy]

    def pack(self, data_lengths: Sequence[int]) -> tuple[PackedDataIndices, int]:
        """Return the packed plan ``[dp][optimizer_step][pack][sample_index]``
        and the padding token count.

        Args:
            data_lengths (Sequence[int]): Training token length of every sample.

        Returns:
            tuple[PackedDataIndices, int]: The packed plan and the total padding tokens
            across all packs (including empty all-padding packs).

        Raises:
            ValueError: A single sample exceeds ``pack_max_length``.
        """
        if not data_lengths:
            return [], 0
        for data_index, data_length in enumerate(data_lengths):
            if data_length > self.pack_max_length:
                raise ValueError(
                    f"Single sample {data_index} seq len {data_length} exceeds pack_max_length {self.pack_max_length}"
                )

        data_indices = list(range(len(data_lengths)))
        packed_data_indices = self._impl(data_indices, data_lengths)
        padding_tokens = self._count_padding_tokens(packed_data_indices, data_lengths)
        return packed_data_indices, padding_tokens

    def _legacy_pack(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        if not XTUNER_DETERMINISTIC:
            random.shuffle(data_indices)
        # Zip is positional, so lengths must follow the shuffled indices.
        num_tokens = [data_lengths[index] for index in data_indices]
        total_pack_indices = get_greedy_pack_infos(data_indices, num_tokens, self.pack_max_length)
        # Interleaved DP allocation over a DP-multiple pack list, then per-rank
        # sequential optimizer-step grouping: this is the exact schedule the previous
        # controller + `iters_per_step` worker regrouping produced.
        pad_num = math.ceil(len(total_pack_indices) / self.dp_size) * self.dp_size - len(total_pack_indices)
        total_pack_indices.extend([[] for _ in range(pad_num)])

        packs_per_dp = len(total_pack_indices) // self.dp_size
        iters_per_step = max(1, math.ceil(packs_per_dp / self.optimizer_steps))
        actual_optimizer_steps = math.ceil(packs_per_dp / iters_per_step)

        packed_data_indices: PackedDataIndices = []
        for dp_rank in range(self.dp_size):
            dp_pack_indices = total_pack_indices[dp_rank :: self.dp_size]
            packed_data_indices.append(
                [
                    dp_pack_indices[step * iters_per_step : (step + 1) * iters_per_step]
                    for step in range(actual_optimizer_steps)
                ]
            )
        return packed_data_indices

    def _greedy_pack(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        if not XTUNER_DETERMINISTIC:
            random.shuffle(data_indices)
        # Zip is positional, so lengths must follow the shuffled indices.
        num_tokens = [data_lengths[index] for index in data_indices]
        total_pack_indices = get_greedy_pack_infos(data_indices, num_tokens, self.pack_max_length)
        pad_num = math.ceil(len(total_pack_indices) / self.dp_size) * self.dp_size - len(total_pack_indices)
        total_pack_indices.extend([[] for _ in range(pad_num)])

        packs_per_dp = len(total_pack_indices) // self.dp_size
        if packs_per_dp < self.optimizer_steps:
            iters_per_step = 1
            actual_optimizer_steps = packs_per_dp
        else:
            iters_per_step = math.ceil(packs_per_dp / self.optimizer_steps)
            actual_optimizer_steps = math.ceil(packs_per_dp / iters_per_step)

        packed_data_indices: PackedDataIndices = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(self.dp_size)
        ]
        for dp_rank in range(self.dp_size):
            for step_idx in range(actual_optimizer_steps):
                start_idx = dp_rank * packs_per_dp + step_idx * iters_per_step
                end_idx = min(start_idx + iters_per_step, packs_per_dp * (dp_rank + 1))
                packed_data_indices[dp_rank][step_idx] = total_pack_indices[start_idx:end_idx]
        return packed_data_indices

    def _balance_pack(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        seed = self.pack_seed if self.pack_seed is not None else _BALANCE_DEFAULT_SEED
        torch_generator = torch.Generator().manual_seed(seed)
        random_generator = random.Random(seed)
        grouped_indices = get_length_grouped_indices(
            max_lengths=list(data_lengths),
            group_batch_size=len(data_indices),
            group_size=self.dp_size,
            torch_generator=torch_generator,
            random_generator=random_generator,
        )

        partitioned_indices: list[list[PackIndices]] = [
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

    def _native_pack(
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

    def _lpt_pack(
        self,
        data_indices: list[int],
        data_lengths: Sequence[int],
    ) -> PackedDataIndices:
        num_tokens = [data_lengths[index] for index in data_indices]
        min_bins = self.optimizer_steps * self.dp_size
        cap_bins = math.ceil(math.ceil(sum(num_tokens) / self.pack_max_length) / self.dp_size) * self.dp_size
        packs = get_lpt_pack_infos(
            data_indices,
            num_tokens,
            self.pack_max_length,
            max(min_bins, cap_bins),
            self.dp_size,
        )
        # Length-sorted assignment correlates pack order with sequence length. Shuffle finished
        # packs so optimizer steps are not length-stratified.
        if not XTUNER_DETERMINISTIC:
            random.shuffle(packs)
        return [
            _partition_packs(packs[dp_rank :: self.dp_size], self.optimizer_steps) for dp_rank in range(self.dp_size)
        ]

    def _pack_indices(self, data_indices: Sequence[int], data_lengths: Sequence[int]) -> OptimizerStepPackIndices:
        scheduled_lengths = [self._get_scheduled_length(data_index, data_lengths) for data_index in data_indices]
        if sum(scheduled_lengths) > self.pack_max_length:
            packs = get_greedy_pack_infos(data_indices, scheduled_lengths, self.pack_max_length)
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
                    pack_tokens = sum(data_lengths[data_index] for data_index in pack_indices)
                    if pack_tokens > self.pack_max_length:
                        raise ValueError(
                            f"Packed samples {pack_indices} have {pack_tokens} tokens, "
                            f"exceeding pack_max_length {self.pack_max_length}"
                        )
                    scheduled_tokens += pack_tokens

        if sorted(seen_indices) != list(range(len(data_lengths))):
            raise RuntimeError("Packing plan must contain every data index exactly once")
        return total_packs * self.pack_max_length - scheduled_tokens
