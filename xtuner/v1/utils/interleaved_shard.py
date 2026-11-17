"""Per-expert column parallel placement and helpers.

This module introduces ``InterleavedShard``, a custom :class:`Placement` for fused MoE weights
where TP needs to cut ``out_features`` *inside* every local expert. The layout cannot be
expressed by torch's built-in ``Shard`` (which would either give each TP rank one whole expert
or break expert boundaries). ``InterleavedShard`` does exactly per-expert column parallel.

It is intentionally a subclass of ``_StridedShard`` so:

  * FSDP2 (``fully_shard``) recognizes it via ``isinstance(..., _StridedShard)`` and prepends
    its own placement on the same tensor dim correctly.
  * All ``_local_shard_size_and_offset``/``_split_tensor``/``_to_replicate_tensor`` semantics
    come from ``_StridedShard`` for free.

The cost is that PyTorch cannot reduce ``(Shard, InterleavedShard)`` (i.e. the strided shard
sitting at the *rightmost* mesh dim) to a ``ShardOrder``. Any code path that relies on
``DTensorSpec.shard_order`` — most notably ``DTensor.redistribute`` / ``DTensor.full_tensor`` —
crashes on such DTensors. xtuner deliberately bypasses those paths:

  * Forward / backward read ``weight.to_local()`` so the op dispatcher is never invoked on
    InterleavedShard parameters.
  * HF save / load are routed through LoadSpec plans, which do not depend on ``shard_order``.
    :func:`reconstruct_full_tensor` is a convenience wrapper over the same save-plan executor.

The reconstruction algorithm and its rationale are documented inline on
:func:`reconstruct_full_tensor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TypeGuard

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import Placement, _StridedShard


__all__ = [
    "InterleavedShard",
    "Run",
    "RuntimeLayout",
    "RuntimeShard",
    "reconstruct_full_tensor",
]


class Run(NamedTuple):
    """One contiguous run of global indices that the current rank owns on the
    sharded dim.

    Used by the DCP planners to build per-run WriteItems / ReadItems. HF load/save derives
    equivalent run ownership from ``ShardDescriptor`` inside its plans.

    Args:
        global_offset (tuple[int, ...]): Offset into the global tensor where this run begins.
            All non-sharded dims are 0.
        sizes (tuple[int, ...]): Chunk size on each tensor dim for this run.
        local_start (int): Row in the local tensor where this run begins.
        local_size (int): Number of rows in this run (== sizes on the sharded dim).
    """

    global_offset: tuple[int, ...]
    sizes: tuple[int, ...]
    local_start: int
    local_size: int


@dataclass(frozen=True)
class RuntimeShard:
    """Stable description of one DTensor partition.

    ``RuntimeLayout`` converts PyTorch placements into these records once so
    checkpoint, FP8, and model code do not depend on private placement types.
    """

    dim: int
    group: dist.ProcessGroup
    world_size: int
    rank: int
    interleave_factor: int = 1

    def local_indices(self, dim_size: int) -> list[int]:
        """Return positions owned by this rank in the current shard input."""
        if self.interleave_factor == 1:
            base_size, remainder = divmod(dim_size, self.world_size)
            local_size = base_size + int(self.rank < remainder)
            offset = self.rank * base_size + min(self.rank, remainder)
            return list(range(offset, offset + local_size))

        total_split = self.world_size * self.interleave_factor
        run_size, remainder = divmod(dim_size, total_split)
        if remainder:
            raise NotImplementedError(
                "Even interleave requires the current dimension to be divisible by "
                f"world_size * interleave_factor, got {dim_size} % "
                f"({self.world_size} * {self.interleave_factor}) != 0"
            )
        indices: list[int] = []
        for run_index in range(self.interleave_factor):
            run_start = (run_index * self.world_size + self.rank) * run_size
            indices.extend(range(run_start, run_start + run_size))
        return indices


@dataclass(frozen=True)
class RuntimeLayout:
    """Project-owned boundary around PyTorch DTensor layout details.

    ``from_dtensor`` is the single conversion point for private
    ``_StridedShard`` semantics, placement carving order, and FSDP's prepended
    placement. Consumers use only the normalized shard records and global
    coordinate runs exposed here.
    """

    global_shape: tuple[int, ...]
    ordered_shards: tuple[RuntimeShard, ...]

    @staticmethod
    def is_sharded_placement(placement: Placement) -> TypeGuard[Shard | _StridedShard]:
        """Hide PyTorch's version-dependent strided-shard hierarchy."""
        return isinstance(placement, (Shard, _StridedShard))

    @classmethod
    def from_dtensor(cls, tensor: DTensor) -> RuntimeLayout:
        mesh = tensor.device_mesh
        placements = tensor.placements

        # Mirror PyTorch's carving-order normalization without importing its
        # version-dependent private helper. A valid chain is equivalent to a
        # sequence of continuous shards in the derived order.
        tensor_dim_to_order: dict[int, list[int]] = {}
        chain_supported = True
        for mesh_dim in reversed(range(len(placements))):
            placement = placements[mesh_dim]
            if not cls.is_sharded_placement(placement):
                continue
            order = tensor_dim_to_order.setdefault(placement.dim, [])
            split_factor = placement.split_factor if isinstance(placement, _StridedShard) else 1
            accumulated = 1
            for position in range(len(order) + 1):
                if accumulated == split_factor:
                    order.insert(position, mesh_dim)
                    break
                if position < len(order):
                    accumulated *= mesh.size(order[position])
            else:
                chain_supported = False
                break

        normalized: list[tuple[int, int, int]] = []
        if chain_supported:
            for tensor_dim in sorted(tensor_dim_to_order):
                normalized.extend((mesh_dim, tensor_dim, 1) for mesh_dim in tensor_dim_to_order[tensor_dim])
        else:
            # ExpertTP's unsupported chain is semantically model-parallel
            # placements first, then FSDP's bookkeeping placement.
            fsdp_prepended: list[tuple[int, int, int]] = []
            for mesh_dim, placement in enumerate(placements):
                if not cls.is_sharded_placement(placement):
                    continue
                is_fsdp_prepended = _is_fsdp_prepended_strided(placement, mesh_dim)
                item = (
                    mesh_dim,
                    placement.dim,
                    placement.split_factor if isinstance(placement, _StridedShard) and not is_fsdp_prepended else 1,
                )
                (fsdp_prepended if is_fsdp_prepended else normalized).append(item)
            normalized.extend(fsdp_prepended)

        return cls(
            global_shape=tuple(tensor.shape),
            ordered_shards=tuple(
                RuntimeShard(
                    dim=tensor_dim,
                    group=mesh.get_group(mesh_dim),
                    world_size=mesh.size(mesh_dim),
                    rank=mesh.get_local_rank(mesh_dim),
                    interleave_factor=interleave_factor,
                )
                for mesh_dim, tensor_dim, interleave_factor in normalized
            ),
        )

    @property
    def is_interleaved(self) -> bool:
        return any(shard.interleave_factor > 1 for shard in self.ordered_shards)

    def shard_size(self, dim: int) -> int:
        shard_size = 1
        for shard in self.ordered_shards:
            if shard.dim == dim:
                shard_size *= shard.world_size
        return shard_size

    def owned_runs(self) -> list[Run]:
        """Return this rank's contiguous runs in global coordinates."""
        dim_indices: dict[int, list[int]] = {}
        for shard in self.ordered_shards:
            previous = dim_indices.get(shard.dim)
            current_size = len(previous) if previous is not None else self.global_shape[shard.dim]
            local_indices = shard.local_indices(current_size)
            dim_indices[shard.dim] = local_indices if previous is None else [previous[i] for i in local_indices]

        sharded_dims = sorted(dim_indices)
        assert sharded_dims == [0], (
            f"RuntimeLayout.owned_runs currently handles dim-0 sharding only, got {sharded_dims}"
        )
        indices = dim_indices[0]
        if not indices:
            return []

        ndim = len(self.global_shape)
        runs: list[Run] = []
        run_start = indices[0]
        run_len = 1
        local_start = 0
        for index, previous_index in zip(indices[1:], indices):
            if index == previous_index + 1:
                run_len += 1
                continue
            runs.append(
                Run(
                    global_offset=(run_start,) + (0,) * (ndim - 1),
                    sizes=(run_len,) + self.global_shape[1:],
                    local_start=local_start,
                    local_size=run_len,
                )
            )
            local_start += run_len
            run_start = index
            run_len = 1
        runs.append(
            Run(
                global_offset=(run_start,) + (0,) * (ndim - 1),
                sizes=(run_len,) + self.global_shape[1:],
                local_start=local_start,
                local_size=run_len,
            )
        )
        return runs


class InterleavedShard(_StridedShard):
    """Per-stripe column-parallel placement for fused MoE weights.

    For a fused weight whose sharded dim contains ``num_local_stripes`` equal-size logical
    stripes per rank, this placement cuts the **inside** of every stripe by ``tp_size`` and
    interleaves the cuts. Each ``(ep, tp)`` rank ends up holding ``num_local_stripes`` runs
    of contiguous rows; consecutive runs are spaced by one full stripe.

    Two common stripe interpretations:

      * **Non-fused MoE weight** (e.g. one projection per expert): one stripe per local expert.
        ``num_local_stripes == num_experts_per_ep``.
      * **Fused MoE weight** (e.g. ``fused_w1w3`` packs ``gate_proj`` and ``up_proj`` per
        expert): one stripe per (expert, fused projection). For ``fused_w1w3`` with 2 projections
        per expert: ``num_local_stripes == num_experts_per_ep * 2``.

    Getting ``num_local_stripes`` wrong silently produces a layout that swaps data between
    fused projections (e.g. ``silu(gate) * up`` becomes ``silu(gate_half) * gate_other_half``),
    so callers must pass the value that matches the HF key concatenation order.

    Internally this is a ``_StridedShard(dim, split_factor=num_local_stripes)``.

    Args:
        dim (int): Tensor dim to shard. For fused MoE weights this is 0.
        num_local_stripes (int): Number of equal-size stripes the per-rank dim contains.
            See class docstring for how to compute this.
    """

    def __init__(self, dim: int, *, num_local_stripes: int):
        super().__init__(dim, split_factor=num_local_stripes)

    @property
    def num_local_stripes(self) -> int:
        return self.split_factor

    def __repr__(self) -> str:
        return f"InterleavedShard(dim={self.dim}, num_local_stripes={self.split_factor})"


def _is_fsdp_prepended_strided(placement, mesh_dim: int) -> bool:
    """Heuristic: a ``_StridedShard`` at mesh dim 0 is FSDP-prepended.

    ``fully_shard`` always prepends its placement at the leftmost mesh dim, and FSDP's actual
    chunking is plain contiguous (``_chunk_with_empty``) despite the strided label. Position
    ``0`` is the most reliable signal because the ``_StridedShard`` subclass identity does not
    survive ``distribute_tensor`` / FSDP2's internal spec construction (C++ layer reconstructs
    a bare ``_StridedShard``).

    This heuristic breaks if a user places an InterleavedShard at mesh dim 0 directly without
    FSDP wrapping. xtuner does not do that — InterleavedShard is always at the TP position.
    """
    return mesh_dim == 0 and isinstance(placement, _StridedShard) and placement.split_factor > 1


def reconstruct_full_tensor(dt: DTensor) -> torch.Tensor:
    """Reconstruct a full runtime tensor through the shared SavePlan executor.

    PyTorch ``DTensor.full_tensor()`` cannot redistribute Expert TP layouts with
    ``shard_order=None``. LoadSpec normalizes those placements into continuous
    and even-interleave descriptors, and its save executor performs the inverse
    collectives for both HF save and this convenience API.

    Returns:
        torch.Tensor: the global tensor materialized on every rank. Dtype and device match
        ``dt._local_tensor``.
    """
    if not isinstance(dt, DTensor):
        raise TypeError(f"reconstruct_full_tensor expects a DTensor, got {type(dt).__name__}")

    from xtuner.v1.utils.load_spec import LoadSpec, unshard_tensors_for_hf_save

    load_spec = LoadSpec.from_tensor(
        name="__full_runtime_tensor__",
        hf_keys=["__full_runtime_tensor__"],
        tensor=dt,
    )
    return unshard_tensors_for_hf_save(
        [dt._local_tensor.contiguous()],
        [load_spec.plan_hf_save()],
    )[0]
