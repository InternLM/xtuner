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
  * Save / load are routed through :func:`reconstruct_full_tensor` (this module) and the LoadSpec
    machinery, neither of which depends on ``shard_order``.

The reconstruction algorithm and its rationale are documented inline on
:func:`reconstruct_full_tensor`.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard


__all__ = [
    "InterleavedShard",
    "Run",
    "compute_runs",
    "has_interleaved_placement",
    "reconstruct_full_tensor",
]


class Run(NamedTuple):
    """One contiguous run of global indices that the current rank owns on the
    sharded dim.

    Used by both the HF save path (build per-run WriteItems / per-run slices) and the HF load
    path (per-run narrow + copy from the loaded global tensor).

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


def has_interleaved_placement(dt: torch.Tensor) -> bool:
    """True if ``dt`` is a DTensor whose placements include a strided shard
    that cannot be reduced to a valid ShardOrder — i.e. our per-expert column
    parallel layout.

    Detection strategy:

      * torch >= 2.10: check ``DTensorSpec.shard_order is None``. The auto-derivation returns
        ``None`` whenever an internal ``_StridedShard`` placement has no consistent
        ``split_factor`` insertion position (exactly our case).
      * torch < 2.10: that attribute does not exist, so fall back to a structural scan —
        look for any ``_StridedShard`` whose position+sf cannot match the cumulative mesh
        sizes to its right.
    """
    if not isinstance(dt, DTensor):
        return False
    shard_order = getattr(dt._spec, "shard_order", _SENTINEL)
    if shard_order is not _SENTINEL:
        return shard_order is None
    # Fallback for torch < 2.10: replicate the carving-order insertion check.
    return _placement_chain_unsupported(dt.placements, dt.device_mesh)


# Marker used to distinguish "attribute missing" (older torch) vs "attribute is None"
# (the case we care about on 2.10+).
_SENTINEL = object()


def _placement_chain_unsupported(placements, mesh) -> bool:
    """Right-to-left insertion check, identical to torch 2.10's
    ``_maybe_convert_StridedShard_to_shard_order``.

    Returns ``True`` iff any
    ``_StridedShard`` cannot be slotted into a consistent carving order.
    """
    tensor_dim_to_order: dict[int, list[int]] = {}
    for mesh_dim in reversed(range(len(placements))):
        p = placements[mesh_dim]
        if not isinstance(p, (Shard, _StridedShard)):
            continue
        order = tensor_dim_to_order.setdefault(p.dim, [])
        sf = p.split_factor if isinstance(p, _StridedShard) else 1
        accumulated = 1
        inserted = False
        for position in range(len(order) + 1):
            if accumulated == sf:
                order.insert(position, mesh_dim)
                inserted = True
                break
            if position < len(order):
                accumulated *= mesh.size(order[position])
        if not inserted:
            return True
    return False


def _strided_indices(placement, curr_size: int, num_chunks: int, rank: int) -> list[int]:
    """Return the list of indices the given rank owns under a ``_StridedShard``
    placement.

    Compatible with both torch 2.9 (no ``return_first_offset`` kwarg, only contiguous offset
    returned) and torch 2.10+ (full index list available). For 2.9 we replicate the formula
    derived from ``_StridedShard._split_tensor``: rank ``r`` owns chunks ``r, r+M, r+2M, …`` of
    the ``M*sf``-way split, each chunk being ``N / (M*sf)`` elements wide.
    """
    sf = placement.split_factor
    total_split = num_chunks * sf
    chunk_size = curr_size // total_split
    if chunk_size * total_split != curr_size:
        raise NotImplementedError(
            f"_strided_indices: uneven sharding (curr_size={curr_size}, "
            f"num_chunks={num_chunks}, split_factor={sf}) is not yet supported."
        )
    indices: list[int] = []
    for j in range(sf):
        chunk_start = (j * num_chunks + rank) * chunk_size
        indices.extend(range(chunk_start, chunk_start + chunk_size))
    return indices


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


def _is_real_strided(placement, mesh_dim: int) -> bool:
    """True iff ``placement`` is a real strided shard whose data layout
    actually requires the interleaved gather+scatter algorithm.

    Excludes FSDP-prepended labels.
    """
    return (
        isinstance(placement, _StridedShard)
        and placement.split_factor > 1
        and not _is_fsdp_prepended_strided(placement, mesh_dim)
    )


def reconstruct_full_tensor(dt: DTensor) -> torch.Tensor:
    """Reconstruct the global tensor from a DTensor's local data, even when the
    spec contains placements that PyTorch's ``redistribute`` cannot handle
    (``shard_order=None``).

    Why a custom routine: ``DTensor.full_tensor()`` goes through ``redistribute`` which asserts
    ``shard_order is not None`` in torch 2.10. For our ``(Shard, InterleavedShard)`` placement
    that assert fires. We bypass redistribute by emitting collectives directly.

    Algorithm:

      1. **Phase 1 — undo FSDP-prepended _StridedShard (mesh_dim 0) as plain Shard.** FSDP2
         actually chunks the parameter contiguously (``_chunk_with_empty``) regardless of the
         strided label. So the right undo is a plain ``all_gather`` along the FSDP mesh dim.
         After this phase every rank holds the pre-FSDP local.

      2. **Phase 2 — undo remaining placements in REVERSE mesh-dim order:**

         * ``InterleavedShard`` (= real strided): ``all_gather`` along the placement's mesh dim,
           then scatter the gathered chunks back to their correct global positions using
           ``_local_shard_size_and_offset(return_first_offset=False)``.
         * Plain ``Shard``: ``all_gather`` and concatenate.

         The reverse direction is essential because ``InterleavedShard.split_factor`` is defined
         relative to the size of the tensor *after* the placements to its right have already
         been undone. Doing TP undo before EP undo keeps the sf math consistent.

    Returns:
        torch.Tensor: the global tensor materialized on every rank. Dtype and device match
        ``dt._local_tensor``.
    """
    if not isinstance(dt, DTensor):
        raise TypeError(f"reconstruct_full_tensor expects a DTensor, got {type(dt).__name__}")

    mesh = dt.device_mesh
    placements = list(dt.placements)
    # Make sure the working buffer is contiguous so all_gather copies see a well-defined layout.
    result = dt._local_tensor.contiguous()

    # Phase 1: FSDP-prepended _StridedShard at mesh_dim 0 → plain gather.
    for mesh_dim, placement in enumerate(placements):
        if not _is_fsdp_prepended_strided(placement, mesh_dim):
            continue
        result = _all_gather_plain(result, placement.dim, mesh.get_group(mesh_dim))

    # Phase 2: remaining placements in reverse mesh-dim order.
    for mesh_dim in reversed(range(len(placements))):
        placement = placements[mesh_dim]
        if not isinstance(placement, (Shard, _StridedShard)):
            continue
        if _is_fsdp_prepended_strided(placement, mesh_dim):
            continue  # already handled in Phase 1
        if _is_real_strided(placement, mesh_dim):
            result = _undo_strided(result, placement, mesh, mesh_dim)
        else:
            # Plain Shard or _StridedShard with sf == 1 (degenerate).
            result = _all_gather_plain(result, placement.dim, mesh.get_group(mesh_dim))

    return result


# ---------------------------------------------------------------------------
# Internal collective helpers
# ---------------------------------------------------------------------------


def _all_gather_plain(local: torch.Tensor, tensor_dim: int, group) -> torch.Tensor:
    """``all_gather_tensor`` along ``tensor_dim`` then materialize the async
    wrapper."""
    gathered = funcol.all_gather_tensor(local, gather_dim=tensor_dim, group=group)
    if isinstance(gathered, funcol.AsyncCollectiveTensor):
        gathered = gathered.wait()
    return gathered


def compute_runs(dt: DTensor) -> list[Run]:
    """Compute the contiguous-run decomposition of this rank's share of the
    global tensor.

    Accumulates the global indices the current rank owns on the sharded dim. Adjacent indices
    are grouped into ``Run`` records so the caller can do per-run narrow + copy without ever
    materializing the full index tensor.

    FSDP prepends its placement at mesh dim 0, but semantically it shards the already EP/TP-local
    parameter. So for index computation we apply non-FSDP placements first and the FSDP-prepended
    shard last, mirroring ``reconstruct_full_tensor`` which undoes FSDP first.

    Restricted to single-dim sharding (the only layout xtuner currently uses for fused MoE
    weights). For multi-dim sharding a Cartesian-product extension is straightforward.
    """
    if not isinstance(dt, DTensor):
        raise TypeError(f"compute_runs expects a DTensor, got {type(dt).__name__}")

    mesh = dt.device_mesh
    global_shape = tuple(dt.shape)
    ndim = len(global_shape)

    fsdp_prepended = []
    placement_order = []
    for mesh_dim, p in enumerate(dt.placements):
        item = (mesh_dim, p)
        if _is_fsdp_prepended_strided(p, mesh_dim):
            fsdp_prepended.append(item)
        else:
            placement_order.append(item)

    dim_indices: dict[int, list[int]] = {}
    for mesh_dim, p in placement_order + fsdp_prepended:
        if not isinstance(p, (Shard, _StridedShard)):
            continue
        d = p.dim
        prev = dim_indices.get(d)
        prev_size = len(prev) if prev is not None else global_shape[d]
        if _is_real_strided(p, mesh_dim):
            new_idx = _strided_indices(p, prev_size, mesh.size(mesh_dim), mesh.get_local_rank(mesh_dim))
        else:
            size, offset = Shard(d)._local_shard_size_and_offset(  # type: ignore[attr-defined]
                prev_size, mesh.size(mesh_dim), mesh.get_local_rank(mesh_dim)
            )
            new_idx = list(range(offset, offset + size))
        dim_indices[d] = new_idx if prev is None else [prev[i] for i in new_idx]

    sharded_dims = sorted(dim_indices.keys())
    assert sharded_dims == [0], f"compute_runs currently handles dim-0 sharding only, got {sharded_dims}"

    indices = dim_indices[0]
    if not indices:
        return []

    runs: list[Run] = []
    run_start = indices[0]
    run_len = 1
    local_start = 0
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            run_len += 1
            continue
        runs.append(
            Run(
                global_offset=(run_start,) + (0,) * (ndim - 1),
                sizes=(run_len,) + global_shape[1:],
                local_start=local_start,
                local_size=run_len,
            )
        )
        local_start += run_len
        run_start = indices[i]
        run_len = 1
    runs.append(
        Run(
            global_offset=(run_start,) + (0,) * (ndim - 1),
            sizes=(run_len,) + global_shape[1:],
            local_start=local_start,
            local_size=run_len,
        )
    )
    return runs


def _undo_strided(
    local: torch.Tensor,
    placement,
    mesh,
    mesh_dim: int,
) -> torch.Tensor:
    """``all_gather`` + scatter for a strided placement.

    Each rank in the mesh dim group holds a strided chunk per ``placement``'s spec. After
    ``all_gather`` the result is the concatenation of those chunks in rank order. To recover
    the original layout we re-index each rank's chunk back to its true positions using
    ``_local_shard_size_and_offset(return_first_offset=False)`` which returns the global
    indices the rank owned within the post-undo tensor.
    """
    tensor_dim = placement.dim
    mesh_size = mesh.size(mesh_dim)
    group = mesh.get_group(mesh_dim)

    gathered = _all_gather_plain(local, tensor_dim, group)
    current_size = gathered.shape[tensor_dim]

    all_indices: list[int] = []
    for r in range(mesh_size):
        all_indices.extend(_strided_indices(placement, current_size, mesh_size, r))

    indices_tensor = torch.tensor(all_indices, device=gathered.device, dtype=torch.long)
    new_result = torch.empty_like(gathered)
    new_result.index_copy_(tensor_dim, indices_tensor, gathered)
    return new_result
