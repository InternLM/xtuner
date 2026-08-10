import math
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import torch
import torch.distributed as dist
import torch.distributed.tensor._utils as dtensor_utils
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch.distributed.tensor import DTensor, Shard

from xtuner.v1.ops.comm.foreach_allgather import foreach_all_gather
from xtuner.v1.utils.device import get_device


def _is_same_process_group(left: dist.ProcessGroup, right: dist.ProcessGroup) -> bool:
    if left is right:
        return True
    return dist.get_process_group_ranks(left) == dist.get_process_group_ranks(right)


class ShardDescriptor(BaseModel):
    """A single partition applied to the fused full tensor.

    The full tensor is obtained by concatenating every ``LoadSpec.global_hf_keys`` along
    ``LoadSpec.fused_dim`` (or taking the sole HF tensor when ``len(global_hf_keys) == 1``).
    Descriptors are applied in order; later descriptors use offsets relative to the sub-tensor produced by all
    earlier descriptors, matching DTensor placement semantics.

    Args:
        dim (int): Tensor dim on which this partition cuts.
        start (int): Inclusive start offset relative to the current sub-tensor.
        end (int): Exclusive end offset relative to the current sub-tensor.
        group (dist.ProcessGroup): Communication group that produced this partition.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    dim: int
    start: int
    end: int
    group: dist.ProcessGroup


def _dtensor_shards(tensor: DTensor) -> list[ShardDescriptor]:
    current_shape = list(tensor.shape)
    shards: list[ShardDescriptor] = []
    for mesh_dim, placement in _ordered_dtensor_placements(tensor):
        if not isinstance(placement, Shard):
            continue

        # DTensor placement order is not always the raw mesh-dim order. FSDP2 can represent right-to-left sharding
        # with _StridedShard, and PyTorch's checkpoint offset helper first expands that into the effective shard
        # order. LoadSpec must preserve the same order so its descriptor intervals match DTensor local tensors.
        #
        # XTuner may initialize modules while the default device is "meta". PyTorch's Shard placement helpers can
        # inherit that default device for temporary shape arithmetic, so force XTuner's real runtime device before
        # calling the helper.
        with torch.device(get_device()):
            local_size, offset = placement._local_shard_size_and_offset(  # type: ignore[attr-defined]
                current_shape[placement.dim],
                tensor.device_mesh.size(mesh_dim),
                tensor.device_mesh.get_local_rank(mesh_dim),
            )
        shards.append(
            ShardDescriptor(
                dim=placement.dim,
                start=offset,
                end=offset + local_size,
                group=tensor.device_mesh.get_group(mesh_dim),
            )
        )
        current_shape[placement.dim] = local_size
    return shards


def _ordered_dtensor_placements(tensor: DTensor) -> list[tuple[int, object]]:
    # PyTorch keeps this helper private and does not expose it in type stubs, but it is the same ordering logic used
    # by `compute_local_shape_and_global_offset`. Access it dynamically so mypy does not reject the private symbol.
    explicit_order_placements = cast(
        Callable[[Any, Any], list[tuple[int, object]]],
        getattr(dtensor_utils, "_explicit_order_placements"),
    )
    return explicit_order_placements(tensor.device_mesh.shape, tensor.placements)


class LoadSlice(BaseModel):
    """A narrow operation in the loaded HF tensor coordinate system.

    Args:
        dim (int): Tensor dimension to narrow.
        start (int): Inclusive start offset in the loaded tensor.
        end (int): Exclusive end offset in the loaded tensor.
    """

    model_config = ConfigDict(extra="forbid")
    dim: int
    start: int
    end: int


class HFLoadPlan(BaseModel):
    """Execution plan for reading HF safetensors into one local tensor.

    Args:
        name (str): Fully-qualified parameter or buffer name on the xtuner side.
        hf_keys (list[str]): HF keys that must be read for this rank.
        fused_dim (int | None): Concatenation dimension when multiple HF keys are loaded.
        slices (list[LoadSlice]): Narrow operations to apply after loading. Offsets are relative to the loaded
            tensor, not the original ``LoadSpec.global_shape``.
        zero_fill (bool): Whether this rank falls entirely in a padded region and should skip checkpoint reads.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    hf_keys: list[str]
    fused_dim: int | None = None
    slices: list[LoadSlice] = Field(default_factory=list)
    zero_fill: bool = False


def _final_intervals(
    global_shape: tuple[int, ...],
    shards: list[ShardDescriptor],
) -> list[tuple[int, int]]:
    intervals = [(0, dim_size) for dim_size in global_shape]
    for shard in shards:
        current_start, _ = intervals[shard.dim]
        intervals[shard.dim] = (current_start + shard.start, current_start + shard.end)
    return intervals


class SaveShardStep(BaseModel):
    """Save-time work item derived from one ``LoadSpec.shards`` entry.

    ``LoadSpec.shards`` is a layout description: each descriptor says how the previous tensor was partitioned.
    Saving needs the inverse operation. ``LoadSpec._save_shard_steps`` converts every shard descriptor into a work
    item that contains the shard itself plus the tensor shapes that existed immediately before that shard was applied.
    The save path then executes these work items in reverse order and batches compatible all-gathers by process group.

    ``load_spec_shard_index`` is only needed when some original shards should stay sharded. RL weight sync preserves
    the EP shard on the fused HF dimension so each EP rank streams only its local expert keys, while later shards such
    as FSDP still need to be all-gathered. Because execution reverses and groups the work items, their list positions
    no longer match ``LoadSpec.shards``. The original index is the stable handle used by the save plan to decide
    which work items to skip and which preserved shards should define the final expected shape.

    Example:
        ``LoadSpec.shards == [ep_shard, fsdp_shard]`` means the full HF tensor was first cut by EP, then the
        EP-local tensor was cut by FSDP. Normal HF save executes ``[fsdp_step, ep_step]`` to rebuild the full tensor.
        RL weight sync can mark ``ep_step`` as preserved, so only the FSDP work item is executed and the result stays
        EP-local.

    Args:
        load_spec_shard_index (int): Index of ``shard`` in the original ``LoadSpec.shards`` list.
        shard (ShardDescriptor): Shard descriptor this save step reverses.
        shape_before_shard (tuple[int, ...]): Runtime tensor shape immediately before ``shard`` was applied.
        unpadded_shape_before_shard (tuple[int, ...]): Checkpoint-visible shape before ``shard`` was applied.
        preserved (bool): Whether this shard should remain applied instead of being all-gathered.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    load_spec_shard_index: int
    shard: ShardDescriptor
    shape_before_shard: tuple[int, ...]
    unpadded_shape_before_shard: tuple[int, ...]
    preserved: bool = False


class HFSavePlan(BaseModel):
    """Execution plan for preparing one runtime tensor for HF safetensors save.

    Args:
        name (str): Fully-qualified parameter or buffer name on the xtuner side.
        hf_keys (list[str]): HF keys represented by the tensor after this plan's pending unshard steps finish.
        global_shape (tuple[int, ...]): Runtime full tensor shape before any shard is applied.
        unpadded_global_shape (tuple[int, ...]): Checkpoint-visible full tensor shape after removing runtime padding.
        fused_dim (int | None): HF key concatenation dim when the underlying ``LoadSpec`` is fused; ``None``
            otherwise.
        distributed_save (bool): Whether non-fused tensors are written only on rank0 and fused keys are split across
            save ranks.
        preserves_shards (bool): Whether the save tensor intentionally remains sharded by some original
            ``LoadSpec.shards`` entries.
        unshard_steps (list[SaveShardStep]): Forward-order shard history with save-time preserved flags.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    hf_keys: list[str]
    global_shape: tuple[int, ...]
    unpadded_global_shape: tuple[int, ...]
    fused_dim: int | None = None
    distributed_save: bool = False
    preserves_shards: bool = False
    unshard_steps: list[SaveShardStep] = Field(default_factory=list)

    def _pending_unshard_steps(self) -> list[SaveShardStep]:
        return [step for step in reversed(self.unshard_steps) if not step.preserved]

    def _preserved_shards(self) -> list[ShardDescriptor]:
        return [step.shard for step in self.unshard_steps if step.preserved]

    def _expected_unsharded_shape(self) -> tuple[int, ...]:
        """Return the save tensor shape after intentionally preserved shards
        remain applied.

        The save path starts from the local tensor and all-gathers every pending shard step. If no shard is preserved,
        the final shape should be ``unpadded_global_shape``. If some shards are preserved, for example an EP shard
        during RL weight sync, the final tensor should still be cut by those preserved shards. This helper applies
        only the preserved shard descriptors to ``unpadded_global_shape`` to compute that expected partially-unsharded
        shape for the final assert.

        Example:
            Suppose the runtime full tensor shape is ``(16, 8)`` because fp8 padding added rows, while
            ``unpadded_global_shape == (14, 8)`` is the shape that should exist in HF. If the preserved EP shard is
            ``ShardDescriptor(dim=0, start=8, end=16)``, that shard owns runtime rows ``[8, 16)``. The last two rows
            are padding-only in HF coordinates, so the checkpoint-visible interval is clipped to ``[8, 14)`` and the
            expected preserved tensor shape is ``(6, 8)``. If a shard were ``[14, 16)``, both boundaries would clip to
            ``14`` and the expected shape on that rank would be ``(0, 8)``.

        Returns:
            tuple[int, ...]: Expected shape after the preserved shards are still applied.
        """
        effective_shape = list(self.unpadded_global_shape)
        for shard in self._preserved_shards():
            # ShardDescriptor offsets are defined against the runtime shape, which may include XTuner-only padding.
            # Clip preserved shard boundaries to the currently visible unpadded shape before computing its length.
            clipped_start = min(shard.start, effective_shape[shard.dim])
            clipped_end = min(shard.end, effective_shape[shard.dim])
            effective_shape[shard.dim] = max(0, clipped_end - clipped_start)
        return tuple(effective_shape)


class _SaveUnshardGroup(NamedTuple):
    """One compatible foreach all-gather batch in the save unshard loop.

    ``tensors`` and ``shard_steps`` are the grouped work payload. ``tensor_indices`` is kept only because the gathered
    tensors must be written back to their original positions in the bucket after the collective finishes.
    """

    tensor_indices: list[int]
    tensors: list[torch.Tensor]
    shard_steps: list[SaveShardStep]


def unshard_tensors_for_hf_save(
    tensors: list[torch.Tensor],
    save_plans: list[HFSavePlan],
) -> list[torch.Tensor]:
    """Run the all-gathers needed to turn local runtime tensors into
    checkpoint-visible save tensors.

    Args:
        tensors (list[torch.Tensor]): Local runtime tensors to unshard.
        save_plans (list[HFSavePlan]): HF save plans corresponding to ``tensors``.

    Returns:
        list[torch.Tensor]: Tensors after all pending save unshard steps have been executed.
    """
    assert len(tensors) == len(save_plans), "Internal error: save tensor and plan count mismatch"
    if not tensors:
        return []

    # Shallow-copy the list, not the tensors. Entries with no gather work can be returned as-is, while entries
    # that do need all-gather are overwritten in this working list with their gathered tensor.
    tensor_list = list(tensors)

    # Convert each tensor's forward shard history into the save-time work queue. Save must undo shards from
    # inner to outer, so the steps are reversed; preserved shards, such as an EP shard kept local for RL weight
    # sync, are removed from the queue and only used later to compute the expected partially-unsharded shape.

    # Example:
    #   tensor A: [ep_a(index=0), fsdp_a(index=1)], preserved {0} -> pending [fsdp_a]
    #   tensor B: [ep_b(index=0), fsdp_b(index=1)], preserved {} -> pending [fsdp_b, ep_b]
    #   tensor C: [fsdp_c(index=0)], preserved {} -> pending [fsdp_c]
    #   tensor D: [tp_d(index=0)], preserved {} -> pending [tp_d]
    #   tensor E: [ep_e(index=0)], preserved {0} -> pending []
    # This produces one pending queue per tensor; the loop below consumes compatible queue heads by group.
    pending_shard_steps_list = [save_plan._pending_unshard_steps() for save_plan in save_plans]

    while True:
        # Build one all-gather round. For one tensor, reverse-unshard steps must run one by one: if a local
        # tensor needs to undo FSDP and then EP, the EP gather must use the tensor produced by the FSDP gather.
        # `_build_ready_save_unshard_groups` consumes `pending_shard_steps_list` gradually. For example, a queue
        # `[fsdp_step, ep_step]` contributes `fsdp_step` in the first round; after its gathered tensor is written
        # back, the next loop consumes `ep_step`. Independent tensors with compatible group/dtype can still be
        # batched together in each round.
        #
        # With the A-E example above, round 1 consumes fsdp_a/fsdp_b/fsdp_c together if they share group/dtype,
        # and consumes tp_d in another group. tensor E contributes no work. Round 2 can then consume ep_b, because
        # ep_b must use tensor B after fsdp_b has been gathered and written back.
        unshard_groups = _build_ready_save_unshard_groups(tensor_list, pending_shard_steps_list)
        if not unshard_groups:
            break

        for unshard_group in unshard_groups:
            gathered_tensors = _foreach_all_gather_save_shards(
                unshard_group.tensors,
                unshard_group.shard_steps,
            )
            for index, gathered_tensor in zip(unshard_group.tensor_indices, gathered_tensors, strict=True):
                tensor_list[index] = gathered_tensor

    for tensor, save_plan in zip(tensor_list, save_plans, strict=True):
        expected_shape = save_plan._expected_unsharded_shape()
        assert tuple(tensor.shape) == expected_shape, (
            f"Saved tensor shape {tuple(tensor.shape)} is incompatible with HFSavePlan global_shape="
            f"{save_plan.global_shape} and unpadded_global_shape={save_plan.unpadded_global_shape} "
            f"for {save_plan.name}"
        )
    return tensor_list


def _build_ready_save_unshard_groups(
    tensor_list: list[torch.Tensor],
    pending_shard_steps_list: list[list[SaveShardStep]],
) -> list[_SaveUnshardGroup]:
    """Build foreach all-gather groups for the save unshard steps that are
    ready to run now."""
    unshard_groups: list[_SaveUnshardGroup] = []
    group_list: list[dist.ProcessGroup] = []
    dtype_list: list[torch.dtype] = []

    for index, pending_shard_steps in enumerate(pending_shard_steps_list):
        if not pending_shard_steps:
            # This tensor has no gather work in the current save context. Common cases are unsharded tensors or
            # tensors whose remaining shards are intentionally preserved, e.g. an EP-only tensor when this pass is
            # only gathering FSDP shards.
            continue

        # Consume one dependency-ready head step from this tensor and place it into a compatible foreach group.
        shard_step = pending_shard_steps.pop(0)
        shard_group = shard_step.shard.group
        tensor_dtype = tensor_list[index].dtype
        for group_index, (existing_group, existing_dtype) in enumerate(zip(group_list, dtype_list, strict=True)):
            if tensor_dtype == existing_dtype and _is_same_process_group(existing_group, shard_group):
                unshard_groups[group_index].tensor_indices.append(index)
                unshard_groups[group_index].tensors.append(tensor_list[index])
                unshard_groups[group_index].shard_steps.append(shard_step)
                break
        else:
            group_list.append(shard_group)
            dtype_list.append(tensor_dtype)
            unshard_groups.append(
                _SaveUnshardGroup(
                    tensor_indices=[index],
                    tensors=[tensor_list[index]],
                    shard_steps=[shard_step],
                )
            )

    return unshard_groups


def _foreach_all_gather_save_shards(
    tensor_list: list[torch.Tensor],
    shard_steps: list[SaveShardStep],
) -> list[torch.Tensor]:
    assert len(tensor_list) == len(shard_steps), "Internal error: tensor and shard-step count mismatch"
    assert tensor_list, "Internal error: empty save all-gather group"
    group = shard_steps[0].shard.group
    assert all(_is_same_process_group(group, shard_step.shard.group) for shard_step in shard_steps), (
        "Internal error: save all-gather group contains different process groups"
    )
    padded_tensor_list = [
        _pad_tensor_for_save_shard(tensor, shard_step)
        for tensor, shard_step in zip(tensor_list, shard_steps, strict=True)
    ]
    gathered_chunks_list = foreach_all_gather(padded_tensor_list, group)
    return [
        _merge_gathered_save_shard(gathered_chunks, shard_step)
        for gathered_chunks, shard_step in zip(gathered_chunks_list, shard_steps, strict=True)
    ]


def _pad_tensor_for_save_shard(tensor: torch.Tensor, shard_step: SaveShardStep) -> torch.Tensor:
    world_size = dist.get_world_size(group=shard_step.shard.group)
    dim = shard_step.shard.dim
    shard_dim_size = shard_step.shape_before_shard[dim]
    padded_local_size = math.ceil(shard_dim_size / world_size)
    pad_len = padded_local_size - tensor.shape[dim]
    assert pad_len >= 0, (
        f"Local tensor shape {tuple(tensor.shape)} exceeds padded shard size {padded_local_size} "
        f"for {shard_step.shard} in save path"
    )
    if not pad_len:
        return tensor

    pad_list = [0] * (2 * tensor.dim())
    pad_idx = 2 * (tensor.dim() - 1 - dim)
    pad_list[pad_idx + 1] = pad_len
    return F.pad(tensor, pad_list)


def _merge_gathered_save_shard(
    gathered_chunks: list[torch.Tensor],
    shard_step: SaveShardStep,
) -> torch.Tensor:
    dim = shard_step.shard.dim
    gathered_tensor = torch.cat(gathered_chunks, dim=dim)
    return gathered_tensor.narrow(dim, 0, shard_step.unpadded_shape_before_shard[dim]).contiguous()


class LoadSpec(BaseModel):
    """Mapping between a local param / buffer and its HF checkpoint keys.

    Args:
        name (str): Fully-qualified parameter or buffer name on the xtuner side.
        global_hf_keys (list[str]): Full HF key list. Concatenating these keys along ``fused_dim`` produces the
            full tensor before local sharding.
        global_shape (tuple[int, ...]): Shape of the fused full tensor before any ``shards`` partition is applied.
            This is the runtime shape and may include padding introduced by XTuner float8 weights.
        fused_dim (int | None): HF key concatenation dim when ``len(global_hf_keys) > 1``; ``None`` otherwise.
        shards (list[ShardDescriptor]): Partitions applied to the full tensor in outer-to-inner order.
        origin_shape (tuple[int, ...] | None): Checkpoint-visible global shape after trimming runtime-only padding.
            The current caller sets it from fp8 tensor metadata; ``None`` means the runtime shape is already the
            checkpoint shape.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    global_hf_keys: list[str]
    global_shape: tuple[int, ...]
    fused_dim: int | None = None
    shards: list[ShardDescriptor] = Field(default_factory=list)
    origin_shape: tuple[int, ...] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_fused(self) -> bool:
        return len(self.global_hf_keys) > 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_sharded(self) -> bool:
        return bool(self.shards)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unpadded_global_shape(self) -> tuple[int, ...]:
        return tuple(self.origin_shape or self.global_shape)

    @classmethod
    def from_tensor(
        cls,
        *,
        name: str,
        hf_keys: list[str],
        tensor: torch.Tensor | DTensor,
        origin_shape: tuple[int, ...] | None = None,
    ) -> "LoadSpec":
        """Build a load spec from a runtime tensor and its HF key mapping.

        This is the conversion boundary from PyTorch runtime layout to ``LoadSpec``. It derives the fused HF
        dimension from ``hf_keys`` and converts DTensor ``Shard`` placements into ``ShardDescriptor`` entries. It does
        not inspect XTuner fp8 wrapper types; callers should pass ``origin_shape`` when runtime-only padding makes the
        checkpoint-visible shape smaller than the runtime shape.

        Args:
            name (str): Fully-qualified parameter or buffer name on the xtuner side.
            hf_keys (list[str]): HF key list corresponding to ``tensor``.
            tensor (torch.Tensor | DTensor): Runtime tensor whose DTensor placements should be captured.
            origin_shape (tuple[int, ...] | None): Optional checkpoint-visible shape after trimming runtime-only
                padding.

        Returns:
            LoadSpec: Spec derived from the runtime tensor layout.
        """
        global_hf_keys = list(hf_keys)
        return cls(
            name=name,
            global_hf_keys=global_hf_keys,
            global_shape=tuple(tensor.shape),
            fused_dim=0 if len(global_hf_keys) > 1 else None,
            shards=_dtensor_shards(tensor) if isinstance(tensor, DTensor) else [],
            origin_shape=origin_shape,
        )

    def plan_hf_load(self) -> HFLoadPlan:
        """Build a safetensors read plan from this layout spec.

        Runtime-only padding currently comes from XTuner float8 weights. In that case, ``origin_shape`` is used as
        the checkpoint-visible full tensor shape, while ``global_shape`` and ``shards`` still describe the padded
        runtime layout that this rank owns.

        Returns:
            HFLoadPlan: The selected HF keys and loaded-tensor-relative slices for this rank.
        """
        effective_intervals = self._effective_intervals_for_shards(self.shards)
        if effective_intervals is None:
            return HFLoadPlan(name=self.name, hf_keys=[], fused_dim=self.fused_dim, zero_fill=True)

        loaded_starts = [0 for _ in self.global_shape]
        loaded_ends = list(self.unpadded_global_shape)
        key_start, key_end = self._local_hf_key_indices(effective_intervals)
        hf_keys = self.global_hf_keys[key_start:key_end]

        if self.is_fused:
            key_size = self._fused_key_size()
            assert self.fused_dim is not None
            loaded_starts[self.fused_dim] = key_start * key_size
            loaded_ends[self.fused_dim] = key_end * key_size

        slices: list[LoadSlice] = []
        for dim, (effective_start, effective_end) in enumerate(effective_intervals):
            loaded_start = loaded_starts[dim]
            loaded_end = loaded_ends[dim]
            if effective_start == loaded_start and effective_end == loaded_end:
                continue
            slices.append(
                LoadSlice(
                    dim=dim,
                    start=effective_start - loaded_start,
                    end=effective_end - loaded_start,
                )
            )

        return HFLoadPlan(name=self.name, hf_keys=hf_keys, fused_dim=self.fused_dim, slices=slices)

    def plan_hf_save(
        self,
        *,
        distributed_save: bool = False,
        preserve_process_group: dist.ProcessGroup | None = None,
        gather_process_group: dist.ProcessGroup | None = None,
    ) -> HFSavePlan:
        """Build a safetensors save plan from this layout spec.

        Args:
            distributed_save (bool): Whether non-fused tensors are written only on rank0 and fused HF keys are split
                across save ranks.
            preserve_process_group (dist.ProcessGroup | None): Fused-dim shard group that should remain sharded,
                used by RL weight sync to stream EP-local expert slices.
            gather_process_group (dist.ProcessGroup | None): If set, only shards from this group are gathered and
                all other shards are preserved. This is used by callers that need an FSDP-only all-gather.

        Returns:
            HFSavePlan: Save-time unshard and HF key planning information.
        """
        assert not (preserve_process_group is not None and gather_process_group is not None), (
            "preserve_process_group and gather_process_group describe different save policies and cannot be combined"
        )
        preserved_shard_indices = self._preserved_shard_indices(
            preserve_process_group=preserve_process_group,
            gather_process_group=gather_process_group,
        )
        unshard_steps = self._save_shard_steps(preserved_shard_indices)
        preserved_shards = [step.shard for step in unshard_steps if step.preserved]
        hf_keys = (
            self._local_hf_keys_for_shards(preserved_shards, require_fused_key_aligned=True)
            if preserved_shards
            else list(self.global_hf_keys)
        )

        return HFSavePlan(
            name=self.name,
            hf_keys=hf_keys,
            global_shape=self.global_shape,
            unpadded_global_shape=self.unpadded_global_shape,
            fused_dim=self.fused_dim,
            distributed_save=distributed_save,
            preserves_shards=bool(preserved_shards),
            unshard_steps=unshard_steps,
        )

    def model_post_init(self, _) -> None:
        if self.is_fused:
            assert self.fused_dim is not None, "fused_dim must be set when global_hf_keys has multiple entries"
        else:
            assert self.fused_dim is None, "fused_dim must be None when global_hf_keys has one entry"
        self._validate_origin_shape()
        self._validate_shards()

    def _effective_intervals_for_shards(
        self,
        shards: list[ShardDescriptor],
    ) -> list[tuple[int, int]] | None:
        effective_shape = self.unpadded_global_shape
        assert len(effective_shape) == len(self.global_shape), (
            f"origin_shape={effective_shape} must have the same rank as global_shape={self.global_shape}"
        )
        assert all(effective <= global_ for effective, global_ in zip(effective_shape, self.global_shape)), (
            f"origin_shape={effective_shape} must not exceed global_shape={self.global_shape}"
        )

        final_intervals = _final_intervals(self.global_shape, shards)
        effective_intervals: list[tuple[int, int]] = []
        for dim, (start, end) in enumerate(final_intervals):
            effective_start = min(start, effective_shape[dim])
            effective_end = min(end, effective_shape[dim])
            if effective_start >= effective_end:
                return None
            effective_intervals.append((effective_start, effective_end))
        return effective_intervals

    def _fused_key_size(self) -> int:
        assert self.fused_dim is not None, "fused_dim must be set when global_hf_keys has multiple entries"
        key_size = self.unpadded_global_shape[self.fused_dim] / len(self.global_hf_keys)
        assert key_size.is_integer(), (
            f"Fused dim size {self.unpadded_global_shape[self.fused_dim]} is not divisible by "
            f"{len(self.global_hf_keys)} HF keys for {self.name}"
        )
        return int(key_size)

    def _local_hf_key_indices(
        self,
        effective_intervals: list[tuple[int, int]],
        *,
        require_fused_key_aligned: bool = False,
    ) -> tuple[int, int]:
        if not self.is_fused:
            return 0, len(self.global_hf_keys)

        assert self.fused_dim is not None
        key_size = self._fused_key_size()
        fused_start, fused_end = effective_intervals[self.fused_dim]
        if require_fused_key_aligned:
            assert fused_start % key_size == 0 and fused_end % key_size == 0, (
                f"Preserved fused shard range [{fused_start}, {fused_end}) for {self.name} must align with "
                f"HF key size {key_size}"
            )

        # Shards may start or end inside a fused HF key, e.g. FSDP slicing an EP-local expert tensor.
        # floor/ceil keeps every overlapping key; LoadSlice later trims load tensors to the exact local range.
        key_start = fused_start // key_size
        key_end = math.ceil(fused_end / key_size)
        assert 0 <= key_start < key_end <= len(self.global_hf_keys), (
            f"Invalid fused key range [{key_start}, {key_end}) for {self.name}"
        )
        return key_start, key_end

    def _local_hf_keys_for_shards(
        self,
        shards: list[ShardDescriptor],
        *,
        require_fused_key_aligned: bool = False,
    ) -> list[str]:
        effective_intervals = self._effective_intervals_for_shards(shards)
        if effective_intervals is None:
            return []
        key_start, key_end = self._local_hf_key_indices(
            effective_intervals,
            require_fused_key_aligned=require_fused_key_aligned,
        )
        return self.global_hf_keys[key_start:key_end]

    def _validate_origin_shape(self) -> None:
        if self.origin_shape is None:
            return

        assert len(self.origin_shape) == len(self.global_shape), (
            f"origin_shape={self.origin_shape} must have the same rank as global_shape={self.global_shape}"
        )
        assert all(origin <= global_ for origin, global_ in zip(self.origin_shape, self.global_shape)), (
            f"origin_shape={self.origin_shape} must not exceed global_shape={self.global_shape}"
        )

    def _validate_shards(self) -> None:
        current_shape = list(self.global_shape)
        for shard in self.shards:
            assert 0 <= shard.dim < len(current_shape), (
                f"Invalid shard dim {shard.dim} for global_shape={self.global_shape}"
            )
            current_size = current_shape[shard.dim]
            assert 0 <= shard.start <= shard.end <= current_size, (
                f"Invalid shard descriptor {shard} against current_shape={tuple(current_shape)}"
            )
            current_shape[shard.dim] = shard.end - shard.start

    def _preserved_shard_indices(
        self,
        *,
        preserve_process_group: dist.ProcessGroup | None,
        gather_process_group: dist.ProcessGroup | None,
    ) -> set[int]:
        """Return ``self.shards`` indices that should remain sharded in this
        save plan.

        ``preserve_process_group`` is only used when a fused HF tensor has an additional runtime partition on
        ``fused_dim``. For example, MoE expert parallel may shard the concatenated expert keys on the same dim that
        HF uses for fused keys, and FSDP may further shard that EP-local tensor on the same dim. RL weight sync wants
        to preserve the EP shard so it can derive the local HF key range from that shard, while all remaining shards
        such as FSDP must still be all-gathered to recover a complete weight for that preserved EP slice.

        ``gather_process_group`` is the inverse policy used by FSDP-only all-gather callers: gather shards from this
        group and preserve every other shard.

        Example:
            Suppose ``global_hf_keys`` represents experts ``[0..7]`` concatenated on dim 0, and the runtime layout is
            ``shards=[ep_shard(dim=0, group=ep_group), fsdp_shard(dim=0, group=fsdp_group)]``. Passing ``ep_group`` as
            ``preserve_process_group`` returns ``{0}``: the EP shard is preserved for local HF key planning, while the
            FSDP shard at index 1 is still all-gathered so the local EP expert slice becomes complete. Passing
            ``fsdp_group`` as ``gather_process_group`` produces the same preserved index set for an FSDP-only gather.

        Returns:
            set[int]: Indices into ``self.shards``, not tensor dimensions.
        """
        if gather_process_group is not None:
            return {
                shard_index
                for shard_index, shard in enumerate(self.shards)
                if not _is_same_process_group(shard.group, gather_process_group)
            }

        if preserve_process_group is None or not self.is_fused:
            return set()

        assert self.fused_dim is not None, (
            f"Internal error: fused LoadSpec {self.name} has no fused_dim. "
            "LoadSpec.model_post_init should reject this layout before save planning."
        )
        return {
            shard_index
            for shard_index, shard in enumerate(self.shards)
            if shard.dim == self.fused_dim and _is_same_process_group(shard.group, preserve_process_group)
        }

    def _save_shard_steps(self, preserved_shard_indices: set[int]) -> list[SaveShardStep]:
        """Convert ``LoadSpec.shards`` into save-time reverse-unshard work
        items.

        ``LoadSpec.shards`` is ordered in the forward partitioning direction: start from the full runtime tensor,
        apply one shard after another, and end at this rank's local tensor. The returned steps keep that same
        largest-to-smallest order. Each step snapshots the runtime shape and the unpadded checkpoint-visible shape
        that existed immediately before its shard was applied.

        Save executes these steps in reverse. Starting from the smallest local tensor, each reverse step all-gathers
        one shard and narrows the gathered tensor back to ``unpadded_shape_before_shard``. This is how the save path
        reconstructs the original shape information one partition layer at a time, while still avoiding fp8 runtime
        padding in the checkpoint-visible tensor.

        Example:
            Suppose ``global_shape=(16, 8)``, ``unpadded_global_shape=(14, 8)``, and
            ``LoadSpec.shards == [ep(dim=0, start=8, end=16), fsdp(dim=0, start=3, end=5)]``. The returned steps are
            in forward order:

            * ``ep_step`` records ``shape_before_shard=(16, 8)`` and
              ``unpadded_shape_before_shard=(14, 8)``.
            * ``fsdp_step`` records ``shape_before_shard=(8, 8)`` and
              ``unpadded_shape_before_shard=(6, 8)``.

            A local save tensor has shape ``(2, 8)``. Save runs ``[fsdp_step, ep_step]``: gather FSDP back toward
            ``(6, 8)``, then gather EP back toward ``(14, 8)``. If EP is preserved, only ``fsdp_step`` remains
            pending and the result stays EP-local.

        Args:
            preserved_shard_indices (set[int]): Original ``LoadSpec.shards`` indices that should remain sharded.

        Returns:
            list[SaveShardStep]: Work items in the same largest-to-smallest order as ``LoadSpec.shards``.
        """
        current_shape = list(self.global_shape)
        effective_shape = list(self.unpadded_global_shape)
        steps: list[SaveShardStep] = []

        for shard_index, shard in enumerate(self.shards):
            steps.append(
                SaveShardStep(
                    load_spec_shard_index=shard_index,
                    shard=shard,
                    shape_before_shard=tuple(current_shape),
                    unpadded_shape_before_shard=tuple(effective_shape),
                    preserved=shard_index in preserved_shard_indices,
                )
            )
            effective_start = min(shard.start, effective_shape[shard.dim])
            effective_end = min(shard.end, effective_shape[shard.dim])
            effective_shape[shard.dim] = max(0, effective_end - effective_start)
            current_shape[shard.dim] = shard.end - shard.start
        return steps
