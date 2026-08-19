import math
from itertools import product
from typing import Callable, NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch.distributed.tensor import DTensor, Shard

from xtuner.v1.ops.comm.foreach_allgather import foreach_all_gather
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.interleaved_shard import RuntimeLayout


def _is_same_process_group(left: dist.ProcessGroup, right: dist.ProcessGroup) -> bool:
    if left is right:
        return True
    return dist.get_process_group_ranks(left) == dist.get_process_group_ranks(right)


class ShardDescriptor(BaseModel):
    """One runtime partition applied to the canonical full tensor.

    Descriptors are applied in forward layout order. ``interleave_factor == 1``
    follows normal ``Shard`` semantics, including uneven continuous shards.
    Larger factors describe even interleave: every rank owns that many equal
    runs from the current tensor dimension.

    Args:
        dim (int): Tensor dim on which this partition cuts.
        group (dist.ProcessGroup): Communication group that produced this partition.
        interleave_factor (int): Number of ordered runs owned by each rank. One means a continuous shard.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    dim: int
    group: dist.ProcessGroup
    interleave_factor: int = Field(default=1, ge=1)

    def local_intervals(self, dim_size: int) -> list[tuple[int, int]]:
        """Return this rank's intervals in the current tensor coordinate."""
        world_size = dist.get_world_size(group=self.group)
        rank = dist.get_rank(group=self.group)
        assert rank >= 0, "ShardDescriptor process group must contain the current rank"

        if self.interleave_factor == 1:
            # XTuner may initialize modules while the default device is meta. PyTorch's
            # placement helper inherits that default for temporary shape arithmetic.
            with torch.device(get_device()):
                local_size, offset = Shard(self.dim)._local_shard_size_and_offset(  # type: ignore[attr-defined]
                    dim_size,
                    world_size,
                    rank,
                )
            return [(offset, offset + local_size)] if local_size else []

        split_count = world_size * self.interleave_factor
        if dim_size % split_count != 0:
            raise NotImplementedError(
                "Even interleave requires size_before_shard to be divisible by "
                f"group_size * interleave_factor, got {dim_size} % "
                f"({world_size} * {self.interleave_factor}) != 0"
            )
        run_size = dim_size // split_count
        return [
            ((run_index * world_size + rank) * run_size, (run_index * world_size + rank + 1) * run_size)
            for run_index in range(self.interleave_factor)
        ]

    def local_size(self, dim_size: int) -> int:
        return sum(end - start for start, end in self.local_intervals(dim_size))


class _OwnedRegion(BaseModel):
    """One contiguous region of the global tensor owned by this rank.

    ``global_offsets`` locate the region in XTuner's canonical global tensor,
    while ``local_offsets`` locate the same data in the runtime local tensor.
    A regular FSDP/EP shard has one region; an interleaved Expert TP layout has
    one region per contiguous run.
    """

    model_config = ConfigDict(extra="forbid")
    global_offsets: tuple[int, ...]
    local_offsets: tuple[int, ...]
    sizes: tuple[int, ...]


class LoadCopyRegion(BaseModel):
    """One source-to-target copy executed by :class:`HFLoadPlan`."""

    model_config = ConfigDict(extra="forbid")
    source_offsets: tuple[int, ...]
    target_offsets: tuple[int, ...]
    sizes: tuple[int, ...]


class HFLoadPlan(BaseModel):
    """Rank-local program for loading HF tensors into one runtime tensor.

    Args:
        name (str): Fully-qualified parameter or buffer name on the xtuner side.
        hf_keys (list[str]): HF keys that must be read for this rank.
        fused_dim (int | None): Concatenation dimension when multiple HF keys are loaded.
        canonical_source_shape (tuple[int, ...] | None): Expected shape after the model adapter converts the loaded
            HF tensor to XTuner's canonical layout. ``None`` means the plan has no checkpoint-backed copy work.
        target_shape (tuple[int, ...]): Expected runtime local-tensor shape.
        copy_regions (list[LoadCopyRegion]): Source-to-target copies in canonical coordinates.
        zero_unwritten_target (bool): Whether to zero the target before executing copies, used for runtime padding.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    hf_keys: list[str]
    fused_dim: int | None = None
    canonical_source_shape: tuple[int, ...] | None = None
    target_shape: tuple[int, ...]
    copy_regions: list[LoadCopyRegion] = Field(default_factory=list)
    zero_unwritten_target: bool = False

    @torch.no_grad()
    def load_into(
        self,
        checkpoint_tensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        canonicalize: Callable[[str, torch.Tensor], torch.Tensor],
    ) -> None:
        """Convert loaded HF tensors and execute this rank's copy program.

        Model adapters own only the HF-layout-to-canonical transformation. This method owns the ordering around that
        adapter and all runtime-layout details, including regular slices, interleaved runs, and padding.
        """
        assert tuple(local_tensor.shape) == self.target_shape, (
            f"Load target shape {tuple(local_tensor.shape)} does not match planned shape "
            f"{self.target_shape} for {self.name}"
        )
        assert len(checkpoint_tensors) == len(self.hf_keys), (
            f"Loaded {len(checkpoint_tensors)} tensors for {len(self.hf_keys)} planned HF keys of {self.name}"
        )

        if self.zero_unwritten_target:
            local_tensor.zero_()
        if not self.copy_regions:
            return

        canonical_tensor = self._canonicalize_source(checkpoint_tensors, canonicalize)
        for region in self.copy_regions:
            source = self._narrow_region(canonical_tensor, region.source_offsets, region.sizes)
            target = self._narrow_region(local_tensor, region.target_offsets, region.sizes)
            target.copy_(source)

    def _canonicalize_source(
        self,
        checkpoint_tensors: list[torch.Tensor],
        canonicalize: Callable[[str, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Build and validate the canonical source before any rank-local
        copy."""
        assert checkpoint_tensors, f"Internal Error. No safetensors were loaded for {self.name}"
        if len(checkpoint_tensors) == 1:
            loaded_tensor = checkpoint_tensors[0]
        else:
            assert self.fused_dim is not None, (
                f"Internal Error. fused_dim must be set when loading multiple HF keys for {self.name}"
            )
            loaded_tensor = torch.cat(checkpoint_tensors, dim=self.fused_dim)

        canonical_tensor = canonicalize(self.name, loaded_tensor)
        assert self.canonical_source_shape is not None
        assert tuple(canonical_tensor.shape) == self.canonical_source_shape, (
            f"Canonical HF tensor shape {tuple(canonical_tensor.shape)} does not match planned shape "
            f"{self.canonical_source_shape} for {self.name}"
        )
        return canonical_tensor

    @staticmethod
    def _narrow_region(
        tensor: torch.Tensor,
        offsets: tuple[int, ...],
        sizes: tuple[int, ...],
    ) -> torch.Tensor:
        assert tensor.dim() == len(offsets) == len(sizes)
        for dim, (offset, size) in enumerate(zip(offsets, sizes)):
            tensor = tensor.narrow(dim, offset, size)
        return tensor


def _layout_segments(
    global_shape: tuple[int, ...],
    shards: list[ShardDescriptor],
) -> list[list[tuple[int, int]]]:
    """Map each local tensor dimension to ordered global segments.

    A dimension starts as one full global segment. Each descriptor slices the *current local order*, so a later FSDP
    shard can cut across multiple ETP runs without materializing an index for every tensor row.
    """
    segments_by_dim = [[(0, dim_size)] if dim_size else [] for dim_size in global_shape]
    for shard in shards:
        assert 0 <= shard.dim < len(global_shape), f"Invalid shard dim {shard.dim} for shape {global_shape}"
        source_segments = segments_by_dim[shard.dim]
        current_size = sum(size for _, size in source_segments)
        selected_intervals = shard.local_intervals(current_size)
        selected_segments: list[tuple[int, int]] = []

        for selected_start, selected_end in selected_intervals:
            local_start = 0
            for global_start, segment_size in source_segments:
                local_end = local_start + segment_size
                overlap_start = max(selected_start, local_start)
                overlap_end = min(selected_end, local_end)
                if overlap_start < overlap_end:
                    mapped_start = global_start + overlap_start - local_start
                    mapped_size = overlap_end - overlap_start
                    previous_start, previous_size = selected_segments[-1] if selected_segments else (0, 0)
                    if selected_segments and previous_start + previous_size == mapped_start:
                        selected_segments[-1] = (previous_start, previous_size + mapped_size)
                    else:
                        selected_segments.append((mapped_start, mapped_size))
                local_start = local_end

        segments_by_dim[shard.dim] = selected_segments
    return segments_by_dim


def _shape_from_segments(
    segments_by_dim: list[list[tuple[int, int]]],
    *,
    visible_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if visible_shape is None:
        return tuple(sum(size for _, size in segments) for segments in segments_by_dim)

    assert len(visible_shape) == len(segments_by_dim)
    return tuple(
        sum(
            max(0, min(global_start + size, visible_size) - min(global_start, visible_size))
            for global_start, size in segments
        )
        for segments, visible_size in zip(segments_by_dim, visible_shape, strict=True)
    )


def _visible_regions_from_segments(
    segments_by_dim: list[list[tuple[int, int]]],
    *,
    visible_shape: tuple[int, ...],
) -> list[_OwnedRegion]:
    """Compile rank ownership into rectangular global-to-local copies."""
    if any(not segments for segments in segments_by_dim):
        return []

    located_segments: list[list[tuple[int, int, int]]] = []
    for segments in segments_by_dim:
        local_offset = 0
        current: list[tuple[int, int, int]] = []
        for global_offset, size in segments:
            current.append((global_offset, local_offset, size))
            local_offset += size
        located_segments.append(current)

    regions: list[_OwnedRegion] = []
    for segment_tuple in product(*located_segments):
        global_offsets: list[int] = []
        local_offsets: list[int] = []
        sizes: list[int] = []
        for dim, (global_offset, local_offset, size) in enumerate(segment_tuple):
            clipped_start = min(global_offset, visible_shape[dim])
            clipped_end = min(global_offset + size, visible_shape[dim])
            clipped_size = max(0, clipped_end - clipped_start)
            if clipped_size == 0:
                break
            global_offsets.append(clipped_start)
            local_offsets.append(local_offset + clipped_start - global_offset)
            sizes.append(clipped_size)
        else:
            regions.append(
                _OwnedRegion(
                    global_offsets=tuple(global_offsets),
                    local_offsets=tuple(local_offsets),
                    sizes=tuple(sizes),
                )
            )
    return regions


class SaveShardStep(BaseModel):
    """Save-time work item derived from one ``LoadSpec.shards`` entry.

    ``LoadSpec.shards`` is a layout description: each descriptor says how the previous tensor was partitioned.
    Saving needs the inverse operation. ``LoadSpec._save_shard_steps`` converts every shard descriptor into a work
    item that contains the shard itself plus the tensor shapes that existed immediately before that shard was applied.
    The save path then executes these work items in reverse order and batches compatible all-gathers by process group.

    Example:
        ``LoadSpec.shards == [ep_shard, fsdp_shard]`` means the full HF tensor was first cut by EP, then the
        EP-local tensor was cut by FSDP. Normal HF save executes ``[fsdp_step, ep_step]`` to rebuild the full tensor.
        RL weight sync can mark ``ep_step`` as preserved, so only the FSDP work item is executed and the result stays
        EP-local.

    Args:
        shard (ShardDescriptor): Shard descriptor this save step reverses.
        shape_before_shard (tuple[int, ...]): Runtime tensor shape immediately before ``shard`` was applied.
        preserved (bool): Whether this shard should remain applied instead of being all-gathered.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    shard: ShardDescriptor
    shape_before_shard: tuple[int, ...]
    preserved: bool = False


class HFSavePlan(BaseModel):
    """Execution plan for preparing one runtime tensor for HF safetensors save.

    Args:
        name (str): Fully-qualified parameter or buffer name on the xtuner side.
        hf_keys (list[str]): HF keys represented by the tensor after this plan's pending unshard steps finish.
        runtime_output_shape (tuple[int, ...]): Shape after pending gathers, before removing FP8 runtime padding.
        output_shape (tuple[int, ...]): Checkpoint-visible shape after pending gathers and final padding trim.
        fused_dim (int | None): HF key concatenation dim when the underlying ``LoadSpec`` is fused; ``None``
            otherwise.
        distributed_save (bool): Whether non-fused tensors are written only on rank0 and fused keys are split across
            save ranks.
        unshard_steps (list[SaveShardStep]): Forward-order shard history with save-time preserved flags.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    hf_keys: list[str]
    runtime_output_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    fused_dim: int | None = None
    distributed_save: bool = False
    unshard_steps: list[SaveShardStep] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def preserves_shards(self) -> bool:
        return any(step.preserved for step in self.unshard_steps)


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
    # sync, are removed from the queue. Their effect is already represented by the plan's output shapes.

    # Example:
    #   tensor A: [ep_a(preserved), fsdp_a] -> pending [fsdp_a]
    #   tensor B: [ep_b, fsdp_b] -> pending [fsdp_b, ep_b]
    #   tensor C: [fsdp_c] -> pending [fsdp_c]
    #   tensor D: [tp_d] -> pending [tp_d]
    #   tensor E: [ep_e(preserved)] -> pending []
    # This produces one pending queue per tensor; the loop below consumes compatible queue heads by group.
    pending_shard_steps_list = [
        [step for step in reversed(save_plan.unshard_steps) if not step.preserved] for save_plan in save_plans
    ]

    while True:
        # Build one all-gather round. For one tensor, reverse-unshard steps must run one by one: if a local
        # tensor needs to undo FSDP and then EP, the EP gather must use the tensor produced by the FSDP gather.
        # `_take_ready_save_unshard_groups` consumes `pending_shard_steps_list` gradually. For example, a queue
        # `[fsdp_step, ep_step]` contributes `fsdp_step` in the first round; after its gathered tensor is written
        # back, the next loop consumes `ep_step`. Independent tensors with compatible group/dtype can still be
        # batched together in each round.
        #
        # With the A-E example above, round 1 consumes fsdp_a/fsdp_b/fsdp_c together if they share group/dtype,
        # and consumes tp_d in another group. tensor E contributes no work. Round 2 can then consume ep_b, because
        # ep_b must use tensor B after fsdp_b has been gathered and written back.
        unshard_groups = _take_ready_save_unshard_groups(tensor_list, pending_shard_steps_list)
        if not unshard_groups:
            break

        for unshard_group in unshard_groups:
            gathered_tensors = _foreach_all_gather_save_shards(
                unshard_group.tensors,
                unshard_group.shard_steps,
            )
            for index, gathered_tensor in zip(unshard_group.tensor_indices, gathered_tensors, strict=True):
                tensor_list[index] = gathered_tensor

    # Collectives reconstruct runtime shapes; checkpoint-invisible FP8 tail
    # padding is removed only after the requested shard history is complete.
    return [
        _finalize_hf_save_tensor(tensor, save_plan) for tensor, save_plan in zip(tensor_list, save_plans, strict=True)
    ]


def _take_ready_save_unshard_groups(
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


def _finalize_hf_save_tensor(tensor: torch.Tensor, save_plan: HFSavePlan) -> torch.Tensor:
    """Validate the reconstructed runtime shape and trim FP8 tail padding."""
    assert tuple(tensor.shape) == save_plan.runtime_output_shape, (
        f"Save reconstruction produced shape {tuple(tensor.shape)}, expected runtime shape "
        f"{save_plan.runtime_output_shape} for {save_plan.name}"
    )
    assert all(output_size <= tensor.shape[dim] for dim, output_size in enumerate(save_plan.output_shape))

    output = tensor[tuple(slice(0, size) for size in save_plan.output_shape)].contiguous()
    assert tuple(output.shape) == save_plan.output_shape, (
        f"Saved tensor shape {tuple(output.shape)} is incompatible with HFSavePlan output_shape="
        f"{save_plan.output_shape} for {save_plan.name}"
    )
    return output


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

    expected_local_size = shard_step.shard.local_size(shard_dim_size)
    assert tensor.shape[dim] == expected_local_size, (
        f"Local tensor shape {tuple(tensor.shape)} does not match descriptor-local size "
        f"{expected_local_size} for {shard_step.shard}"
    )
    if shard_step.shard.interleave_factor > 1:
        # Even interleave guarantees equal local tensors, so collective padding
        # would only hide an invalid layout.
        return tensor

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
    runtime_dim_size = shard_step.shape_before_shard[dim]
    if shard_step.shard.interleave_factor == 1:
        gathered_tensor = torch.cat(gathered_chunks, dim=dim)
        return gathered_tensor.narrow(dim, 0, runtime_dim_size).contiguous()

    world_size = dist.get_world_size(group=shard_step.shard.group)
    interleave_factor = shard_step.shard.interleave_factor
    assert len(gathered_chunks) == world_size
    assert runtime_dim_size % (world_size * interleave_factor) == 0
    run_size = runtime_dim_size // (world_size * interleave_factor)
    ordered_runs = [
        gathered_chunks[rank].narrow(dim, run_index * run_size, run_size)
        for run_index in range(interleave_factor)
        for rank in range(world_size)
    ]
    return torch.cat(ordered_runs, dim=dim).contiguous()


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
        local_shape (tuple[int, ...] | None): Runtime local-tensor shape. It is recorded explicitly for layouts such
            as InterleavedShard and checked against the shape derived from ``global_shape`` and ``shards``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    global_hf_keys: list[str]
    global_shape: tuple[int, ...]
    fused_dim: int | None = None
    shards: list[ShardDescriptor] = Field(default_factory=list)
    origin_shape: tuple[int, ...] | None = None
    local_shape: tuple[int, ...] | None = None

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

        It derives the fused HF dimension from ``hf_keys`` and converts the stable
        ``RuntimeLayout`` records into ``ShardDescriptor`` entries. Callers should
        pass ``origin_shape`` when runtime-only padding makes the checkpoint-visible
        shape smaller than the runtime shape.

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
        if isinstance(tensor, DTensor):
            runtime_layout = RuntimeLayout.from_dtensor(tensor)
            shards = [
                ShardDescriptor(
                    dim=shard.dim,
                    group=shard.group,
                    interleave_factor=shard.interleave_factor,
                )
                for shard in runtime_layout.ordered_shards
            ]
        else:
            shards = []
        local_tensor = tensor._local_tensor if isinstance(tensor, DTensor) else tensor
        return cls(
            name=name,
            global_hf_keys=global_hf_keys,
            global_shape=tuple(tensor.shape),
            fused_dim=0 if len(global_hf_keys) > 1 else None,
            shards=shards,
            origin_shape=origin_shape,
            local_shape=tuple(local_tensor.shape),
        )

    def plan_hf_load(self) -> HFLoadPlan:
        """Build a safetensors read plan from this layout spec.

        Runtime-only padding currently comes from XTuner float8 weights. In that case, ``origin_shape`` is used as
        the checkpoint-visible full tensor shape, while ``global_shape`` and ``shards`` still describe the padded
        runtime layout that this rank owns.

        Returns:
            HFLoadPlan: The selected HF keys and canonical source-to-local copy program for this rank.
        """
        segments_by_dim = _layout_segments(self.global_shape, self.shards)
        target_shape = _shape_from_segments(segments_by_dim)
        owned_regions = _visible_regions_from_segments(
            segments_by_dim,
            visible_shape=self.unpadded_global_shape,
        )
        if not owned_regions:
            return HFLoadPlan(
                name=self.name,
                hf_keys=[],
                fused_dim=self.fused_dim,
                target_shape=target_shape,
                zero_unwritten_target=math.prod(target_shape) > 0,
            )

        hf_keys, source_offsets, source_shape = self._hf_load_source(owned_regions)
        copy_regions = [
            LoadCopyRegion(
                source_offsets=tuple(
                    region.global_offsets[dim] - source_offsets[dim] for dim in range(len(self.global_shape))
                ),
                target_offsets=region.local_offsets,
                sizes=region.sizes,
            )
            for region in owned_regions
        ]
        copied_numel = sum(math.prod(region.sizes) for region in copy_regions)
        target_numel = math.prod(target_shape)
        assert copied_numel <= target_numel, (
            f"Owned regions for {self.name} copy {copied_numel} values into a target with {target_numel} values"
        )

        return HFLoadPlan(
            name=self.name,
            hf_keys=hf_keys,
            fused_dim=self.fused_dim,
            canonical_source_shape=source_shape,
            target_shape=target_shape,
            copy_regions=copy_regions,
            zero_unwritten_target=copied_numel < target_numel,
        )

    def _hf_load_source(
        self,
        owned_regions: list[_OwnedRegion],
    ) -> tuple[list[str], tuple[int, ...], tuple[int, ...]]:
        """Select HF keys and express their canonical tensor in global
        coordinates."""
        key_start, key_end = self._hf_key_range_for_regions(owned_regions)
        source_offsets = [0 for _ in self.global_shape]
        source_shape = list(self.unpadded_global_shape)
        if self.is_fused:
            assert self.fused_dim is not None
            key_size = self._fused_key_size()
            source_offsets[self.fused_dim] = key_start * key_size
            source_shape[self.fused_dim] = (key_end - key_start) * key_size

        return (
            self.global_hf_keys[key_start:key_end],
            tuple(source_offsets),
            tuple(source_shape),
        )

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
        if preserve_process_group is not None and preserved_shards:
            hf_keys = self._local_hf_keys_for_shards(preserved_shards, require_fused_key_aligned=True)
        else:
            # FSDP-only gather keeps ETP's runtime layout and does not produce HF
            # tensors, so its key list is informational and needs no key alignment.
            hf_keys = list(self.global_hf_keys)

        output_segments = _layout_segments(self.global_shape, preserved_shards)
        runtime_output_shape = _shape_from_segments(output_segments)
        output_shape = _shape_from_segments(
            output_segments,
            visible_shape=self.unpadded_global_shape,
        )

        return HFSavePlan(
            name=self.name,
            hf_keys=hf_keys,
            runtime_output_shape=runtime_output_shape,
            output_shape=output_shape,
            fused_dim=self.fused_dim,
            distributed_save=distributed_save,
            unshard_steps=unshard_steps,
        )

    def model_post_init(self, _) -> None:
        if self.is_fused:
            assert self.fused_dim is not None, "fused_dim must be set when global_hf_keys has multiple entries"
        else:
            assert self.fused_dim is None, "fused_dim must be None when global_hf_keys has one entry"
        self._validate_origin_shape()
        self._validate_shards()

    def _fused_key_size(self) -> int:
        assert self.fused_dim is not None, "fused_dim must be set when global_hf_keys has multiple entries"
        key_size = self.unpadded_global_shape[self.fused_dim] / len(self.global_hf_keys)
        assert key_size.is_integer(), (
            f"Fused dim size {self.unpadded_global_shape[self.fused_dim]} is not divisible by "
            f"{len(self.global_hf_keys)} HF keys for {self.name}"
        )
        return int(key_size)

    def _hf_key_range_for_regions(
        self,
        regions: list[_OwnedRegion],
        *,
        require_fused_key_aligned: bool = False,
    ) -> tuple[int, int]:
        if not self.is_fused:
            return 0, len(self.global_hf_keys)

        assert self.fused_dim is not None
        key_size = self._fused_key_size()
        fused_start = min(region.global_offsets[self.fused_dim] for region in regions)
        fused_end = max(region.global_offsets[self.fused_dim] + region.sizes[self.fused_dim] for region in regions)
        if require_fused_key_aligned:
            assert fused_start % key_size == 0 and fused_end % key_size == 0, (
                f"Preserved fused shard range [{fused_start}, {fused_end}) for {self.name} must align with "
                f"HF key size {key_size}"
            )

        # Shards may start or end inside a fused HF key, e.g. FSDP slicing an EP-local expert tensor.
        # floor/ceil keeps every overlapping key; the load plan's copy regions trim to the exact local range.
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
        segments_by_dim = _layout_segments(self.global_shape, shards)
        regions = _visible_regions_from_segments(
            segments_by_dim,
            visible_shape=self.unpadded_global_shape,
        )
        if not regions:
            return []
        key_start, key_end = self._hf_key_range_for_regions(
            regions,
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
        segments_by_dim = _layout_segments(self.global_shape, self.shards)
        derived_shape = _shape_from_segments(segments_by_dim)

        assert self.local_shape is None or derived_shape == self.local_shape, (
            f"Recorded local_shape={self.local_shape} does not match descriptor-derived shape "
            f"{derived_shape} for {self.name}"
        )

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

        ``LoadSpec.shards`` is ordered in the forward partitioning direction. Each step snapshots only the runtime
        shape before its shard. Save executes the steps in reverse and restores those runtime shapes; checkpoint-only
        FP8 trimming happens once at the final ``HFSavePlan`` output boundary.

        Args:
            preserved_shard_indices (set[int]): Original ``LoadSpec.shards`` indices that should remain sharded.

        Returns:
            list[SaveShardStep]: Work items in the same largest-to-smallest order as ``LoadSpec.shards``.
        """
        current_shape = list(self.global_shape)
        steps: list[SaveShardStep] = []

        for shard_index, shard in enumerate(self.shards):
            steps.append(
                SaveShardStep(
                    shard=shard,
                    shape_before_shard=tuple(current_shape),
                    preserved=shard_index in preserved_shard_indices,
                )
            )
            current_shape[shard.dim] = sum(
                end - start for start, end in shard.local_intervals(current_shape[shard.dim])
            )
        return steps
