"""DCP planners that make ``InterleavedShard`` (tpep) DTensors participate in
DCP.

An ``InterleavedShard`` DTensor (per-expert column-parallel fused MoE weights, produced when
``tp_size > 1``) stores a local tensor that is **several interleaved runs** of the global
tensor rather than one contiguous slice. DCP's default planners model each DTensor as a single
contiguous chunk via ``compute_local_shape_and_global_offset`` — which cannot describe this
placement: on torch >= 2.9 it raises (``_StridedShard`` split_factor != aggregate mesh size),
and on 2.8 it silently returns a wrong ``(size, offset)``. Either way the default path is
unusable for these params, which is why they used to be dropped from DCP.

The fix routes those DTensors through ``RuntimeLayout.owned_runs()``, which
decomposes the local tensor into contiguous ``Run`` records mapped to their true global
offsets. Each run becomes one ``WriteItem`` (save) / ``ReadItem`` (load), so the checkpoint is
stored in global coordinates and reshards correctly across different tp/ep topologies.

Every other object (plain tensors, non-interleaved DTensors, optimizer state) falls through to
the default planner unchanged.
"""

from __future__ import annotations

import dataclasses

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    ReadItem,
    SavePlan,
    SavePlanner,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (  # type: ignore[attr-defined]
    _compare_save_plans,
    _create_read_items,
    _create_write_items,
    create_read_items_for_chunk_list,
)
from torch.distributed.tensor import DTensor

from xtuner.v1.utils.interleaved_shard import Run, RuntimeLayout

from .xtuner_cache_planner import XtunerCacheSavePlanner


__all__ = ["InterleavedShardSavePlanner", "InterleavedShardLoadPlanner"]


class InterleavedShardSavePlanner(XtunerCacheSavePlanner):
    """DCP ``SavePlanner`` that emits one ``WriteItem`` per contiguous run for
    InterleavedShard DTensors, while preserving
    :class:`XtunerCacheSavePlanner`'s incremental-save plan caching for every
    other object."""

    _interleaved_runs: dict[str, dict[tuple, Run]]

    def create_local_plan(self) -> SavePlan:
        # Build write items directly. Interleaved DTensors must NOT reach torch's default write-item
        # builder: ``compute_local_shape_and_global_offset`` raises for their ``_StridedShard``
        # placement (split_factor != aggregate mesh size) on torch >= 2.9, and silently returns a
        # wrong single chunk on 2.8. Route them through per-run items; defer every other object to
        # the default builder (matching DefaultSavePlanner, incl. the DTensor submesh-coordinate
        # guard). Done before the plan-caching comparison so the cached plan matches what
        # ``resolve_data`` streams.
        requests: list[WriteItem] = []
        self._interleaved_runs = {}
        for fqn, obj in self.state_dict.items():
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is None:
                    continue
                layout = RuntimeLayout.from_dtensor(obj)
                if layout.is_interleaved:
                    items, run_map = _interleaved_write_items(fqn, obj, layout.owned_runs())
                    requests.extend(items)
                    self._interleaved_runs[fqn] = run_map
                else:
                    requests.extend(_create_write_items(fqn, obj))
            else:
                requests.extend(_create_write_items(fqn, obj))
        plan = SavePlan(requests)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan

        # Mirror DefaultSavePlanner.create_local_plan's caching short-circuit (torch 2.7.x
        # incremental save): skip re-sending an unchanged local plan to the coordinator.
        if self._enable_plan_caching:  # type: ignore[attr-defined]
            cached = SavePlanner._cached_save_plan  # type: ignore[attr-defined]
            if self._cached_plans_key in cached and _compare_save_plans(plan, cached[self._cached_plans_key]):
                return SavePlan([], usable=False)  # type: ignore[call-arg]
            cached[self._cached_plans_key] = plan
        return self.plan

    def resolve_data(self, write_item: WriteItem):
        offset = write_item.index.offset
        if offset is not None:
            run = self._interleaved_runs.get(write_item.index.fqn, {}).get(tuple(offset))
            if run is not None:
                local = self.state_dict[write_item.index.fqn]._local_tensor
                return local.narrow(0, run.local_start, run.local_size).contiguous()
        return super().resolve_data(write_item)


class InterleavedShardLoadPlanner(DefaultLoadPlanner):
    """DCP ``LoadPlanner`` that reads InterleavedShard DTensors as one
    ``ReadItem`` per contiguous run, resharding from the global-coordinate
    checkpoint into this rank's interleaved runs."""

    _interleaved_runs: dict[str, dict[tuple, Run]]

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None
        # Reimplement the default per-fqn read-item loop so interleaved DTensors never reach torch's
        # default chunk builder (``compute_local_shape_and_global_offset`` raises for their
        # ``_StridedShard`` placement on torch >= 2.9). Interleaved DTensors get per-run read items
        # against the checkpoint metadata (so DCP reshards into this rank's runs); every other object
        # uses the default builder unchanged. The pre-2.4 checkpoint version fallback in
        # ``DefaultLoadPlanner.create_local_plan`` is dropped — xtuner never writes those.
        self._interleaved_runs = {}
        requests: list[ReadItem] = []
        strict = not self.allow_partial_load
        for fqn, obj in self.state_dict.items():
            if fqn not in self.metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                continue
            md = self.metadata.state_dict_metadata[fqn]
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is None:
                    continue
                layout = RuntimeLayout.from_dtensor(obj)
                if layout.is_interleaved:
                    runs = layout.owned_runs()
                    local_chunks = [
                        ChunkStorageMetadata(offsets=torch.Size(run.global_offset), sizes=torch.Size(run.sizes))
                        for run in runs
                    ]
                    self._interleaved_runs[fqn] = {tuple(run.global_offset): run for run in runs}
                    requests.extend(create_read_items_for_chunk_list(fqn, md, local_chunks))  # type: ignore[arg-type]
                else:
                    requests.extend(_create_read_items(fqn, md, obj))
            else:
                requests.extend(_create_read_items(fqn, md, obj))
        return LoadPlan(requests)

    def resolve_tensor(self, read_item: ReadItem):
        offset = read_item.dest_index.offset
        run = (
            self._interleaved_runs.get(read_item.dest_index.fqn, {}).get(tuple(offset)) if offset is not None else None
        )
        if run is not None:
            local = self.state_dict[read_item.dest_index.fqn]._local_tensor
            run_slice = local.narrow(0, run.local_start, run.local_size)
            # ``dest_offsets`` / ``lengths`` are relative to the run (the "current shard"), so narrow
            # the run slice — not the whole local tensor — to land the checkpoint bytes correctly.
            return narrow_tensor_by_index(run_slice, read_item.dest_offsets, read_item.lengths)
        return super().resolve_tensor(read_item)


def _interleaved_write_items(
    fqn: str,
    dt: DTensor,
    runs: list[Run],
) -> tuple[list[WriteItem], dict[tuple, Run]]:
    properties = TensorProperties.create_from_tensor(dt._local_tensor)
    global_size = torch.Size(dt.shape)
    items: list[WriteItem] = []
    run_map: dict[tuple, Run] = {}
    for run in runs:
        offsets = torch.Size(run.global_offset)
        sizes = torch.Size(run.sizes)
        items.append(
            WriteItem(
                index=MetadataIndex(fqn, offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
                    properties=properties,
                    size=global_size,
                ),
            )
        )
        run_map[tuple(run.global_offset)] = run
    return items, run_map
