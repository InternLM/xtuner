"""XTuner-owned VMM layout for MoonEP expert weights and gradients.

MoonEP owns the transport and low-level VMM primitives.  XTuner owns this
layout because it is coupled to XTuner's FSDP lifecycle and grouped-GEMM
contract: communication addresses ``E + B`` expert chunks while the single
grouped GEMM consumes one contiguous ``2B`` alias (home followed by duplicate).
"""

from __future__ import annotations

import os
import socket
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from typing import TypeAlias, cast

import torch
import torch.distributed as dist
from typing_extensions import TypedDict


# Shape legend used by every workspace structure below:
#   E: global experts, R: EP ranks, B=E/R: home experts per rank,
#   P=2: fused projections, G=2: FSDP home generations, N: gradient slots.
# Process-lifetime quarantine for an abandoned workspace's complete tensor
# reference graph. Rank-divergent ``__del__`` must not unmap VMM storage that
# a surviving peer may still use; explicit ``destroy()`` never appends here.
_UNDISPOSED_WORKSPACE_TENSORS: list[object] = []

# (physical tensor [B, O_p, I_p], still-open local export FD). The matching
# CUDA allocation handle is deliberately owned separately by ``ExitStack``.
_VMMAllocation: TypeAlias = tuple[torch.Tensor, int]
# Rank-ordered imported FDs: [P][G][R] for home weights or [P][N][R] for
# duplicate gradients. Every descriptor closes after all views are mapped.
_FDGraph: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


class _WorkspaceAllocations(TypedDict):
    """Temporary ownership graph for physical chunks and local export FDs.

    It exists only inside ``allocate()`` while its ``ExitStack`` is open.
    Collections are projection-first: ``P=2`` projections, ``G=2`` home
    generations, ``N`` gradient slots, and every chunk is ``[B, O_p, I_p]``.
    """

    # [P], each (B, O_p, I_p): allocation and mapping granularity.
    chunk_shapes: tuple[tuple[int, ...], ...]
    # [P][G]: local home chunks/FDS that become FSDP AllGather landings.
    home_weights: tuple[tuple[_VMMAllocation, ...], ...]
    # [P]: local duplicate-weight destination/FDS shared by both generations.
    duplicate_weights: tuple[_VMMAllocation, ...]
    # [P][N]: local-home WGrad chunks/FDS, one per invocation slot.
    home_gradients: tuple[tuple[_VMMAllocation, ...], ...]
    # [P][N]: duplicate WGrad chunks/FDS published to every EP owner.
    duplicate_gradients: tuple[tuple[_VMMAllocation, ...], ...]
    # Flat strong references to all physical tensors; excludes mapped views.
    keepalives: tuple[torch.Tensor, ...]


class _WorkspaceLayout(TypedDict):
    """Completed runtime view graph, transposed consumer-first.

    ``P=2`` projections, ``G=2`` home generations, ``N`` gradient slots,
    ``R`` EP ranks, ``B=E/R``, and projection ``p`` has shape ``[O_p, I_p]``.
    This structure crosses the allocation commit point and initializes the
    long-lived ``_ExpertVMMWorkspace``; it contains no open descriptors.
    """

    # [G][P], each [B, O_p, I_p]: FSDP AllGather output targets.
    landings: tuple[tuple[torch.Tensor, ...], ...]
    # [G][P], each [E+B, O_p, I_p]: MoonEP prefetch addresses all home experts plus local duplicates.
    global_weights: tuple[tuple[torch.Tensor, ...], ...]
    # [G][P], each [2B, O_p, I_p]: zero-copy [home, duplicate] weights consumed by grouped GEMM.
    local_weights: tuple[tuple[torch.Tensor, ...], ...]
    # [N][P], each [2B, O_p, I_p]: direct grouped-GEMM WGrad targets for one invocation slot.
    local_grad_outputs: tuple[tuple[torch.Tensor, ...], ...]
    # [N][P], each [R, B, O_p, I_p]: every rank's duplicate WGrad, mapped for return to local home.
    distributed_duplicate_grads: tuple[tuple[torch.Tensor, ...], ...]
    # Physical chunks that own storage backing every non-owning mapped view.
    keepalives: tuple[torch.Tensor, ...]


class _ExpertVMMWorkspace:
    """Own one model/EP group's completed ``_WorkspaceLayout``.

    ``_WorkspaceLayout`` is the single source of truth for every runtime
    view's shape, indexing, and storage ownership. This object adds the EP
    metadata and explicit distributed lifecycle around that layout.
    """

    def __init__(
        self,
        *,
        layout: _WorkspaceLayout,
        ep_group: dist.ProcessGroup,
        ep_rank: int,
        num_experts: int,
        experts_per_rank: int,
    ) -> None:
        # Publish the completed layout under lifecycle-managed private names;
        # their detailed contracts remain centralized on _WorkspaceLayout.
        self._landings = layout["landings"]
        self._global_weights = layout["global_weights"]
        self._local_weights = layout["local_weights"]
        self._local_grad_outputs = layout["local_grad_outputs"]
        self._distributed_duplicate_grads = layout["distributed_duplicate_grads"]
        self._keepalives = layout["keepalives"]
        self._ep_group = ep_group
        self._ep_rank = ep_rank
        self._num_experts = num_experts
        self._experts_per_rank = experts_per_rank
        self._gradient_slots = len(layout["local_grad_outputs"])
        self._destroyed = False

    @classmethod
    def allocate(
        cls,
        *,
        projection_shapes: Sequence[tuple[int, int]],
        num_experts: int,
        ep_group: dist.ProcessGroup,
        gradient_slots: int,
    ) -> _ExpertVMMWorkspace:
        """Validate, allocate, and publish one complete VMM workspace."""
        ep_size, ep_rank = cls._validate_and_resolve_topology(
            projection_shapes=projection_shapes,
            num_experts=num_experts,
            ep_group=ep_group,
        )
        experts_per_rank = num_experts // ep_size

        # Keep descriptors and allocation handles alive across all three
        # setup phases. They are released together after every VMM view has
        # imported them, including when a later phase fails.
        with ExitStack() as resources:
            allocations = cls._allocate_physical_chunks(
                projection_shapes=projection_shapes,
                experts_per_rank=experts_per_rank,
                gradient_slots=gradient_slots,
                resources=resources,
            )
            home_weight_graph, duplicate_gradient_graph = cls._build_ipc_fd_graph(
                allocations=allocations,
                ep_group=ep_group,
                ep_size=ep_size,
                ep_rank=ep_rank,
                resources=resources,
            )
            layout = cls._map_workspace_views(
                allocations=allocations,
                home_weight_graph=home_weight_graph,
                duplicate_gradient_graph=duplicate_gradient_graph,
                ep_size=ep_size,
                ep_rank=ep_rank,
            )

        return cls(
            layout=layout,
            ep_group=ep_group,
            ep_rank=ep_rank,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
        )

    @staticmethod
    def _validate_and_resolve_topology(
        *,
        projection_shapes: Sequence[tuple[int, int]],
        num_experts: int,
        ep_group: dist.ProcessGroup,
    ) -> tuple[int, int]:
        """Resolve the EP coordinates after group-wide topology checks."""
        if not dist.is_initialized():
            raise RuntimeError("MoonEP workspace requires an initialized process group")
        if ep_group is None:
            raise ValueError("ep_group must be provided explicitly")

        # These are implementation preconditions, not a second configuration
        # validation layer.  Dispatcher construction already validates EP size,
        # dtype, top-k, and model metadata before this allocation boundary.
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
        if num_experts % ep_size:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})")
        if len(projection_shapes) != 2:
            raise ValueError("MoonEP requires fused w1/w3 and w2 projections")

        # Host coordination is restricted to this one-time initialization.
        # The forward/backward hot path uses only VMM aliases and CUDA events.
        # ``members[R]`` is ordered by EP rank and stores (hostname, device).
        members: list[tuple[str, int] | None] = [None] * ep_size
        dist.all_gather_object(
            members,
            (socket.gethostname(), torch.cuda.current_device()),
            group=ep_group,
        )
        hosts = {member[0] for member in members if member is not None}
        if len(hosts) != 1:
            raise ValueError("MoonEP requires a node-local ep_group")
        devices = [member[1] for member in members if member is not None]
        if len(set(devices)) != ep_size:
            raise ValueError("each EP rank must use a distinct CUDA device")
        local_device = torch.cuda.current_device()
        if any(peer != local_device and not torch.cuda.can_device_access_peer(local_device, peer) for peer in devices):
            raise ValueError("all EP devices must be CUDA peer-accessible")
        return ep_size, ep_rank

    @staticmethod
    def _allocate_physical_chunks(
        *,
        projection_shapes: Sequence[tuple[int, int]],
        experts_per_rank: int,
        gradient_slots: int,
        resources: ExitStack,
    ) -> _WorkspaceAllocations:
        """Allocate the physical chunks before any cross-rank mapping."""
        # Import the optional backend only at resource installation time. A
        # normal XTuner import or meta model build remains MoonEP-independent.
        from moonep._C import get_vmm_granularity, nvl_dist_alloc, nvl_release_mem_handle

        dtype = torch.bfloat16
        chunk_shapes = tuple((experts_per_rank, *projection_shape) for projection_shape in projection_shapes)
        granularity = get_vmm_granularity()
        element_size = torch.empty((), dtype=dtype).element_size()
        for chunk_shape in chunk_shapes:
            chunk_bytes = element_size
            for dim in chunk_shape:
                chunk_bytes *= dim
            if chunk_bytes % granularity:
                raise ValueError(
                    f"expert home chunk requires {granularity}-byte VMM alignment, "
                    f"got {chunk_bytes} bytes for shape {chunk_shape}"
                )

        # Flat ownership list for every physical tensor. Mapped views do not
        # themselves keep the underlying CUDA physical allocations alive.
        keepalives: list[torch.Tensor] = []

        def allocate_chunk(chunk_shape: tuple[int, ...]) -> _VMMAllocation:
            tensor, fd, handle = nvl_dist_alloc(shape=list(chunk_shape), dtype=dtype)
            # ExitStack runs callbacks in reverse: close the exported FD before
            # releasing its allocation handle, matching MoonEP's lifecycle.
            resources.callback(nvl_release_mem_handle, handle)
            resources.callback(os.close, fd)
            keepalives.append(tensor)
            return tensor, fd

        # Build projection-first ``[P][G/N]`` collections because each
        # projection has its own (O_p, I_p) chunk shape.
        home_weights: list[tuple[_VMMAllocation, ...]] = []
        duplicate_weights: list[_VMMAllocation] = []
        home_gradients: list[tuple[_VMMAllocation, ...]] = []
        duplicate_gradients: list[tuple[_VMMAllocation, ...]] = []
        for chunk_shape in chunk_shapes:
            duplicate_weights.append(allocate_chunk(chunk_shape))
            home_weights.append(tuple(allocate_chunk(chunk_shape) for _ in range(2)))
            home_gradients.append(tuple(allocate_chunk(chunk_shape) for _ in range(gradient_slots)))
            duplicate_gradients.append(tuple(allocate_chunk(chunk_shape) for _ in range(gradient_slots)))

        return _WorkspaceAllocations(
            chunk_shapes=chunk_shapes,
            home_weights=tuple(home_weights),
            duplicate_weights=tuple(duplicate_weights),
            home_gradients=tuple(home_gradients),
            duplicate_gradients=tuple(duplicate_gradients),
            keepalives=tuple(keepalives),
        )

    @staticmethod
    def _build_ipc_fd_graph(
        *,
        allocations: _WorkspaceAllocations,
        ep_group: dist.ProcessGroup,
        ep_size: int,
        ep_rank: int,
        resources: ExitStack,
    ) -> tuple[_FDGraph, _FDGraph]:
        """Exchange the FDs needed by the global weight and gradient views."""
        from moonep.buffer import _exchange_ipc_fds

        sender_ranks = list(range(ep_size))

        def exchange(local_fd: int) -> tuple[int, ...]:
            exchanged = _exchange_ipc_fds(
                local_fd,
                sender_ranks,
                ep_rank,
                ep_size,
                ep_group,
            )
            ordered_fds = tuple(exchanged[rank] for rank in sender_ranks)
            for fd in ordered_fds:
                resources.callback(os.close, fd)
            return ordered_fds

        # Both graphs preserve projection and generation/slot ordering; each
        # innermost tuple is ordered by EP rank for direct ``nvl_dist_map`` use.
        home_weight_graph = tuple(
            tuple(exchange(home_fd) for _, home_fd in projection) for projection in allocations["home_weights"]
        )
        duplicate_gradient_graph = tuple(
            tuple(exchange(duplicate_fd) for _, duplicate_fd in projection)
            for projection in allocations["duplicate_gradients"]
        )
        return home_weight_graph, duplicate_gradient_graph

    @staticmethod
    def _map_workspace_views(
        *,
        allocations: _WorkspaceAllocations,
        home_weight_graph: _FDGraph,
        duplicate_gradient_graph: _FDGraph,
        ep_size: int,
        ep_rank: int,
    ) -> _WorkspaceLayout:
        """Map the physical/IPC graph into the views consumed at runtime."""
        from moonep._C import nvl_dist_map

        dtype = torch.bfloat16
        # Mapping is easiest projection-first (shape varies with p). The final
        # return transposes these builders to the runtime's [G/N][P] indexing.
        projection_landings: list[tuple[torch.Tensor, ...]] = []
        projection_globals: list[tuple[torch.Tensor, ...]] = []
        projection_locals: list[tuple[torch.Tensor, ...]] = []
        projection_grad_locals: list[tuple[torch.Tensor, ...]] = []
        projection_distributed_grads: list[tuple[torch.Tensor, ...]] = []

        for projection, chunk_shape in enumerate(allocations["chunk_shapes"]):
            duplicate_weight_fd = allocations["duplicate_weights"][projection][1]
            projection_landings.append(tuple(tensor for tensor, _ in allocations["home_weights"][projection]))

            global_generations: list[torch.Tensor] = []
            local_generations: list[torch.Tensor] = []
            for all_home_fds in home_weight_graph[projection]:
                # Communication addresses all E home chunks followed by this
                # rank's B duplicate chunks. Grouped GEMM instead receives the
                # zero-copy local [home B, duplicate B] alias.
                global_generations.append(
                    nvl_dist_map(
                        chunk_shape=list(chunk_shape),
                        dtype=dtype,
                        fds=[*all_home_fds, duplicate_weight_fd],
                        local_rank=ep_rank,
                        world_size=ep_size + 1,
                    )
                )
                local_generations.append(
                    nvl_dist_map(
                        chunk_shape=list(chunk_shape),
                        dtype=dtype,
                        fds=[all_home_fds[ep_rank], duplicate_weight_fd],
                        local_rank=0,
                        world_size=2,
                    )
                )
            projection_globals.append(tuple(global_generations))
            projection_locals.append(tuple(local_generations))

            grad_locals: list[torch.Tensor] = []
            distributed_grads: list[torch.Tensor] = []
            for slot, all_duplicate_fds in enumerate(duplicate_gradient_graph[projection]):
                home_fd = allocations["home_gradients"][projection][slot][1]
                duplicate_fd = allocations["duplicate_gradients"][projection][slot][1]
                # Grouped GEMM writes one contiguous [home, duplicate] segment.
                # Owners also map every rank's duplicate chunk for in-place
                # BF16 gradient return.
                grad_locals.append(
                    nvl_dist_map(
                        chunk_shape=list(chunk_shape),
                        dtype=dtype,
                        fds=[home_fd, duplicate_fd],
                        local_rank=0,
                        world_size=2,
                    )
                )
                distributed_grads.append(
                    nvl_dist_map(
                        chunk_shape=list(chunk_shape),
                        dtype=dtype,
                        fds=list(all_duplicate_fds),
                        local_rank=ep_rank,
                        world_size=ep_size,
                    ).view(ep_size, *chunk_shape)
                )
            projection_grad_locals.append(tuple(grad_locals))
            projection_distributed_grads.append(tuple(distributed_grads))

        return _WorkspaceLayout(
            landings=tuple(zip(*projection_landings, strict=True)),
            global_weights=tuple(zip(*projection_globals, strict=True)),
            local_weights=tuple(zip(*projection_locals, strict=True)),
            local_grad_outputs=tuple(zip(*projection_grad_locals, strict=True)),
            distributed_duplicate_grads=tuple(zip(*projection_distributed_grads, strict=True)),
            keepalives=allocations["keepalives"],
        )

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def landing(self, generation: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projection-paired FSDP targets, each ``[B, O_p, I_p]``."""
        if self._destroyed:
            raise RuntimeError("MoonEP workspace has been destroyed")
        if generation not in (0, 1):
            raise ValueError(f"generation must be 0 or 1, got {generation}")
        return cast(tuple[torch.Tensor, torch.Tensor], self._landings[generation])

    def local_token_counts(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Convert ``[E+B]`` endpoints to ``[2B]`` home/duplicate counts."""
        if self._destroyed:
            raise RuntimeError("MoonEP workspace has been destroyed")
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.numel() != (self._num_experts + self._experts_per_rank):
            raise ValueError("cu_seqlens must be int32 with E+B cumulative endpoints")

        # Static device slices select home B and duplicate B.  No route value
        # reaches the host and activation rows are not repacked.
        counts = torch.diff(cu_seqlens, prepend=torch.zeros_like(cu_seqlens[:1]))
        home_start = self._ep_rank * self._experts_per_rank
        return torch.cat(
            (
                counts[home_start : home_start + self._experts_per_rank],
                counts[self._num_experts : self._num_experts + self._experts_per_rank],
            )
        )

    def prefetch_weights(self, *, buffer, plan, generation: int, grad_slot: int):
        """Prefetch ``[E+B]`` weights and return local ``[2B]`` weight/WGrad
        pairs."""
        if self._destroyed:
            raise RuntimeError("MoonEP workspace has been destroyed")
        if generation not in (0, 1):
            raise ValueError(f"generation must be 0 or 1, got {generation}")
        if not 0 <= grad_slot < self._gradient_slots:
            raise ValueError(f"gradient slot out of range: {grad_slot}")
        buffer.prefetch_weight(
            plan=plan,
            projections=self._global_weights[generation],
            async_finish=False,
        )
        # A slot is reused sequentially across physical layers.  Each
        # invocation receives a fresh TensorImpl/version counter over the same
        # VMM storage, avoiding both payload allocation and AOT version clashes.
        grad_outputs = tuple(
            target.new_empty(0).set_(
                target.untyped_storage(),
                target.storage_offset(),
                target.shape,
                target.stride(),
            )
            for target in self._local_grad_outputs[grad_slot]
        )
        return self._local_weights[generation], grad_outputs

    def complete_gradients(self, *, buffer, plan, local_grads, grad_slot: int):
        """Reduce ``[2B]`` WGrads and return their local-home ``[B]``
        prefixes."""
        if self._destroyed:
            raise RuntimeError("MoonEP workspace has been destroyed")
        if not 0 <= grad_slot < self._gradient_slots:
            raise ValueError(f"gradient slot out of range: {grad_slot}")
        local_grads = tuple(local_grads)
        targets = self._local_grad_outputs[grad_slot]
        if len(local_grads) != 2:
            raise ValueError("local_grads must contain the two fused projections")
        for actual, target in zip(local_grads, targets, strict=True):
            if (
                actual.dtype != torch.bfloat16
                or actual.shape != target.shape
                or actual.data_ptr() != target.data_ptr()
            ):
                raise ValueError("local_grads must be the selected workspace gradient slot")

        buffer.reduce_grad_bf16(
            plan=plan,
            local_grads=local_grads,
            distributed_duplicate_grads=self._distributed_duplicate_grads[grad_slot],
            async_finish=False,
        )
        b = self._experts_per_rank
        return local_grads[0][:b], local_grads[1][:b]

    def destroy(self) -> None:
        """Release mappings at an explicit, rank-coordinated boundary."""
        if self._destroyed:
            return
        torch.cuda.synchronize()
        dist.barrier(group=self._ep_group)
        self._landings = ()
        self._global_weights = ()
        self._local_weights = ()
        self._local_grad_outputs = ()
        self._distributed_duplicate_grads = ()
        self._keepalives = ()
        self._destroyed = True

    def __del__(self) -> None:
        if getattr(self, "_destroyed", True):
            return
        warnings.warn(
            "MoonEP workspace was not destroyed explicitly; resources may leak.",
            ResourceWarning,
        )
        # Keep mappings alive instead of tearing CUDA/VMM state down after
        # distributed ranks may have diverged during interpreter shutdown.
        _UNDISPOSED_WORKSPACE_TENSORS.append(
            (
                self._landings,
                self._global_weights,
                self._local_weights,
                self._local_grad_outputs,
                self._distributed_duplicate_grads,
                self._keepalives,
            )
        )
        self._landings = ()
        self._global_weights = ()
        self._local_weights = ()
        self._local_grad_outputs = ()
        self._distributed_duplicate_grads = ()
        self._keepalives = ()
