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

import torch
import torch.distributed as dist


_UNDISPOSED_WORKSPACE_TENSORS: list[object] = []


class _ExpertVMMWorkspace:
    """Hide the VMM mappings needed by one XTuner model and EP group."""

    def __init__(
        self,
        *,
        landings,
        global_weights,
        local_weights,
        local_grad_outputs,
        distributed_duplicate_grads,
        keepalives,
        ep_group: dist.ProcessGroup,
        ep_rank: int,
        num_experts: int,
        experts_per_rank: int,
    ) -> None:
        self._landings = landings
        self._global_weights = global_weights
        self._local_weights = local_weights
        self._local_grad_outputs = local_grad_outputs
        self._distributed_duplicate_grads = distributed_duplicate_grads
        self._keepalives = keepalives
        self._ep_group = ep_group
        self._ep_rank = ep_rank
        self._num_experts = num_experts
        self._experts_per_rank = experts_per_rank
        self._gradient_slots = len(local_grad_outputs)
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
        """Allocate the fixed BF16, two-generation, one-segment layout."""
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

        # Import the optional backend only at resource installation time.  A
        # normal XTuner import or meta model build remains MoonEP-independent.
        from moonep._C import (
            get_vmm_granularity,
            nvl_dist_alloc,
            nvl_dist_map,
            nvl_release_mem_handle,
        )
        from moonep.buffer import _exchange_ipc_fds

        dtype = torch.bfloat16
        home_generations = 2
        home_experts = num_experts // ep_size
        granularity = get_vmm_granularity()
        projection_landings: list[list[torch.Tensor]] = []
        projection_globals: list[list[torch.Tensor]] = []
        projection_locals: list[list[torch.Tensor]] = []
        projection_grad_locals: list[list[torch.Tensor]] = []
        projection_distributed_grads: list[list[torch.Tensor]] = []
        keepalives: list[torch.Tensor] = []
        for projection_shape in projection_shapes:
            chunk_shape = [home_experts, *projection_shape]
            chunk_bytes = torch.empty((), dtype=dtype).element_size()
            for dim in chunk_shape:
                chunk_bytes *= dim
            if chunk_bytes % granularity:
                raise ValueError(
                    f"expert home chunk requires {granularity}-byte VMM alignment, "
                    f"got {chunk_bytes} bytes for shape {tuple(chunk_shape)}"
                )

            duplicate, duplicate_fd, duplicate_handle = nvl_dist_alloc(shape=chunk_shape, dtype=dtype)
            home_allocations = [nvl_dist_alloc(shape=chunk_shape, dtype=dtype) for _ in range(home_generations)]
            keepalives.append(duplicate)
            generations: list[torch.Tensor] = []
            global_generations: list[torch.Tensor] = []
            local_generations: list[torch.Tensor] = []
            try:
                for landing, home_fd, _ in home_allocations:
                    keepalives.append(landing)
                    generations.append(landing)
                    exchanged = _exchange_ipc_fds(
                        home_fd,
                        list(range(ep_size)),
                        ep_rank,
                        ep_size,
                        ep_group,
                    )
                    all_home_fds = [exchanged[rank] for rank in range(ep_size)]
                    try:
                        # Communication addresses all E home chunks followed
                        # by this rank's B duplicate chunks.  The one-segment
                        # grouped GEMM sees the zero-copy [home B, duplicate B]
                        # alias below, so no packing copy or host decision is
                        # introduced by the different logical views.
                        global_generations.append(
                            nvl_dist_map(
                                chunk_shape=chunk_shape,
                                dtype=dtype,
                                fds=[*all_home_fds, duplicate_fd],
                                local_rank=ep_rank,
                                world_size=ep_size + 1,
                            )
                        )
                        local_generations.append(
                            nvl_dist_map(
                                chunk_shape=chunk_shape,
                                dtype=dtype,
                                fds=[all_home_fds[ep_rank], duplicate_fd],
                                local_rank=0,
                                world_size=2,
                            )
                        )
                    finally:
                        for fd in all_home_fds:
                            os.close(fd)
            finally:
                os.close(duplicate_fd)
                nvl_release_mem_handle(duplicate_handle)
                for _, home_fd, home_handle in home_allocations:
                    os.close(home_fd)
                    nvl_release_mem_handle(home_handle)

            projection_landings.append(generations)
            projection_globals.append(global_generations)
            projection_locals.append(local_generations)

            grad_locals: list[torch.Tensor] = []
            distributed_grads: list[torch.Tensor] = []
            for _ in range(gradient_slots):
                home_grad, home_fd, home_handle = nvl_dist_alloc(shape=chunk_shape, dtype=dtype)
                duplicate_grad, duplicate_fd, duplicate_handle = nvl_dist_alloc(shape=chunk_shape, dtype=dtype)
                keepalives.extend((home_grad, duplicate_grad))
                try:
                    exchanged = _exchange_ipc_fds(
                        duplicate_fd,
                        list(range(ep_size)),
                        ep_rank,
                        ep_size,
                        ep_group,
                    )
                    all_duplicate_fds = [exchanged[rank] for rank in range(ep_size)]
                    try:
                        # Grouped GEMM writes one contiguous [home, duplicate]
                        # segment.  Owners separately map every rank's duplicate
                        # chunk so MoonEP can return BF16 gradients in place.
                        grad_locals.append(
                            nvl_dist_map(
                                chunk_shape=chunk_shape,
                                dtype=dtype,
                                fds=[home_fd, duplicate_fd],
                                local_rank=0,
                                world_size=2,
                            )
                        )
                        distributed_grads.append(
                            nvl_dist_map(
                                chunk_shape=chunk_shape,
                                dtype=dtype,
                                fds=all_duplicate_fds,
                                local_rank=ep_rank,
                                world_size=ep_size,
                            ).view(ep_size, home_experts, *projection_shape)
                        )
                    finally:
                        for fd in all_duplicate_fds:
                            os.close(fd)
                finally:
                    os.close(home_fd)
                    os.close(duplicate_fd)
                    nvl_release_mem_handle(home_handle)
                    nvl_release_mem_handle(duplicate_handle)

            projection_grad_locals.append(grad_locals)
            projection_distributed_grads.append(distributed_grads)

        landings = tuple(
            tuple(projection_landings[projection][generation] for projection in range(2))
            for generation in range(home_generations)
        )
        global_weights = tuple(
            tuple(projection_globals[projection][generation] for projection in range(2))
            for generation in range(home_generations)
        )
        local_weights = tuple(
            tuple(projection_locals[projection][generation] for projection in range(2))
            for generation in range(home_generations)
        )
        local_grad_outputs = tuple(
            tuple(projection_grad_locals[projection][slot] for projection in range(2))
            for slot in range(gradient_slots)
        )
        distributed_duplicate_grads = tuple(
            tuple(projection_distributed_grads[projection][slot] for projection in range(2))
            for slot in range(gradient_slots)
        )
        return cls(
            landings=landings,
            global_weights=global_weights,
            local_weights=local_weights,
            local_grad_outputs=local_grad_outputs,
            distributed_duplicate_grads=distributed_duplicate_grads,
            keepalives=tuple(keepalives),
            ep_group=ep_group,
            ep_rank=ep_rank,
            num_experts=num_experts,
            experts_per_rank=home_experts,
        )

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def landing(self, generation: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._destroyed:
            raise RuntimeError("MoonEP workspace has been destroyed")
        if generation not in (0, 1):
            raise ValueError(f"generation must be 0 or 1, got {generation}")
        return self._landings[generation]

    def local_token_counts(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Select device-resident counts for the one [home, duplicate]
        segment."""
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

    def materialize(self, *, buffer, plan, generation: int, grad_slot: int):
        """Prefetch weights and expose the invocation's one-segment aliases."""
        if self._destroyed:
            raise RuntimeError("MoonEP workspace has been destroyed")
        if generation not in (0, 1):
            raise ValueError(f"generation must be 0 or 1, got {generation}")
        if not 0 <= grad_slot < self._gradient_slots:
            raise ValueError(f"gradient slot out of range: {grad_slot}")
        done = buffer.prefetch_weight(
            plan=plan,
            projections=self._global_weights[generation],
            async_finish=True,
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
        return self._local_weights[generation], grad_outputs, done

    def complete_gradients(self, *, buffer, plan, local_grads, grad_slot: int):
        """Return duplicate BF16 partials into the local home gradients."""
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

        done = buffer.reduce_grad(
            plan=plan,
            local_grads=local_grads,
            distributed_duplicate_grads=self._distributed_duplicate_grads[grad_slot],
            accumulation_dtype=torch.float32,
            async_finish=True,
        )
        b = self._experts_per_rank
        return (local_grads[0][:b], local_grads[1][:b]), done

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
