"""Optional UltraEP runtime integration for Xtuner MoE layers."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Protocol

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


_PHASE_NVTX_ENABLED = (
    os.getenv("XTUNER_NSYS_PHASE_NVTX", "0") == "1"
    and torch.cuda.is_available()
    and hasattr(torch.cuda, "nvtx")
)


class _PhaseNvtxRange:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.cuda.nvtx.range_pop()


def _phase_nvtx(name: str):
    """Emit optional CPU-side NVTX ranges for phase-aligned Nsight traces."""
    if not _PHASE_NVTX_ENABLED:
        return nullcontext()
    return _PhaseNvtxRange(name)


if TYPE_CHECKING:
    from ultra_ep import EventHandle, Manager

    from xtuner.v1.model.moe.moe import MoEConfig


class UltraEPGroupedLinear(Protocol):
    """The small grouped-linear surface owned by one UltraEP layer binding."""

    weight: torch.Tensor

    def configure_ultra_ep_buffers(self, replica_weight: torch.Tensor, replica_grad: torch.Tensor) -> None: ...

    def select_ultra_ep_slot(self, slot: int) -> None: ...


class UltraEPManager:
    """One UltraEP Manager and its shared replica slots per EP group."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        num_layers: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        expert_fc1_numel: int,
        expert_fc2_numel: int,
        max_microbatches: int,
    ) -> None:
        try:
            import ultra_ep
        except ImportError as exc:
            raise ImportError(
                "UltraEP is enabled but its Python package/CUDA extension is unavailable. "
                "Build UltraEP outside the Xtuner environment and prepend its build/lib.* directory to PYTHONPATH."
            ) from exc

        if group.size() <= 1:
            raise ValueError("UltraEP requires an EP process group with size > 1")
        if max_microbatches <= 0:
            raise ValueError("UltraEP requires max_microbatches > 0")
        world_size = dist.get_world_size()
        if world_size % group.size() != 0:
            raise ValueError(
                f"UltraEP EP group size {group.size()} must evenly divide world size {world_size}"
            )
        self.dp_size = world_size // group.size()

        self.group = group
        self.num_layers = num_layers
        self.num_local_master_experts = num_local_master_experts
        self.num_local_redundant_experts = num_local_redundant_experts
        self.expert_fc1_numel = expert_fc1_numel
        self.expert_fc2_numel = expert_fc2_numel
        self.max_microbatches = max_microbatches
        try:
            self.runtime: Manager = ultra_ep.Manager(
                group=group,
                num_layers=num_layers,
                num_local_master_experts=num_local_master_experts,
                num_local_redundant_experts=num_local_redundant_experts,
                expert_fc1_numel=expert_fc1_numel,
                expert_fc2_numel=expert_fc2_numel,
                is_train=True,
                explicitly_destroy=False,
                max_microbatches=max_microbatches,
                weight_data_dtype=torch.bfloat16,
                grad_dtype=torch.bfloat16,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "UltraEP FSDP requires the native BF16 grad-reduce patch; "
                "the loaded UltraEP extension rejected grad_dtype=torch.bfloat16"
            ) from exc
        # FSDP expert gradients and the native UltraEP grad-reduce path use
        # BF16.  One staging pair is sufficient because the handoff join
        # completes before the next layer claims the pair.
        device = torch.device("cuda", torch.cuda.current_device())
        self.master_fc1_grad_staging = torch.empty(
            num_local_master_experts,
            expert_fc1_numel,
            dtype=torch.bfloat16,
            device=device,
        )
        self.master_fc2_grad_staging = torch.empty(
            num_local_master_experts,
            expert_fc2_numel,
            dtype=torch.bfloat16,
            device=device,
        )
        self._staging_owner: int | None = None
        self._master_weight_ptr_hosts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # FSDP may call ``sync_weights`` repeatedly while the current
        # unsharded view is still backed by the same allocation.  Keep the
        # last addresses so pointer-pool refresh becomes a no-op in that
        # case.  This does not cache weight contents: ``weight_sync`` still
        # runs every time and copies the current values to replica slots.
        self._master_weight_ptr_values: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    @property
    def local_replica_fc1_weight_buffer(self) -> torch.Tensor:
        return self.runtime.local_replica_fc1_weight_buffer

    @property
    def local_replica_fc2_weight_buffer(self) -> torch.Tensor:
        return self.runtime.local_replica_fc2_weight_buffer

    @property
    def local_replica_fc1_grad_buffer(self) -> torch.Tensor:
        return self.runtime.local_replica_fc1_grad_buffer

    @property
    def local_replica_fc2_grad_buffer(self) -> torch.Tensor:
        return self.runtime.local_replica_fc2_grad_buffer

    def allocate_microbatch_slot(self, layer_id: int) -> int:
        return self.runtime.allocate_microbatch_slot(layer_id)

    def replica_slot(self, virtual_layer_id: int) -> int:
        """Return the buffer slot encoded by a native virtual layer id."""
        return int(self.runtime.microbatch_slot(virtual_layer_id))

    def update_placement_sparse(self, layer_id: int, logical_topk_ids: torch.Tensor) -> None:
        self.runtime.update_placement_sparse(layer_id, logical_topk_ids)

    def reroute_sparse(self, layer_id: int, physical_topk_ids: torch.Tensor) -> None:
        self.runtime.reroute_sparse(layer_id, physical_topk_ids)

    @staticmethod
    def _local(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def stage_master_gradients(
        self,
        *,
        virtual_layer_id: int,
        fc1_grad: torch.Tensor,
        fc2_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy the current master BF16 gradients into shared staging."""
        if self._staging_owner is not None:
            raise RuntimeError(
                "UltraEP BF16 master-grad staging is still owned by virtual layer "
                f"{self._staging_owner}; attempted to start {virtual_layer_id}"
            )
        local_fc1 = self._local(fc1_grad).view(self.num_local_master_experts, -1)
        local_fc2 = self._local(fc2_grad).view(self.num_local_master_experts, -1)
        if (
            local_fc1.numel() != self.master_fc1_grad_staging.numel()
            or local_fc2.numel() != self.master_fc2_grad_staging.numel()
        ):
            raise ValueError("Master expert gradient shapes do not match UltraEP BF16 staging")
        with torch.profiler.record_function("UltraEP::staging_copy"):
            self.master_fc1_grad_staging.copy_(local_fc1)
            self.master_fc2_grad_staging.copy_(local_fc2)
        self._staging_owner = virtual_layer_id
        return self.master_fc1_grad_staging, self.master_fc2_grad_staging

    def restore_master_gradients(
        self,
        *,
        virtual_layer_id: int,
        fc1_grad: torch.Tensor,
        fc2_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the completed staging tensors and release the slot."""
        if self._staging_owner != virtual_layer_id:
            raise RuntimeError(f"UltraEP staging owner is {self._staging_owner}, not {virtual_layer_id}")
        result = (self.master_fc1_grad_staging, self.master_fc2_grad_staging)
        self._staging_owner = None
        return result

    def register_master_pointers(
        self,
        *,
        layer_id: int,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc1_grad: torch.Tensor,
        fc2_grad: torch.Tensor,
    ) -> None:
        fc1_weight = self._local(fc1_weight).view(self.num_local_master_experts, -1)
        fc2_weight = self._local(fc2_weight).view(self.num_local_master_experts, -1)
        if fc1_weight.shape[1] != self.expert_fc1_numel or fc2_weight.shape[1] != self.expert_fc2_numel:
            raise ValueError("Master expert weight shapes do not match the UltraEP Manager configuration")

        fc1_grad = self._local(fc1_grad).view(self.num_local_master_experts, -1)
        fc2_grad = self._local(fc2_grad).view(self.num_local_master_experts, -1)
        if fc1_grad.dtype != torch.bfloat16 or fc2_grad.dtype != torch.bfloat16:
            raise TypeError(f"UltraEP requires BF16 master grads, got {fc1_grad.dtype} and {fc2_grad.dtype}")
        fc1_grads = list(fc1_grad.unbind(0))
        fc2_grads = list(fc2_grad.unbind(0))

        self.runtime.construct_local_master_ptr_pool(
            layer_id=layer_id,
            fc1_weights=list(fc1_weight.unbind(0)),
            fc2_weights=list(fc2_weight.unbind(0)),
            fc1_grads=fc1_grads,
            fc2_grads=fc2_grads,
        )
        # Xtuner FSDP may replace the local parameter storage between gradient
        # accumulation microbatches even with reshard_after_forward=False. Keep
        # reusable pinned host arrays so each weight_sync can refresh only the
        # two device pointer arrays without rebuilding all weight/grad pools.
        self._master_weight_ptr_hosts[layer_id] = (
            torch.empty(
                self.num_local_master_experts,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            ),
            torch.empty(
                self.num_local_master_experts,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            ),
        )
        self._master_weight_ptr_values.pop(layer_id, None)

    def refresh_master_weight_pointers(
        self,
        *,
        layer_id: int,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> None:
        """Refresh only FSDP-movable master-weight addresses in cached
        pools."""
        with torch.profiler.record_function("UltraEP::pointer_refresh"):
            hosts = self._master_weight_ptr_hosts.get(layer_id)
            if hosts is None:
                raise RuntimeError(f"UltraEP master pointer pool for layer {layer_id} is not registered")
            fc1_weight = self._local(fc1_weight).view(self.num_local_master_experts, -1)
            fc2_weight = self._local(fc2_weight).view(self.num_local_master_experts, -1)
            fc1_ptrs = tuple(weight.data_ptr() for weight in fc1_weight.unbind(0))
            fc2_ptrs = tuple(weight.data_ptr() for weight in fc2_weight.unbind(0))
            if self._master_weight_ptr_values.get(layer_id) == (fc1_ptrs, fc2_ptrs):
                # Keep a distinct range so Nsight can report the cache hit
                # rate without counting this fast path as a device copy.
                if os.getenv("XTUNER_NSYS_POINTER_NVTX", "0") == "1" and torch.cuda.is_available():
                    torch.cuda.nvtx.range_push("UltraEP::pointer_refresh_skip")
                    torch.cuda.nvtx.range_pop()
                return
            fc1_host, fc2_host = hosts
            for expert_idx, (fc1_ptr, fc2_ptr) in enumerate(zip(fc1_ptrs, fc2_ptrs, strict=True)):
                fc1_host[expert_idx] = fc1_ptr
                fc2_host[expert_idx] = fc2_ptr

            fc1_device = self.runtime.local_master_fc1_weight_ptr_pool[layer_id]
            fc2_device = self.runtime.local_master_fc2_weight_ptr_pool[layer_id]
            if fc1_device is None or fc2_device is None:
                raise RuntimeError(f"UltraEP device pointer pool for layer {layer_id} is unavailable")
            fc1_device.copy_(fc1_host, non_blocking=True)
            fc2_device.copy_(fc2_host, non_blocking=True)
            self._master_weight_ptr_values[layer_id] = (fc1_ptrs, fc2_ptrs)

    def weight_sync(self, layer_id: int, *, async_finish: bool) -> EventHandle:
        return self.runtime.weight_sync(layer_id=layer_id, async_finish=async_finish)

    def grad_reduce(self, layer_id: int, *, async_finish: bool) -> EventHandle:
        return self.runtime.grad_reduce(layer_id=layer_id, async_finish=async_finish)


class UltraEPManagerProvider:
    """Lazy process-group-level owner of one shared UltraEP Manager."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        num_model_layers: int,
        num_logical_experts: int,
        hidden_size: int,
        expert_intermediate_size: int,
        num_redundant_experts_per_rank: int,
        max_microbatches: int,
    ) -> None:
        if group.size() <= 1:
            raise ValueError("UltraEP requires an EP process group with size > 1")
        if num_logical_experts % group.size() != 0:
            raise ValueError("UltraEP requires logical experts to be evenly sharded by EP")
        if num_model_layers <= 0:
            raise ValueError("UltraEP requires num_model_layers > 0")
        if max_microbatches <= 0:
            raise ValueError("UltraEP requires max_microbatches > 0")

        self.group = group
        self.num_model_layers = num_model_layers
        self.num_logical_experts = num_logical_experts
        self.hidden_size = hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_redundant_experts_per_rank = num_redundant_experts_per_rank
        self.max_microbatches = max_microbatches
        self._manager: UltraEPManager | None = None
        self._state = "CREATED"
        self._fsdp_root = None

    @classmethod
    def from_xtuner_config(
        cls,
        *,
        group: dist.ProcessGroup,
        config: MoEConfig,
        max_microbatches: int = 1,
    ) -> UltraEPManagerProvider:
        """Build a provider from Xtuner's model config.

        The native manager allocates placement/virtual-layer slots at
        construction time.  Xtuner normally does not know the requested
        intra-layer micro-batch count until the :class:`TrainEngine` starts a
        model call, so callers may increase the capacity with
        :meth:`configure_max_microbatches` before the first manager access.
        """
        ultraep_cfg = config.ultraep_cfg
        if ultraep_cfg is None:
            raise ValueError("UltraEP manager provider requires config.ultraep_cfg")

        return cls(
            group=group,
            num_model_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            expert_intermediate_size=config.moe_intermediate_size,
            num_redundant_experts_per_rank=ultraep_cfg.num_redundant_experts_per_rank,
            max_microbatches=max_microbatches,
        )

    def _requires_reshard_after_forward(self, fsdp_config) -> bool:
        """Return whether UltraEP needs a closed FSDP unshard window.

        EP4×DP2 (and HSDP) shards the same local expert across a DP mesh.
        That path was only validated with ``reshard_after_forward=True``.
        EP-only (``world_size == ep_size``) historically keeps parameters
        unsharded after forward; UltraEP refreshes FSDP-movable master
        pointers instead of requiring a reshard.
        """
        if getattr(fsdp_config, "hsdp_sharding_size", None):
            return True
        ep_size = int(getattr(fsdp_config, "ep_size", 0) or 0)
        if ep_size <= 0 or not (dist.is_available() and dist.is_initialized()):
            return False
        world_size = dist.get_world_size()
        if world_size % ep_size != 0:
            raise ValueError(
                f"UltraEP FSDP ep_size={ep_size} must evenly divide world_size={world_size}"
            )
        return world_size > ep_size

    def validate_before_fsdp(self, fsdp_config) -> None:
        """Validate the FSDP settings required by the BF16 handoff path."""
        if fsdp_config.param_dtype is not torch.bfloat16 or fsdp_config.reduce_dtype is not torch.bfloat16:
            raise ValueError("UltraEP FSDP requires param_dtype=reduce_dtype=torch.bfloat16")
        if fsdp_config.cpu_offload:
            raise ValueError("UltraEP FSDP does not support CPU offload")
        if not getattr(fsdp_config, "requires_grad", True):
            raise ValueError("UltraEP FSDP requires requires_grad=True")
        if self._requires_reshard_after_forward(fsdp_config) and not getattr(
            fsdp_config, "reshard_after_forward", True
        ):
            ep_size = int(getattr(fsdp_config, "ep_size", 0) or 0)
            world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else "<unknown>"
            raise ValueError(
                "UltraEP FSDP with DP>1 requires reshard_after_forward=True "
                f"(ep_size={ep_size}, world_size={world})"
            )
        if getattr(fsdp_config, "recompute_ratio", 0) > 0:
            raise ValueError("UltraEP FSDP does not support activation recompute")

    def install_after_fsdp(self, *, fsdp_root, targets) -> None:
        """Install the FSDP identity seam before exposing native resources.

        Binding and state transition are transactional: a failed binding leaves
        the provider in CREATED, so a later retry cannot observe a half-installed
        provider or materialize a manager against stale parameter identities.
        """
        if self._state == "CLOSED":
            raise RuntimeError("UltraEP provider is closed")
        if self._state == "CREATED":
            from .fsdp_expert_binding import install_ultraep_fsdp_binding

            binding = install_ultraep_fsdp_binding(fsdp_root=fsdp_root, targets=targets)
            self._fsdp_root = fsdp_root
            self.fsdp_binding = binding
            self._state = "INSTALLED"
        elif self._fsdp_root is not fsdp_root:
            raise RuntimeError("UltraEP provider was installed on a different FSDP root")

    def ensure_materialized(self) -> UltraEPManager:
        if self._state == "CLOSED":
            raise RuntimeError("UltraEP provider is closed")
        if self._state == "CREATED":
            raise RuntimeError(
                "UltraEP provider must be installed after FSDP setup before materialization"
            )
        manager = self.get_manager()
        if self._state == "INSTALLED":
            self._state = "MATERIALIZED"
        return manager

    def close(self) -> None:
        if self._state == "CLOSED":
            return
        # Remove only the XTuner/FSDP identity seam.  Native UltraEP owns a
        # process-level NVSHMEM singleton, so its manager, registry entry and
        # staging tensors must stay alive until process teardown.
        binding = getattr(self, "fsdp_binding", None)
        if binding is not None:
            from .fsdp_expert_binding import uninstall_ultraep_fsdp_binding

            uninstall_ultraep_fsdp_binding(binding)
            self.fsdp_binding = None
        self._state = "CLOSED"
        self._fsdp_root = None

    def configure_max_microbatches(self, requested: int) -> None:
        """Set the native virtual-layer capacity before materialization.

        A manager owns one shared replica weight/gradient buffer.  Its native
        placement tables nevertheless need one virtual id per in-flight
        micro-batch.  Capacity is therefore fixed once the manager is created;
        changing it afterwards would invalidate already allocated ids.
        """
        if requested <= 0:
            raise ValueError(f"UltraEP microbatch capacity must be > 0, got {requested}")
        requested = max(1, int(requested))
        if self._manager is not None and requested > self.max_microbatches:
            raise RuntimeError(
                "UltraEP manager was already materialized with insufficient virtual-layer capacity: "
                f"existing={self.max_microbatches}, requested={requested}. "
                "Configure intra_layer_micro_batch before the first UltraEP forward."
            )
        self.max_microbatches = max(self.max_microbatches, requested)

    @property
    def num_dispatch_experts(self) -> int:
        """Global physical-expert count expected by the dispatcher."""
        return self.num_logical_experts + self.group.size() * self.num_redundant_experts_per_rank

    def get_manager(self) -> UltraEPManager:
        if self._state == "CLOSED":
            raise RuntimeError("UltraEP provider is closed")
        if self._state == "CREATED":
            raise RuntimeError(
                "UltraEP provider must be installed after FSDP setup before materialization"
            )
        if self._manager is None:
            self._manager = get_or_create_ultra_ep_manager(
                group=self.group,
                num_layers=self.num_model_layers,
                num_local_master_experts=self.num_logical_experts // self.group.size(),
                num_local_redundant_experts=self.num_redundant_experts_per_rank,
                expert_fc1_numel=2 * self.expert_intermediate_size * self.hidden_size,
                expert_fc2_numel=self.hidden_size * self.expert_intermediate_size,
                max_microbatches=self.max_microbatches,
            )
        return self._manager


class UltraEPLayerRuntime:
    """Runtime-only UltraEP binding for one MoE layer.

    The decoder owns ordinary model modules and the autograd graph boundaries. This object owns every interaction with
    the process-group-level UltraEP manager and never registers a tensor as model state.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        manager_provider: UltraEPManagerProvider,
        fused_w1w3: UltraEPGroupedLinear,
        fused_w2: UltraEPGroupedLinear,
    ) -> None:
        if layer_id < 0 or layer_id >= manager_provider.num_model_layers:
            raise ValueError(f"UltraEP layer_id must be in [0, {manager_provider.num_model_layers}), got {layer_id}")

        self.layer_id = layer_id
        self.manager_provider = manager_provider
        self.num_logical_experts = manager_provider.num_logical_experts
        self.hidden_size = manager_provider.hidden_size
        self.expert_intermediate_size = manager_provider.expert_intermediate_size
        self.num_redundant_experts_per_rank = manager_provider.num_redundant_experts_per_rank
        self.max_microbatches = manager_provider.max_microbatches
        self.fused_w1w3 = fused_w1w3
        self.fused_w2 = fused_w2

        self._buffers_configured = False
        self._master_pointers_registered = False
        self._grad_reduce_events: dict[int, tuple[object, torch.Tensor, torch.Tensor]] = {}
        # Restore is deliberately split into a launch and a join.  Replica
        # slots are mutable communication buffers, so the launch can safely
        # overlap combine backward while the join remains immediately before
        # expert DGrad (the first consumer of the restored weights).
        self._weight_restore_events: dict[int, object | None] = {}

    @property
    def num_dispatch_experts(self) -> int:
        """Global physical-expert count expected by the dispatcher."""
        return self.manager_provider.num_dispatch_experts

    def validate_microbatch_capacity(self, requested_microbatches: int) -> None:
        """Fail before allocation rather than silently reusing a virtual
        slot."""
        if requested_microbatches > self.max_microbatches:
            raise ValueError(
                "UltraEP virtual-layer capacity is too small for this layer call: "
                f"requested={requested_microbatches}, max_microbatches={self.max_microbatches}. "
                "UltraEP capacity is resolved from Trainer/TrainEngine.intra_layer_micro_batch."
            )

    def configure_max_microbatches(self, requested_microbatches: int) -> None:
        """Propagate call capacity to this layer and its shared provider."""
        self.manager_provider.configure_max_microbatches(requested_microbatches)
        self.max_microbatches = self.manager_provider.max_microbatches

    def allocate_virtual_layer_id(self) -> int:
        """Allocate the UltraEP virtual-layer slot for this forward
        microbatch."""
        return self._ensure_manager().allocate_microbatch_slot(self.layer_id)

    def update_placement(
        self,
        logical_topk_ids: torch.Tensor,
        virtual_layer_id: int,
    ) -> None:
        """Build the replication placement from logical expert IDs."""
        with _phase_nvtx("UltraEP::update_placement"), torch.profiler.record_function("UltraEP::placement"):
            self._ensure_manager().update_placement_sparse(virtual_layer_id, logical_topk_ids)

    def reroute(self, logical_topk_ids: torch.Tensor, virtual_layer_id: int) -> torch.Tensor:
        """Return a dispatcher-only copy rewritten into physical expert IDs."""
        with _phase_nvtx("UltraEP::reroute"), torch.profiler.record_function("UltraEP::reroute"):
            physical_topk_ids = logical_topk_ids.clone()
            self._ensure_manager().reroute_sparse(virtual_layer_id, physical_topk_ids)
        return physical_topk_ids

    def sync_weights(self, virtual_layer_id: int, *, async_finish: bool):
        with _phase_nvtx("UltraEP::weight_sync"), torch.profiler.record_function("UltraEP::weight_sync_launch"):
            manager = self._ensure_manager()
            fc1_weight, fc2_weight = self._current_expert_parameters()
            manager.refresh_master_weight_pointers(
                layer_id=self.layer_id,
                fc1_weight=fc1_weight,
                fc2_weight=fc2_weight,
            )
            return manager.weight_sync(virtual_layer_id, async_finish=async_finish)

    def start_weight_restore(self, virtual_layer_id: int) -> None:
        """Launch backward weight restore without waiting on the compute stream.

        The restore must happen after FSDP has materialized this layer's
        current master parameters, but it does not need to block combine
        backward: that path only consumes activations and routing metadata.
        The matching :meth:`finish_weight_restore` waits just before the
        grouped GEMM DGrad node reads the mutable replica slot.
        """
        events = getattr(self, "_weight_restore_events", None)
        if events is None:
            events = self._weight_restore_events = {}
        if virtual_layer_id in events:
            raise RuntimeError(f"UltraEP weight restore for virtual layer slot {virtual_layer_id} is still in use")
        with _phase_nvtx("UltraEP::weight_restore_start"), torch.profiler.record_function("UltraEP::restore_start"):
            events[virtual_layer_id] = self.sync_weights(virtual_layer_id, async_finish=True)

    def finish_weight_restore(self, virtual_layer_id: int) -> None:
        """Wait for a previously launched restore and select its replica slot."""
        events = getattr(self, "_weight_restore_events", None)
        if events is None or virtual_layer_id not in events:
            raise RuntimeError(f"UltraEP weight restore for virtual layer slot {virtual_layer_id} was not started")
        event = events.pop(virtual_layer_id)
        with _phase_nvtx("UltraEP::weight_restore_join"), torch.profiler.record_function("UltraEP::restore_wait"):
            if event is not None:
                # The real UltraEP EventHandle exposes ``event=None`` for the
                # synchronous path. Keep duck-typed fakes used by tests
                # working when they only provide ``current_stream_wait``.
                if not hasattr(event, "event") or event.event is not None:  # type: ignore[attr-defined]
                    event.current_stream_wait()  # type: ignore[attr-defined]
            # Selecting the slot only after the event dependency is installed
            # ensures the following grouped GEMM DGrad reads this vid's data.
            self.bind_virtual_layer_slot(virtual_layer_id)

    @staticmethod
    def _bind_staging_grad(parameter: torch.Tensor, staging: torch.Tensor) -> None:
        """Bind a BF16 staging view as the parameter gradient for FSDP."""
        local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        local_grad = staging.view_as(local_parameter)
        if isinstance(parameter, DTensor):
            parameter.grad = DTensor.from_local(
                local_grad, parameter.device_mesh, parameter.placements, run_check=False
            )
        else:
            parameter.grad = local_grad

    def start_grad_reduce(self, virtual_layer_id: int) -> None:
        if virtual_layer_id in self._grad_reduce_events:
            raise RuntimeError(f"UltraEP virtual layer slot {virtual_layer_id} is still in use")
        fc1_weight, fc2_weight = self._current_expert_parameters()
        fc1_grad = fc1_weight.grad
        fc2_grad = fc2_weight.grad
        if fc1_grad is None or fc2_grad is None:
            raise RuntimeError(
                f"UltraEP master gradients are unavailable at layer {self.layer_id}; "
                "the FSDP/autograd hook ordering is incompatible with replica grad-reduce"
            )
        manager = self._ensure_manager()
        with _phase_nvtx("UltraEP::grad_stage"), torch.profiler.record_function("UltraEP::grad_stage"):
            manager.stage_master_gradients(
                virtual_layer_id=virtual_layer_id,
                fc1_grad=fc1_grad,
                fc2_grad=fc2_grad,
            )
        with _phase_nvtx("UltraEP::grad_reduce_start"), torch.profiler.record_function("UltraEP::grad_reduce_launch"):
            event = manager.grad_reduce(virtual_layer_id, async_finish=True)
        self._grad_reduce_events[virtual_layer_id] = (event, fc1_grad, fc2_grad)

    def finish_grad_reduce(self, virtual_layer_id: int) -> None:
        state = self._grad_reduce_events.pop(virtual_layer_id, None)
        if state is None:
            raise RuntimeError(f"UltraEP grad-reduce event for virtual layer {virtual_layer_id} was not started")
        event, fc1_grad, fc2_grad = state
        if event is not None:
            # The real UltraEP EventHandle exposes ``event=None`` for the
            # synchronous path. Keep duck-typed test fakes working too.
            with _phase_nvtx("UltraEP::grad_reduce_join"), torch.profiler.record_function("UltraEP::grad_reduce_wait"):
                if not hasattr(event, "event") or event.event is not None:  # type: ignore[attr-defined]
                    event.current_stream_wait()  # type: ignore[attr-defined]
        with _phase_nvtx("UltraEP::grad_bind"), torch.profiler.record_function("UltraEP::grad_bind"):
            reduced = self._ensure_manager().restore_master_gradients(
                virtual_layer_id=virtual_layer_id,
                fc1_grad=fc1_grad,
                fc2_grad=fc2_grad,
            )
            # Legacy test doubles may perform the in-place copy and return None.
            if reduced is not None:
                fc1_staging, fc2_staging = reduced
                from .fsdp_expert_binding import fsdp_binding_installed

                if fsdp_binding_installed((self.fused_w1w3, self.fused_w2)):
                    from .fsdp_expert_binding import (
                        fsdp_current_unsharded_expert_parameters,
                        writeback_fsdp_unsharded_expert_gradients,
                    )

                    parameters = fsdp_current_unsharded_expert_parameters(
                        (self.fused_w1w3, self.fused_w2)
                    )
                    writeback_fsdp_unsharded_expert_gradients(
                        parameters,
                        (fc1_staging, fc2_staging),
                    )
                else:
                    self._bind_staging_grad(self.fused_w1w3.weight, fc1_staging)
                    self._bind_staging_grad(self.fused_w2.weight, fc2_staging)

    def _ensure_manager(self) -> UltraEPManager:
        manager = self.manager_provider.get_manager()
        if not self._buffers_configured:
            redundant = manager.num_local_redundant_experts
            slots = int(getattr(manager, "max_microbatches", self.max_microbatches))

            def slot_view(buffer: torch.Tensor, *shape: int) -> torch.Tensor:
                expected = slots * redundant
                if buffer.numel() != expected * math.prod(shape):
                    raise ValueError(
                        "UltraEP replica buffer capacity does not match the configured micro-batch slots: "
                        f"slots={slots}, redundant={redundant}, shape={shape}, numel={buffer.numel()}"
                    )
                return buffer.reshape(slots, redundant, *shape)

            self.fused_w1w3.configure_ultra_ep_buffers(
                slot_view(
                    manager.local_replica_fc1_weight_buffer,
                    2 * self.expert_intermediate_size,
                    self.hidden_size,
                ),
                slot_view(
                    manager.local_replica_fc1_grad_buffer,
                    2 * self.expert_intermediate_size,
                    self.hidden_size,
                ),
            )
            self.fused_w2.configure_ultra_ep_buffers(
                slot_view(
                    manager.local_replica_fc2_weight_buffer,
                    self.hidden_size,
                    self.expert_intermediate_size,
                ),
                slot_view(
                    manager.local_replica_fc2_grad_buffer,
                    self.hidden_size,
                    self.expert_intermediate_size,
                ),
            )
            self._buffers_configured = True

        if not self._master_pointers_registered:
            fc1_weight, fc2_weight = self._current_expert_parameters()
            manager.register_master_pointers(
                layer_id=self.layer_id,
                fc1_weight=fc1_weight,
                fc2_weight=fc2_weight,
                fc1_grad=manager.master_fc1_grad_staging,
                fc2_grad=manager.master_fc2_grad_staging,
            )
            self._master_pointers_registered = True
        return manager

    def _current_expert_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the current FSDP views when the optional seam is installed."""
        try:
            from .fsdp_expert_binding import fsdp_current_unsharded_expert_parameters

            return fsdp_current_unsharded_expert_parameters((self.fused_w1w3, self.fused_w2))
        except RuntimeError as exc:
            if "binding is not installed" not in str(exc):
                raise
            return self.fused_w1w3.weight, self.fused_w2.weight

    def bind_virtual_layer_slot(self, virtual_layer_id: int) -> None:
        """Select the replica slices used by this virtual layer's GEMMs."""
        manager = self._ensure_manager()
        slot = manager.replica_slot(virtual_layer_id)
        self.fused_w1w3.select_ultra_ep_slot(slot)
        self.fused_w2.select_ultra_ep_slot(slot)


_MANAGERS: dict[int, tuple[tuple[int, ...], UltraEPManager]] = {}


def get_or_create_ultra_ep_manager(
    *,
    group: dist.ProcessGroup,
    num_layers: int,
    num_local_master_experts: int,
    num_local_redundant_experts: int,
    expert_fc1_numel: int,
    expert_fc2_numel: int,
    max_microbatches: int,
) -> UltraEPManager:
    """Return the single Manager associated with this process-local EP
    group."""
    signature = (
        group.size(),
        num_layers,
        num_local_master_experts,
        num_local_redundant_experts,
        expert_fc1_numel,
        expert_fc2_numel,
        max_microbatches,
    )
    key = id(group)
    cached = _MANAGERS.get(key)
    if cached is not None:
        cached_signature, manager = cached
        if cached_signature != signature:
            raise RuntimeError(
                "All MoE layers sharing an EP group must use the same UltraEP shape/configuration: "
                f"existing={cached_signature}, requested={signature}"
            )
        return manager

    manager = UltraEPManager(
        group=group,
        num_layers=num_layers,
        num_local_master_experts=num_local_master_experts,
        num_local_redundant_experts=num_local_redundant_experts,
        expert_fc1_numel=expert_fc1_numel,
        expert_fc2_numel=expert_fc2_numel,
        max_microbatches=max_microbatches,
    )
    _MANAGERS[key] = (signature, manager)
    return manager
