"""Optional UltraEP runtime integration for Xtuner MoE layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


if TYPE_CHECKING:
    from ultra_ep import EventHandle, Manager

    from xtuner.v1.model.moe.moe import MoEConfig


class UltraEPGroupedLinear(Protocol):
    """The small grouped-linear surface owned by one UltraEP layer binding."""

    weight: torch.Tensor

    def configure_ultra_ep_buffers(self, replica_weight: torch.Tensor, replica_grad: torch.Tensor) -> None: ...


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

        if dist.get_world_size() != group.size():
            raise NotImplementedError(
                "Xtuner UltraEP currently requires DP=1 (the EP group must cover the world); "
                "FSDP gradient-reduction ordering for DP>1 is not implemented yet."
            )

        self.group = group
        self.num_layers = num_layers
        self.num_local_master_experts = num_local_master_experts
        self.num_local_redundant_experts = num_local_redundant_experts
        self.expert_fc1_numel = expert_fc1_numel
        self.expert_fc2_numel = expert_fc2_numel
        self.max_microbatches = max_microbatches
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
            grad_dtype=torch.float32,
        )
        # UltraEP's native grad-reduce kernel is FP32-only, while Xtuner FSDP
        # keeps BF16 gradients beside BF16 expert parameters.  One staging
        # pair is sufficient: backward joins each layer's async reduction
        # before the preceding layer can start its own reduction.  Reusing the
        # pair avoids allocating FP32 master-grad copies for every layer.
        device = torch.device("cuda", torch.cuda.current_device())
        self.master_fc1_grad_staging = torch.empty(
            num_local_master_experts,
            expert_fc1_numel,
            dtype=torch.float32,
            device=device,
        )
        self.master_fc2_grad_staging = torch.empty(
            num_local_master_experts,
            expert_fc2_numel,
            dtype=torch.float32,
            device=device,
        )
        self._staging_owner: int | None = None
        self._master_weight_ptr_hosts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

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
        """Cast Xtuner's local BF16 master grads into shared FP32 staging."""
        if self._staging_owner is not None:
            raise RuntimeError(
                "UltraEP FP32 master-grad staging is still owned by virtual layer "
                f"{self._staging_owner}; attempted to start {virtual_layer_id}"
            )
        local_fc1 = self._local(fc1_grad).view(self.num_local_master_experts, -1)
        local_fc2 = self._local(fc2_grad).view(self.num_local_master_experts, -1)
        if (
            local_fc1.numel() != self.master_fc1_grad_staging.numel()
            or local_fc2.numel() != self.master_fc2_grad_staging.numel()
        ):
            raise ValueError("Master expert gradient shapes do not match UltraEP FP32 staging")
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
    ) -> None:
        """Cast the reduced FP32 staging tensors back to Xtuner FSDP grads."""
        if self._staging_owner != virtual_layer_id:
            raise RuntimeError(f"UltraEP staging owner is {self._staging_owner}, not {virtual_layer_id}")
        local_fc1 = self._local(fc1_grad).view(self.num_local_master_experts, -1)
        local_fc2 = self._local(fc2_grad).view(self.num_local_master_experts, -1)
        local_fc1.copy_(self.master_fc1_grad_staging)
        local_fc2.copy_(self.master_fc2_grad_staging)
        self._staging_owner = None

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
        if fc1_grad.dtype != torch.float32 or fc2_grad.dtype != torch.float32:
            raise TypeError(f"UltraEP requires FP32 master grads, got {fc1_grad.dtype} and {fc2_grad.dtype}")
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

    def refresh_master_weight_pointers(
        self,
        *,
        layer_id: int,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> None:
        """Refresh only FSDP-movable master-weight addresses in cached
        pools."""
        hosts = self._master_weight_ptr_hosts.get(layer_id)
        if hosts is None:
            raise RuntimeError(f"UltraEP master pointer pool for layer {layer_id} is not registered")
        fc1_weight = self._local(fc1_weight).view(self.num_local_master_experts, -1)
        fc2_weight = self._local(fc2_weight).view(self.num_local_master_experts, -1)
        fc1_host, fc2_host = hosts
        for expert_idx in range(self.num_local_master_experts):
            fc1_host[expert_idx] = fc1_weight[expert_idx].data_ptr()
            fc2_host[expert_idx] = fc2_weight[expert_idx].data_ptr()

        fc1_device = self.runtime.local_master_fc1_weight_ptr_pool[layer_id]
        fc2_device = self.runtime.local_master_fc2_weight_ptr_pool[layer_id]
        if fc1_device is None or fc2_device is None:
            raise RuntimeError(f"UltraEP device pointer pool for layer {layer_id} is unavailable")
        fc1_device.copy_(fc1_host, non_blocking=True)
        fc2_device.copy_(fc2_host, non_blocking=True)

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

        self.group = group
        self.num_model_layers = num_model_layers
        self.num_logical_experts = num_logical_experts
        self.hidden_size = hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_redundant_experts_per_rank = num_redundant_experts_per_rank
        self.max_microbatches = max_microbatches
        self._manager: UltraEPManager | None = None

    @classmethod
    def from_xtuner_config(
        cls,
        *,
        group: dist.ProcessGroup,
        config: MoEConfig,
    ) -> UltraEPManagerProvider:
        """Build a provider from Xtuner's model config for the single-
        microbatch path."""
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
            max_microbatches=1,
        )

    @property
    def num_dispatch_experts(self) -> int:
        """Global physical-expert count expected by the dispatcher."""
        return self.num_logical_experts + self.group.size() * self.num_redundant_experts_per_rank

    def get_manager(self) -> UltraEPManager:
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
        self._ensure_manager().update_placement_sparse(virtual_layer_id, logical_topk_ids)

    def reroute(self, logical_topk_ids: torch.Tensor, virtual_layer_id: int) -> torch.Tensor:
        """Return a dispatcher-only copy rewritten into physical expert IDs."""
        physical_topk_ids = logical_topk_ids.clone()
        self._ensure_manager().reroute_sparse(virtual_layer_id, physical_topk_ids)
        return physical_topk_ids

    def sync_weights(self, virtual_layer_id: int, *, async_finish: bool):
        manager = self._ensure_manager()
        manager.refresh_master_weight_pointers(
            layer_id=self.layer_id,
            fc1_weight=self.fused_w1w3.weight,
            fc2_weight=self.fused_w2.weight,
        )
        return manager.weight_sync(virtual_layer_id, async_finish=async_finish)

    def start_grad_reduce(self, virtual_layer_id: int) -> None:
        if virtual_layer_id in self._grad_reduce_events:
            raise RuntimeError(f"UltraEP virtual layer slot {virtual_layer_id} is still in use")
        fc1_grad = self.fused_w1w3.weight.grad
        fc2_grad = self.fused_w2.weight.grad
        if fc1_grad is None or fc2_grad is None:
            raise RuntimeError(
                f"UltraEP master gradients are unavailable at layer {self.layer_id}; "
                "the FSDP/autograd hook ordering is incompatible with replica grad-reduce"
            )
        manager = self._ensure_manager()
        manager.stage_master_gradients(
            virtual_layer_id=virtual_layer_id,
            fc1_grad=fc1_grad,
            fc2_grad=fc2_grad,
        )
        event = manager.grad_reduce(virtual_layer_id, async_finish=True)
        self._grad_reduce_events[virtual_layer_id] = (event, fc1_grad, fc2_grad)

    def finish_grad_reduce(self, virtual_layer_id: int) -> None:
        state = self._grad_reduce_events.pop(virtual_layer_id, None)
        if state is None:
            raise RuntimeError(f"UltraEP grad-reduce event for virtual layer {virtual_layer_id} was not started")
        event, fc1_grad, fc2_grad = state
        event.current_stream_wait()  # type: ignore[attr-defined]
        self._ensure_manager().restore_master_gradients(
            virtual_layer_id=virtual_layer_id,
            fc1_grad=fc1_grad,
            fc2_grad=fc2_grad,
        )

    def _ensure_manager(self) -> UltraEPManager:
        manager = self.manager_provider.get_manager()
        if not self._buffers_configured:
            redundant = manager.num_local_redundant_experts
            self.fused_w1w3.configure_ultra_ep_buffers(
                manager.local_replica_fc1_weight_buffer.view(
                    redundant,
                    2 * self.expert_intermediate_size,
                    self.hidden_size,
                ),
                manager.local_replica_fc1_grad_buffer.view(
                    redundant,
                    2 * self.expert_intermediate_size,
                    self.hidden_size,
                ),
            )
            self.fused_w2.configure_ultra_ep_buffers(
                manager.local_replica_fc2_weight_buffer.view(
                    redundant,
                    self.hidden_size,
                    self.expert_intermediate_size,
                ),
                manager.local_replica_fc2_grad_buffer.view(
                    redundant,
                    self.hidden_size,
                    self.expert_intermediate_size,
                ),
            )
            self._buffers_configured = True

        if not self._master_pointers_registered:
            manager.register_master_pointers(
                layer_id=self.layer_id,
                fc1_weight=self.fused_w1w3.weight,
                fc2_weight=self.fused_w2.weight,
                fc1_grad=manager.master_fc1_grad_staging,
                fc2_grad=manager.master_fc2_grad_staging,
            )
            self._master_pointers_registered = True
        return manager


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
