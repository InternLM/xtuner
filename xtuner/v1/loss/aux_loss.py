from typing import TypedDict, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.loss.base_loss_ctx import BaseLossContext
from xtuner.v1.loss.moe_loss import BalancingLossContext, ZLossContext
from xtuner.v1.utils import get_torch_device_module


DEVICE_MODULE = get_torch_device_module()


def _select_nonpad(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Select non-padding positions from tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        mask (torch.Tensor): Attention mask.
        dim (int): Select dimension.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Selected tensor and selected indices.
    """
    indices = torch.nonzero(mask, as_tuple=True)[1]
    selected = torch.index_select(tensor, dim, indices).contiguous().float()
    return selected, indices


class AuxLossConfig(BaseModel):
    """Configuration for layer-wise split MoE auxiliary loss."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    num_layers: int | None = None
    n_routed_experts: int | None = None
    device: torch.device | str | int | None = None

    def build(
        self,
        *,
        num_layers: int | None = None,
        n_routed_experts: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> "AuxLoss":
        """Build a layer-wise MoE auxiliary loss context."""
        resolved_num_layers = num_layers if num_layers is not None else self.num_layers
        resolved_n_routed_experts = n_routed_experts if n_routed_experts is not None else self.n_routed_experts
        assert resolved_num_layers is not None, "num_layers must be provided either in config or build()."
        assert resolved_n_routed_experts is not None, "n_routed_experts must be provided either in config or build()."

        resolved_device = device if device is not None else self.device
        if resolved_device is None:
            resolved_device = DEVICE_MODULE.current_device()

        return AuxLoss(
            num_layers=resolved_num_layers,
            n_routed_experts=resolved_n_routed_experts,
            device=resolved_device,
        )


class AuxLossKwargs(BaseModel):
    """Keyword arguments for layer-wise split MoE auxiliary loss context."""

    model_config = ConfigDict(title="layer moe loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)
    device: torch.device | str | int | None


class AuxLossRuntimeState(TypedDict):
    """Runtime accumulators for split MoE auxiliary loss."""

    local_load_logits: torch.Tensor
    local_load: torch.Tensor | None
    routing_weights_sum_list: list[torch.Tensor]
    z_loss_logsum: torch.Tensor | None
    z_loss_token_count: torch.Tensor | None
    active_layers: int
    balancing_ctx: BaseLossContext | None
    z_ctx: BaseLossContext | None


class AuxLossContext(nn.Module):
    """Layer-wise split MoE auxiliary loss dispatcher.

    This context no longer owns per-forward mutable states. Runtime accumulators are
    provided by caller (`moe.py`) through `AuxLossRuntimeState`.
    """

    def __init__(self, loss_cfg: AuxLossConfig, loss_kwargs: AuxLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        num_layers = self.loss_cfg.num_layers
        n_routed_experts = self.loss_cfg.n_routed_experts
        assert num_layers is not None, "num_layers must be resolved before creating AuxLossContext."
        assert n_routed_experts is not None, "n_routed_experts must be resolved before creating AuxLossContext."
        self.num_layers: int = num_layers
        self.n_routed_experts: int = n_routed_experts

    def accumulate(
        self,
        *,
        runtime_state: AuxLossRuntimeState,
        layer_idx: int,
        router_weights: torch.Tensor,
        router_logits: torch.Tensor,
        num_experts_per_tok: int,
        mask: torch.Tensor,
        balancing_ctx: BaseLossContext | None = None,
        z_ctx: BaseLossContext | None = None,
        dim: int = 1,
    ) -> None:
        """Accumulate routing statistics for one layer."""
        selected_router_weights, _ = _select_nonpad(router_weights, mask, dim=dim)
        selected_router_logits, _ = _select_nonpad(router_logits, mask, dim=dim)

        local_load_logits = runtime_state["local_load_logits"]
        _, selected_experts = torch.topk(selected_router_logits, num_experts_per_tok, dim=-1)
        tokens_per_expert_logits = torch.histc(
            selected_experts.view(-1),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        ).to(torch.long)
        local_load_logits[layer_idx] = tokens_per_expert_logits

        if balancing_ctx is not None:
            balancing_loss_ctx = cast("BalancingLossContext", balancing_ctx)
            local_load = runtime_state["local_load"]
            assert local_load is not None
            balancing_loss_ctx.update_split_aux(
                layer_idx=layer_idx,
                router_weights=selected_router_weights,
                num_experts_per_tok=num_experts_per_tok,
                n_routed_experts=self.n_routed_experts,
                local_load=local_load,
                routing_weights_sum_list=runtime_state["routing_weights_sum_list"],
            )
            runtime_state["balancing_ctx"] = balancing_ctx

        if z_ctx is not None:
            z_loss_ctx = cast("ZLossContext", z_ctx)
            z_loss_logsum = runtime_state["z_loss_logsum"]
            z_loss_token_count = runtime_state["z_loss_token_count"]
            assert z_loss_logsum is not None
            assert z_loss_token_count is not None
            z_loss_ctx.update_split_aux(
                layer_idx=layer_idx,
                router_logits=selected_router_logits,
                z_loss_logsum=z_loss_logsum,
                z_loss_token_count=z_loss_token_count,
            )
            runtime_state["z_ctx"] = z_ctx

        runtime_state["active_layers"] = max(runtime_state["active_layers"], layer_idx + 1)

    def _cal_tokens_per_expert(self, runtime_state: AuxLossRuntimeState) -> torch.Tensor:
        """Get tokens-per-expert tensor for logging/bias update."""
        local_load_logits = runtime_state["local_load_logits"]
        active_layers = runtime_state["active_layers"]
        if active_layers == 0:
            return torch.zeros(0, self.n_routed_experts, dtype=torch.int64, device=self.loss_kwargs.device)
        active_load_logits = local_load_logits[:active_layers]
        if dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            return all_reduce(active_load_logits, "sum", group)
        return active_load_logits

    def finalize(
        self,
        *,
        runtime_state: AuxLossRuntimeState,
        num_experts_per_tok: int,
        non_pad_token: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor] | None:
        """Finalize split auxiliary losses and expert counts from runtime
        state."""
        balancing_loss = None
        balancing_ctx = runtime_state["balancing_ctx"]
        if balancing_ctx is not None:
            balancing_loss_ctx = cast("BalancingLossContext", balancing_ctx)
            local_load = runtime_state["local_load"]
            assert local_load is not None
            balancing_loss = balancing_loss_ctx.finalize_split_aux_loss(
                local_load=local_load,
                routing_weights_sum_list=runtime_state["routing_weights_sum_list"],
                active_layers=runtime_state["active_layers"],
                n_routed_experts=self.n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                non_pad_token=non_pad_token,
            )

        z_loss = None
        z_ctx = runtime_state["z_ctx"]
        if z_ctx is not None:
            z_loss_ctx = cast("ZLossContext", z_ctx)
            z_loss_logsum = runtime_state["z_loss_logsum"]
            z_loss_token_count = runtime_state["z_loss_token_count"]
            assert z_loss_logsum is not None
            assert z_loss_token_count is not None
            z_loss = z_loss_ctx.finalize_split_aux_loss(
                z_loss_logsum=z_loss_logsum,
                z_loss_token_count=z_loss_token_count,
                active_layers=runtime_state["active_layers"],
            )

        tokens_per_expert_global = self._cal_tokens_per_expert(runtime_state)
        return balancing_loss, z_loss, tokens_per_expert_global


class AuxLoss(AuxLossContext):
    """Unified layer-wise MoE auxiliary loss wrapper."""

    def __init__(self, num_layers: int, n_routed_experts: int, device: torch.device | str | int):
        cfg = AuxLossConfig(num_layers=num_layers, n_routed_experts=n_routed_experts, device=device)
        kwargs = AuxLossKwargs(device=device)
        super().__init__(cfg, kwargs)
