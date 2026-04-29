import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

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
    n_routed_experts: int | None = None
    device: torch.device | str | int | None = None

    def build(
        self,
        *,
        n_routed_experts: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> "AuxLoss":
        """Build a layer-wise MoE auxiliary loss context."""
        resolved_n_routed_experts = n_routed_experts if n_routed_experts is not None else self.n_routed_experts
        assert resolved_n_routed_experts is not None, "n_routed_experts must be provided either in config or build()."

        resolved_device = device if device is not None else self.device
        if resolved_device is None:
            resolved_device = DEVICE_MODULE.current_device()

        return AuxLoss(
            n_routed_experts=resolved_n_routed_experts,
            device=resolved_device,
        )


class AuxLossKwargs(BaseModel):
    """Keyword arguments for layer-wise split MoE auxiliary loss context."""

    model_config = ConfigDict(title="layer moe loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)
    device: torch.device | str | int | None


class AuxLossContext(nn.Module):
    """Layer-wise split MoE auxiliary loss dispatcher.

    Runtime accumulators primarily live on auxiliary loss contexts and are released when finalize completes.
    """

    def __init__(self, loss_cfg: AuxLossConfig, loss_kwargs: AuxLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        n_routed_experts = self.loss_cfg.n_routed_experts
        assert n_routed_experts is not None, "n_routed_experts must be resolved before creating AuxLossContext."
        self.n_routed_experts: int = n_routed_experts
        self._local_load_logits_list: list[torch.Tensor] = []

    def accumulate(
        self,
        *,
        router_weights: torch.Tensor,
        router_logits: torch.Tensor,
        num_experts_per_tok: int,
        mask: torch.Tensor,
        balancing_ctx: list[BalancingLossContext] | BalancingLossContext | None = None,
        z_ctx: list[ZLossContext] | ZLossContext | None = None,
        dim: int = 1,
    ) -> None:
        """Accumulate routing statistics for one layer."""
        selected_router_weights, _ = _select_nonpad(router_weights, mask, dim=dim)
        selected_router_logits, _ = _select_nonpad(router_logits, mask, dim=dim)

        _, selected_experts = torch.topk(selected_router_weights, num_experts_per_tok, dim=-1)
        tokens_per_expert_per_layer = torch.histc(
            selected_experts.view(-1),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        ).to(torch.long)

        if balancing_ctx is not None:
            if isinstance(balancing_ctx, list):
                for balancing_ctx_item in balancing_ctx:
                    balancing_ctx_item.accumulate(
                        router_weights=selected_router_weights,
                        tokens_per_expert=tokens_per_expert_per_layer,
                    )
            else:
                balancing_ctx.accumulate(
                    router_weights=selected_router_weights,
                    tokens_per_expert=tokens_per_expert_per_layer,
                )

        self._local_load_logits_list.append(tokens_per_expert_per_layer)

        if z_ctx is not None:
            if isinstance(z_ctx, list):
                for z_ctx_item in z_ctx:
                    z_ctx_item.accumulate(
                        router_logits=selected_router_logits,
                    )
            else:
                z_ctx.accumulate(
                    router_logits=selected_router_logits,
                )

    def _cal_tokens_per_expert(self) -> torch.Tensor:
        """Get tokens-per-expert tensor for logging/bias update."""
        local_load_logits = self._local_load_logits_list
        self._local_load_logits_list = []

        if not local_load_logits:
            raise RuntimeError(
                "No MoE routing statistics were accumulated before finalize(). "
                "This usually means the model has no MoE layers or finalize() was called "
                "without a preceding accumulate()."
            )
        active_load_logits = torch.stack(local_load_logits, dim=0)
        if dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            return all_reduce(active_load_logits, "sum", group)
        return active_load_logits

    def finalize(
        self,
        *,
        balancing_ctx: list[BalancingLossContext] | BalancingLossContext | None,
        z_ctx: list[ZLossContext] | ZLossContext | None,
        num_experts_per_tok: int,
        non_pad_token: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor] | None:
        """Finalize split auxiliary losses and expert counts from runtime
        state."""
        balancing_loss = None
        if balancing_ctx is not None:
            if isinstance(balancing_ctx, list):
                balancing_loss = torch.sum(
                    torch.stack(
                        [
                            ctx.finalize(
                                n_routed_experts=self.n_routed_experts,
                                num_experts_per_tok=num_experts_per_tok,
                                non_pad_token=non_pad_token,
                            )
                            for ctx in balancing_ctx
                        ]
                    ),
                    dim=0,
                )
            else:
                balancing_loss = balancing_ctx.finalize(
                    n_routed_experts=self.n_routed_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    non_pad_token=non_pad_token,
                )

        z_loss = None
        if z_ctx is not None:
            if isinstance(z_ctx, list):
                z_loss = torch.sum(torch.stack([ctx.finalize() for ctx in z_ctx]), dim=0)
            else:
                z_loss = z_ctx.finalize()

        tokens_per_expert_global = self._cal_tokens_per_expert()
        return balancing_loss, z_loss, tokens_per_expert_global


class AuxLoss(AuxLossContext):
    """Unified MoE auxiliary loss wrapper."""

    def __init__(self, n_routed_experts: int, device: torch.device | str | int):
        cfg = AuxLossConfig(n_routed_experts=n_routed_experts, device=device)
        kwargs = AuxLossKwargs(device=device)
        super().__init__(cfg, kwargs)
