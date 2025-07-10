import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.config import MoELossConfig


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        tensor = all_reduce(tensor, op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)


def all_reduce_autograd(tensor, op, group):
    return _AllReduce.apply(op, group, tensor)


class BalancingLoss(nn.Module):
    def __init__(self, moe_loss_cfg: MoELossConfig) -> None:
        super().__init__()
        self.loss_type = moe_loss_cfg.balancing_loss_type
        self.loss_weight = moe_loss_cfg.balancing_loss_alpha
        self.global_average = moe_loss_cfg.balancing_loss_global_average

    def forward(self, router_logits, n_routed_experts, num_experts_per_tok):
        if self.loss_weight == 0:
            return 0.0

        num_layers = router_logits.shape[0]
        router_logits = router_logits.float()  # (nlayers, seq, ne)
        if self.loss_type == "softmax":
            routing_weights = F.softmax(router_logits, dim=-1)
        elif self.loss_type == "sigmoid":
            routing_weights = router_logits / torch.sum(router_logits, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
        selected_experts_flat = selected_experts.view(num_layers, -1)
        offset = torch.arange(num_layers, device=router_logits.device).unsqueeze(1) * n_routed_experts
        selected_experts_offset = selected_experts_flat + offset
        tokens_per_expert_flat = torch.histc(
            selected_experts_offset.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)  # (nlayers, ne)

        tokens_per_expert_global = tokens_per_expert.to(routing_weights.dtype)  # (nlayers, ne)
        if self.global_average and dist.is_initialized():
            tokens_per_expert_global = all_reduce(tokens_per_expert_global, "sum", dist.group.WORLD)  # (nlayers, ne)
            tokens_global = tokens_per_expert_global.sum(-1)  # (nlayers, )
            seqlen_global = tokens_global // num_experts_per_tok
            routing_weights_sum_global = all_reduce_autograd(
                routing_weights.sum(dim=1), "sum", dist.group.WORLD
            )  # (nlayers, )
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
        else:
            scale_global = n_routed_experts / (router_logits.shape[1] * num_experts_per_tok)
            routing_weights_mean_global = routing_weights.mean(dim=1)
        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
        loss = loss.sum()

        return loss * self.loss_weight


def z_loss(router_logits: torch.Tensor, global_average: bool = False):
    router_logits = router_logits.float()  # (nlayers, seq, ne)
    z_loss = torch.logsumexp(router_logits, dim=-1).square().mean(dim=-1).sum()
    if global_average and dist.is_initialized():
        unmasked_num = router_logits.shape[1]
        unmasked_num_rank = torch.tensor(unmasked_num, device=router_logits.device, dtype=torch.int64)
        unmasked_num_global = all_reduce(unmasked_num_rank, "sum", dist.group.WORLD)  # type: ignore
        world_size = dist.get_world_size()
        z_loss = z_loss * unmasked_num * world_size / unmasked_num_global
    return z_loss


class ZLoss(nn.Module):
    def __init__(self, moe_loss_cfg: MoELossConfig) -> None:
        super().__init__()
        self.loss_weight = moe_loss_cfg.z_loss_alpha
        self.global_average = moe_loss_cfg.z_loss_global_average

    def forward(self, router_logits):
        if self.loss_weight == 0:
            return 0.0
        loss = z_loss(router_logits, self.global_average)
        return loss * self.loss_weight
