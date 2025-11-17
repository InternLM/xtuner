from typing import cast

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_reduce


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
    def __init__(
        self,
        balancing_loss_alpha: float,
        balancing_loss_global_average: bool,
    ) -> None:
        super().__init__()
        self.loss_weight = balancing_loss_alpha
        self.global_average = balancing_loss_global_average

    def forward(
        self,
        router_weights: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_n_groups: int,
    ):
        if self.loss_weight == 0:
            return torch.tensor(0.0, device=router_weights.device, dtype=torch.float32)

        router_weights = router_weights.float()  # (nlayers, seq, ne)
        tokens_per_expert = self._get_tokens_per_experts(
            router_weights,
            n_routed_experts,
            num_experts_per_tok,
            router_n_groups,
        )  # (nlayers, ne)

        tokens_per_expert_global = tokens_per_expert.to(router_weights.dtype)  # (nlayers, ne)
        if self.global_average and dist.is_initialized():
            tokens_per_expert_global = all_reduce(  # (nlayers, ne)
                tokens_per_expert_global, "sum", cast(ProcessGroup, dist.group.WORLD)
            )
            tokens_global = tokens_per_expert_global.sum(-1)  # (nlayers, )
            seqlen_global = tokens_global // num_experts_per_tok
            routing_weights_sum_global = all_reduce_autograd(
                router_weights.sum(dim=1), "sum", dist.group.WORLD
            )  # (nlayers, )
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
        else:
            scale_global = n_routed_experts / (router_weights.shape[1] * num_experts_per_tok)
            routing_weights_mean_global = router_weights.mean(dim=1)
        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
        loss = loss.sum()
        # from xtuner.v1.profiler.prober import ProberList
        # ProberList.record_tensor(routing_weights_mean_global, "[balancing_loss][after]routing_weights_mean_global")
        # ProberList.record_tensor(tokens_per_expert_global, "[balancing_loss][after]tokens_per_expert_global")
        # ProberList.record_tensor(scale_global, "[balancing_loss][after]scale_global")
        return loss * self.loss_weight

    def _get_tokens_per_experts(
        self,
        router_weights: torch.Tensor,  # (nlayers, seq, ne)
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_groups: int,
    ):
        num_layers, seq, n_routed_experts = router_weights.shape
        group_size = max(1, n_routed_experts // n_groups)

        scores_for_choice = router_weights.view(num_layers, seq, n_groups, group_size)
        _, group_local_max_idx = torch.topk(
            scores_for_choice, k=num_experts_per_tok // n_groups, dim=3
        )  # nlayers, seq, n_groups, top_k_per_group
        group_offsets = torch.arange(num_layers * n_groups, device=router_weights.device) * group_size
        group_offsets = group_offsets.view(num_layers, 1, n_groups, 1)

        topk_ids = (group_local_max_idx + group_offsets).to(torch.long)  # [seq, n_groups, top_k_per_group]
        tokens_per_expert_flat = torch.histc(
            topk_ids.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)
        return tokens_per_expert


def z_loss(router_logits: torch.Tensor, global_average: bool = False):
    router_logits = router_logits.float()  # (nlayers, seq, ne)
    num_seq = max(1, router_logits.shape[1])
    logsum_square = z_loss = torch.logsumexp(router_logits, dim=-1).square()
    z_loss = (logsum_square.sum(dim=-1) / num_seq).sum()

    if global_average and dist.is_initialized():
        unmasked_num = router_logits.shape[1]
        unmasked_num_rank = torch.tensor(unmasked_num, device=router_logits.device, dtype=torch.int64)
        unmasked_num_global = all_reduce(unmasked_num_rank, "sum", dist.group.WORLD)  # type: ignore
        world_size = dist.get_world_size()
        z_loss = z_loss * unmasked_num * world_size / unmasked_num_global

    return z_loss


class ZLoss(nn.Module):
    def __init__(
        self,
        z_loss_alpha: float,
        z_loss_global_average: bool,
    ) -> None:
        super().__init__()
        self.loss_weight = z_loss_alpha
        self.global_average = z_loss_global_average

    def forward(self, router_logits):
        if self.loss_weight == 0:
            return torch.tensor(0.0, device=router_logits.device, dtype=torch.float32)
        loss = z_loss(router_logits, self.global_average)
        return loss * self.loss_weight
