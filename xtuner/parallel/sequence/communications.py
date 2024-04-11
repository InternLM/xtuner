# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


def _all_to_all(
    input: Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input: Input tensor
        sp_group: Sequence parallel process group
        scatter_dim: Scatter dimension
        gather_dim: Gather dimension
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, sp_group: dist.ProcessGroup,
                scatter_dim: int, gather_dim: int):
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(sp_group)
        output = _all_to_all(input, ctx.world_size, sp_group, scatter_dim,
                             gather_dim)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple:
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.sp_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input: Tensor,
    sp_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input, sp_group, scatter_dim, gather_dim)


def split_for_sequence_parallel(input, dim: int, sp_group: dist.ProcessGroup):
    world_size = dist.get_world_size(sp_group)
    if world_size == 1:
        return input

    rank = dist.get_rank(sp_group)
    dim_size = input.size(dim)
    assert dim_size % world_size == 0, (
        f'The dimension to split ({dim_size}) is not a multiple of '
        f'world size ({world_size}), cannot split tensor evenly')

    tensor_list = torch.split(input, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def gather_for_sequence_parallel(input, dim: int, sp_group: dist.ProcessGroup):
    input = input.contiguous()
    world_size = dist.get_world_size(sp_group)
    dist.get_rank(sp_group)

    if world_size == 1:
        return input

    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    assert input.device.type == 'cuda'
    dist.all_gather(tensor_list, input, group=sp_group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input during forward.

    Scale and split the grad and keep only the corresponding chuck to the rank
    during backward.
    """

    @staticmethod
    def forward(ctx, input, dim, sp_group, grad_scale):
        ctx.dim = dim
        ctx.sp_group = sp_group
        ctx.grad_scale = grad_scale
        return gather_for_sequence_parallel(input, dim, sp_group)

    @staticmethod
    def backward(ctx, grad_output):
        """Why we need grad scale? Here is a simple case.

        -------- SP 1 -----------
        Suppose here is a toy model with only one linear module
        (in_features = 2, out_features = 1) and the input x has shape(2, 2).
        Y = [[y1], = [[w11x11 + w21x12], = [[x11, x12], dot [[w11],
             [y2]]    [w11x21 + w21x22]]    [x21, x22]]      [w21]]
        z = sum(Y) = y1 + y2
        Here is the partial derivative of z with respect to w11:
        ∂z / ∂w11 = ∂z / ∂y1 * ∂y1 / ∂w11 + ∂z / ∂y2 * ∂y2 / ∂w11
                  = 1 * x11 + 1 * x21 = x11 + x21

        -------- SP 2 -----------
        When sequence parallel world size is set to 2, we will split the input x
        and scatter them to the two rank in the same sequence parallel group.
        ```Step 1
        Y_rank0 = [[y1]] = [[w11x11 + w21x12]] = [[x11, x12]] dot [[w11, w21]]^T
        Y_rank1 = [[y2]] = [[w11x21 + w21x22]] = [[x21, x22]] dot [[w11, w21]]^T
        ```

        Then, we have to gather them:
        ```Step 2
        Y_rank0 = [[y1],
                   detach([y2])]
        Y_rank1 = [detach([y1]),
                   [y2]]
        ```
        Note that y2 in Y_rank0 does not have grad, neither does y1 in Y_rank1.

        Similarly, we calculate the loss in each rank:
        ```Step 3
        z_rank0 = sum(Y_rank0) = y1 + detach(y2)
        z_rank1 = sum(Y_rank1) = detach(y1) + y2
        ```
        So the partial derivative of loss_rank0 with respect to w11:
        ```∂z / ∂w11 = ∂z / ∂y1 * ∂y1 / ∂w11 = x11```
        The same for rank1:
        ```∂z / ∂w11 = ∂z / ∂y2 * ∂y2 / ∂w11 = x21```

        Finally, we need to all_reduce them:
        ```Step 4
        In both rank:
        ∂z / ∂w11 = (x11 + x21) / 2
        ```

        In SP2, the gradient of each param is only half of that in SP1.
        So we should scale up the grad during the backward process in Step 2.
        """  # noqa: E501
        if ctx.grad_scale == 'up':
            grad_output = grad_output * dist.get_world_size(ctx.sp_group)
        elif ctx.grad_scale == 'down':
            grad_output = grad_output / dist.get_world_size(ctx.sp_group)

        return (split_for_sequence_parallel(grad_output, ctx.dim,
                                            ctx.sp_group), None, None, None)


class _SplitForwardGatherBackward(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank during
    forward.

    Scale and gather the grad during backward.
    """

    @staticmethod
    def forward(ctx, input, dim, sp_group, grad_scale):
        ctx.dim = dim
        ctx.sp_group = sp_group
        ctx.grad_scale = grad_scale
        return split_for_sequence_parallel(input, dim, sp_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == 'up':
            grad_output = grad_output * dist.get_world_size(ctx.sp_group)
        elif ctx.grad_scale == 'down':
            grad_output = grad_output / dist.get_world_size(ctx.sp_group)
        return (gather_for_sequence_parallel(grad_output, ctx.dim,
                                             ctx.sp_group), None, None, None)


def split_forward_gather_backward(input, dim, sp_group, grad_scale=None):
    return _SplitForwardGatherBackward.apply(input, dim, sp_group, grad_scale)


def gather_forward_split_backward(input, dim, sp_group, grad_scale=None):
    return _GatherForwardSplitBackward.apply(input, dim, sp_group, grad_scale)
