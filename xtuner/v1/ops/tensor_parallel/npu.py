from typing import Tuple

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh


class ColumnSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight0, weight1, weight2, bias, group):
        ctx.save_for_backward(input_)
        ctx.use_bias = bias is not None
        ctx.weight0 = weight0
        ctx.weight1 = weight1
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        # adapt for DTensor
        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2]).to_local()
        seqs = [weight0.shape[0], weight1.shape[0]]
        if weight2 is not None:
            seqs.append(weight2.shape[0])
            weight = torch.cat((weight0, weight1, weight2), dim=0)
            ctx.weight2 = weight2
        else:
            weight = torch.cat((weight0, weight1), dim=0)
        ctx.seqs = seqs

        weight = weight.to_local() if isinstance(weight, DTensor) else weight

        # npu_all_gather_base_mm currently do not support bias
        output, all_gather_output = torch_npu.npu_all_gather_base_mm(
            x,
            weight.t(),
            hcomm_info,
            world_size,
            bias=None,
            gather_index=0,
            gather_output=True,
        )

        if bias is not None:
            output = output + bias

        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])

        ctx.all_gather_output = None  # all_gather_output
        ctx.world_size = world_size
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors[0]  # [b, s//tp, h1] Shard(1)
        weight0 = ctx.weight0  # [h2//tp, h1], Shard(0)
        weight1 = ctx.weight1  # [h2//tp, h1], Shard(0)
        weight2 = None
        if hasattr(ctx, "weight2"):
            weight2 = ctx.weight2
            weight = torch.cat((weight0, weight1, weight2), dim=0)
        else:
            weight = torch.cat((weight0, weight1), dim=0)
        device_mesh = input_.device_mesh
        input_, weight, grad_output = input_.to_local(), weight, grad_output  # .to_local()

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )  # [b, s, h2//tp] Shard(2) -> [bs, h2//tp] Shard(1)

        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * ctx.world_size
        all_gather_output = torch.empty(
            dim_size,
            dtype=input_.dtype,
            device=torch.npu.current_device(),
            requires_grad=False,
        )
        all_gather_work = torch.distributed._all_gather_base(
            all_gather_output, input_.contiguous(), group=ctx.group, async_op=True
        )

        grad_input = grad_output_.matmul(weight)
        grad_input = grad_input.reshape(
            grad_output.shape[0], grad_output.shape[1], weight.shape[1]
        )  # [b, s, h2//tp]x[h2//tp, h1 -> [b, s, h1] Partial()

        sub_grad_input = torch.empty(
            list(input_.size()), dtype=input_.dtype, device=torch.npu.current_device()
        )  # [b, s//tp, h1] Shard(1)
        reduce_scatter_work = torch.distributed._reduce_scatter_base(
            sub_grad_input, grad_input, group=ctx.group, async_op=True
        )

        all_gather_work.wait()
        all_gather_output = all_gather_output.reshape(
            all_gather_output.shape[0] * all_gather_output.shape[1],
            all_gather_output.shape[2],
        )

        grad_weight = grad_output_.t().matmul(all_gather_output)  # [h2//tp, bs]x[bs, h1] -> [h2//tp, h1] Shard(0)
        grad_weight2 = None
        if len(ctx.seqs) == 3:
            grad_weight0, grad_weight1, grad_weight2 = torch.split(grad_weight, ctx.seqs, dim=0)
        else:
            grad_weight0, grad_weight1 = torch.split(grad_weight, ctx.seqs, dim=0)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = grad_output_.sum(dim=0) if grad_output_.is_contiguous() else grad_output_.t().sum(dim=1)
        else:
            grad_bias = None
        # [h2 // tp] shard(0)
        reduce_scatter_work.wait()
        sub_grad_input = DTensor.from_local(sub_grad_input, device_mesh=device_mesh, placements=(Shard(1),))
        if isinstance(weight0, DTensor):
            grad_weight0 = DTensor.from_local(grad_weight0, device_mesh=device_mesh, placements=(Shard(0),))
            grad_bias = DTensor.from_local(grad_bias, device_mesh=device_mesh, placements=(Shard(0),))
        if isinstance(weight1, DTensor):
            grad_weight1 = DTensor.from_local(grad_weight1, device_mesh=device_mesh, placements=(Shard(0),))
        if hasattr(ctx, "weight2") and isinstance(weight2, DTensor):
            grad_weight2 = DTensor.from_local(grad_weight2, device_mesh=device_mesh, placements=(Shard(0),))

        return sub_grad_input, grad_weight0, grad_weight1, grad_weight2, grad_bias, None


class RowSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_)
        ctx.use_bias = bias is not None
        ctx.weight = weight

        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2]).to_local()
        weight = weight.to_local() if isinstance(weight, DTensor) else weight

        # npu_mm_reduce_scatter_base currently do not support bias
        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=None
        )

        if bias is not None:
            output = output + bias

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(input_.shape[0], int(output.shape[0] / input_.shape[0]), output.shape[1])

        # output = DTensor.from_local(output,
        #                             device_mesh=device_mesh,
        #                             placements=(Shard(1),))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors[0]
        weight = ctx.weight
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size
        device_mesh = input_.device_mesh
        input_, grad_output = input_.to_local(), grad_output  # .to_local()

        grad_output_ = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])

        grad_input, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )

        grad_input = grad_input.view_as(input_)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        grad_weight = all_gather_grad_output.t().matmul(x)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = grad_output.sum(dim=0) if grad_output.is_contiguous() else grad_output.t().sum(dim=1)
        else:
            grad_bias = None

        grad_input = DTensor.from_local(grad_input, device_mesh=device_mesh, placements=(Shard(2),))
        if isinstance(weight, DTensor):
            grad_weight = DTensor.from_local(grad_weight, device_mesh=device_mesh, placements=(Shard(1),))
            grad_bias = DTensor.from_local(grad_bias, device_mesh=device_mesh, placements=(Shard(1),))

        return grad_input, grad_weight, grad_bias, None


class ColumnSeqParallelLinearWithFrozenWeight(ColumnSeqParallelLinear):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.input_shape = input_.shape
        ctx.use_bias = bias is not None
        ctx.weight = weight

        rank = torch.distributed.get_rank(group)

        hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        world_size = group.get_world_size()
        # npu_all_gather_base_mm currently do not support bias
        output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            x,
            weight.t(),
            hcomm_info,
            world_size,
            bias=None,
            gather_index=0,
            gather_output=False,
        )

        if bias is not None:
            output = output + bias

        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])
        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        weight = ctx.weight

        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size
        grad_output_ = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])

        sub_grad_input = torch_npu.npu_mm_reduce_scatter_base(grad_output_, weight, hcomm_info, world_size, bias=None)

        sub_grad_input = sub_grad_input.view(input_shape)

        return sub_grad_input, None, None, None


class RowSeqParallelLinearWithFrozenWeight(RowSeqParallelLinear):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.input_shape = input_.shape
        ctx.use_bias = bias is not None
        ctx.weight = weight

        rank = torch.distributed.get_rank(group)
        world_size = group.size()

        hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        # npu_mm_reduce_scatter_base currently do not support bias
        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=None
        )

        if bias is not None:
            output = output + bias

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        weight = ctx.weight
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size
        grad_output_ = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])

        grad_input, _ = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )

        grad_input = grad_input.view(input_shape)

        return grad_input, None, None, None


def get_tp_info(x):
    assert isinstance(x, DTensor), "current only support DTensor as input"
    tp_mesh = x.device_mesh
    tp_group = tp_mesh.get_group()
    tp_world_size = dist.get_world_size(tp_group)
    tp_local_rank = dist.get_rank(tp_group)
    return tp_group, tp_world_size, tp_local_rank


def attn_column_parallel_forward(
    hidden_states: torch.Tensor, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, tp_mesh: DeviceMesh
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = DTensor.from_local(hidden_states, tp_mesh, [Shard(1)])
    tp_group = tp_mesh.get_group()

    # bqkv = torch.cat((self.q_proj.bias.to_local(), self.k_proj.bias.to_local(), self.v_proj.bias.to_local()), dim=0)
    w_q = q_proj.weight
    w_k = k_proj.weight
    w_v = v_proj.weight

    assert isinstance(w_q, DTensor)
    assert isinstance(w_k, DTensor)
    assert isinstance(w_v, DTensor)

    w_q = w_q.to_local()
    w_k = w_k.to_local()
    w_v = w_v.to_local()

    h_q, h_k, h_v = (
        w_q.shape[0],
        w_k.shape[0],
        w_v.shape[0],
    )

    qkv_states = ColumnSeqParallelLinear.apply(hidden_states, w_q, w_k, w_v, None, tp_group)

    query_states, key_states, value_states = qkv_states.split([h_q, h_k, h_v], dim=2)

    return query_states, key_states, value_states


def attn_row_parallel_forward(attn_output: torch.Tensor, o_proj: nn.Linear, tp_mesh: DeviceMesh) -> torch.Tensor:
    attn_output = DTensor.from_local(attn_output, device_mesh=tp_mesh, placements=(Shard(2),))
    tp_group = tp_mesh.get_group()

    w_o = o_proj.weight
    assert isinstance(w_o, DTensor)
    w_o = w_o.to_local()
    output = RowSeqParallelLinear.apply(attn_output, w_o, None, tp_group)

    return output
