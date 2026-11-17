import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _route_weight_rows_backward_kernel(
    grad_weighted,
    expert_output,
    route_weights,
    grad_expert,
    grad_route,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    # Match grouped-gemm BF16 unpermute backward: the FP32 router weight is
    # rounded before multiplication, and each route-gradient product is
    # rounded before its FP32 reduction.
    route_weight = tl.load(route_weights + row).to(tl.bfloat16).to(tl.float32)
    route_grad = 0.0
    for start in tl.static_range(0, hidden_size, block_size):
        offsets = start + tl.arange(0, block_size)
        mask = offsets < hidden_size
        grad = tl.load(grad_weighted + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        output = tl.load(expert_output + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            grad_expert + row * hidden_size + offsets,
            (grad * route_weight).to(tl.bfloat16),
            mask=mask,
        )
        route_grad += tl.sum((grad * output).to(tl.bfloat16).to(tl.float32), axis=0)
    tl.store(grad_route + row, route_grad)


@torch.library.custom_op("moe::route_weight_rows_backward", mutates_args=())
def route_weight_rows_backward(
    grad_weighted: Tensor,
    expert_output: Tensor,
    route_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Differentiate fused BF16 row scaling without a full FP32 activation."""
    assert grad_weighted.dtype is torch.bfloat16 and grad_weighted.is_contiguous()
    assert expert_output.dtype is torch.bfloat16 and expert_output.is_contiguous()
    assert route_weights.dtype is torch.float32 and route_weights.is_contiguous()
    assert grad_weighted.shape == expert_output.shape
    assert route_weights.shape == grad_weighted.shape[:1]

    grad_expert = torch.empty_like(grad_weighted)
    grad_route = torch.empty_like(route_weights)
    _route_weight_rows_backward_kernel[(grad_weighted.shape[0],)](
        grad_weighted,
        expert_output,
        route_weights,
        grad_expert,
        grad_route,
        hidden_size=grad_weighted.shape[1],
        block_size=256,
        num_warps=4,
    )
    return grad_expert, grad_route


@route_weight_rows_backward.register_fake
def _(grad_weighted: Tensor, expert_output: Tensor, route_weights: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(grad_weighted), torch.empty_like(route_weights)


__all__ = ["route_weight_rows_backward"]
