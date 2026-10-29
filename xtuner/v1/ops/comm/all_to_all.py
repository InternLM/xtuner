import os

import torch
from torch.autograd.function import FunctionCtx
from torch.distributed._functional_collectives import all_to_all_single_autograd as _all_to_all_single_autograd
from torch.distributed.device_mesh import DeviceMesh


# Opt-in: use a tiled transpose for the Ulysses communication pack.
_A2A_TRITON_PACK = os.getenv("XTUNER_A2A_TRITON_PACK", "0") == "1"
# Keep sequence-major activations throughout the KDA q/k/v redistribution.
_A2A_TOKEN_MAJOR = os.getenv("XTUNER_A2A_TOKEN_MAJOR", "0") == "1"


@torch.library.custom_op("xtuner::transpose_pack_2d", mutates_args=(), device_types="cuda")
def _transpose_pack_2d(x: torch.Tensor) -> torch.Tensor:
    """Materialize a CUDA transpose with independent contiguous storage.

    Args:
        x (torch.Tensor): CUDA tensor [tokens, channels], with arbitrary strides.
            Supports float16, bfloat16 and float32.

    Returns:
        torch.Tensor: Contiguous [channels, tokens] tensor of the same dtype.
    """
    x = x.contiguous()
    s, d = x.shape
    y = x.new_empty((d, s))
    if x.numel():
        # Custom-op execution is opaque to Dynamo; fake registration stays Torch-only.
        from ._a2a_triton import _transpose_2d_kernel

        grid = ((s + 255) // 256, (d + 255) // 256)
        _transpose_2d_kernel[grid](x, y, s, d, BLOCK=256, num_warps=4)
    return y


@_transpose_pack_2d.register_fake
def _transpose_pack_2d_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[1], x.shape[0]))


def _transpose_pack_2d_backward(ctx: FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
    return _transpose_pack_2d(grad)


_transpose_pack_2d.register_autograd(_transpose_pack_2d_backward)


@torch.library.custom_op("xtuner::a2a_swap_groups", mutates_args=(), device_types="cuda")
def _swap_groups(x: torch.Tensor) -> torch.Tensor:
    """Swap the first two axes into independent contiguous CUDA storage.

    Args:
        x (torch.Tensor): Tensor of shape [tokens, ranks, channels], or the inverse
            layout [ranks, tokens, channels]. Arbitrary input strides are supported.

    Returns:
        torch.Tensor: Contiguous tensor with its first two axes exchanged, preserving dtype.
    """
    a, b, c = x.shape
    output = x.new_empty((b, a, c))
    if x.numel():
        from ._a2a_triton import _swap_groups_kernel

        _swap_groups_kernel[((x.numel() + 1023) // 1024,)](x, output, a, c, x.numel(), *x.stride(), BLOCK=1024)
    return output


@_swap_groups.register_fake
def _swap_groups_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[1], x.shape[0], x.shape[2]))


def _swap_groups_backward(ctx: FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
    return _swap_groups(grad)


_swap_groups.register_autograd(_swap_groups_backward)


def ulysses_all_to_all(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    mesh: DeviceMesh,
) -> torch.Tensor:
    """Performs a collective all-to-all operation on a tensor across a device
    mesh.

    This function redistributes data from the scatter dimension to the gather dimension
    across all processes in the device mesh.

    Args:
        input (torch.Tensor): Per-rank tensor with arbitrary axes. The packed KDA
            case is [1, local_tokens, channels], scattering channels (axis 2)
            and gathering contiguous sequence shards (axis 1).
        scatter_dim (int): The dimension along which to scatter the input tensor.
        gather_dim (int): The dimension along which to gather the output tensor.
        mesh (DeviceMesh): The device mesh defining the process group for communication.

    Returns:
        torch.Tensor: Independent contiguous tensor with the scatter axis divided
            by mesh size and the gather axis multiplied by mesh size. The packed
            KDA result is [1, global_tokens, channels / mesh_size], in source-rank order.

    Note:
        The scatter dimension must be evenly divisible by the group size.
    """
    world_size = mesh.size()
    if input.ndim == 3 and input.size(0) == 1 and scatter_dim == 2 and gather_dim == 1:
        tokens, channels = input.shape[1:]
        if channels % world_size:
            raise ValueError("The channel dimension must be divisible by the sequence-parallel mesh size.")
        if _A2A_TOKEN_MAJOR and input.is_cuda and input.dtype in (torch.bfloat16, torch.float32):
            # Pack [tokens, ranks, channels/ranks] as rank-major NCCL chunks.
            # Received source-rank chunks are already in global sequence order.
            packed = _swap_groups(input.reshape(tokens, world_size, channels // world_size))
            received = _all_to_all_single_autograd(
                packed,
                group=mesh.get_group(),
                input_split_sizes=[1] * world_size,
                output_split_sizes=[1] * world_size,
            )
            return received.reshape(1, tokens * world_size, channels // world_size)
        # Preserve the original KDA layout chain as the opt-out reference, including
        # its tiled transpose pack when XTUNER_A2A_TRITON_PACK is enabled.
        return ulysses_all_to_all(input.transpose(1, 2), 1, 2, mesh).transpose(1, 2).contiguous()

    split_size = input.size(scatter_dim) // world_size
    input_split_sizes = [split_size] * world_size
    output_split_sizes = input_split_sizes

    if (
        _A2A_TRITON_PACK
        and input.is_cuda
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and input.dim() == 3
        and input.size(0) == 1
        and scatter_dim == 1
        and input.stride() == (input.size(1) * input.size(2), 1, input.size(1))
    ):
        # input is [1, D, S], a transpose view of a contiguous [1, S, D]. Pack it
        # straight into the [D, 1, S] A2A layout with a tiled transpose instead of
        # letting Inductor lower the same permutation to an element-wise copy.
        # input[0].t() is the underlying contiguous [S, D] base as a pure view;
        # reshape() would instead copy the transposed logical order.
        input = _transpose_pack_2d(input.squeeze(0).t()).unsqueeze(1)
    else:
        input = input.contiguous()
        # Here, only movedim can be used, not transpose. Otherwise, it will lead to
        # a bug "ValueError: Tensors must be contiguous" in torch compile mode.
        # (a bug of torch compile)
        input = input.movedim(scatter_dim, 0)

    output = _all_to_all_single_autograd(
        input,
        group=mesh.get_group(),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )
    output = output.movedim(0, scatter_dim)

    output_list = torch.tensor_split(output, world_size, scatter_dim)
    output = torch.cat(output_list, dim=gather_dim).contiguous()
    return output


all_to_all_single_autograd = _all_to_all_single_autograd
