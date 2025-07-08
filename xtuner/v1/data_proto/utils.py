# Copyright (c) OpenMMLab. All rights reserved.
from typing import TypeVar, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


T = TypeVar("T", bound=torch.Tensor)

# TODO: (yehaochen) Missing typehint here


def pad_to_multiple_of(sequence, padding_value, multiple_of, dim=-1):
    length = sequence.shape[dim]
    if length % multiple_of == 0:
        return sequence

    pad_num = multiple_of - (length % multiple_of)
    pad_shape = (
        (*sequence.shape[:dim], pad_num, *sequence.shape[dim + 1 :]) if dim != -1 else (*sequence.shape[:dim], pad_num)
    )
    pad = torch.full(pad_shape, padding_value, dtype=sequence.dtype, device=sequence.device)
    sequence = torch.cat([sequence, pad], dim=dim)
    return sequence


def pad_to_max_length(sequence, padding_value, max_length, dim=-1):
    length = sequence.shape[dim]
    assert length <= max_length
    pad_num = max_length - length
    pad_shape = (
        (*sequence.shape[:dim], pad_num, *sequence.shape[dim + 1 :]) if dim != -1 else (*sequence.shape[:dim], pad_num)
    )
    pad = torch.full(pad_shape, padding_value, dtype=sequence.dtype, device=sequence.device)
    sequence = torch.cat([sequence, pad], dim=dim)
    return sequence


def unpack_sequence(packed: torch.Tensor, num_tokens: torch.Tensor | list, dim=1):
    if isinstance(num_tokens, torch.Tensor):
        num_tokens = num_tokens.tolist()
    sequences = torch.split(packed, num_tokens, dim=dim)
    return sequences


def pack_sequence(sequences, dim=1):
    num_tokens = torch.IntTensor([seq.size(dim) for seq in sequences])
    packed = torch.cat(sequences, dim=dim)
    return packed, num_tokens.to(packed.device)


def packed_cumulative_length(num_tokens: torch.Tensor):
    device = num_tokens.device
    _zero_pad = torch.zeros(1, device=device)
    _pad_length = torch.cat([_zero_pad, num_tokens]).int()
    return torch.cumsum(_pad_length, 0).int()


def split_for_sequence_parallel(input: T, dim: int, sp_mesh: DeviceMesh) -> T:
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    sp_group = sp_mesh.get_group()
    sp_size = sp_mesh.size()
    if sp_size == 1:
        return input

    rank = dist.get_rank(sp_group)
    dim_size = input.size(dim)
    assert dim_size % sp_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of sp size ({sp_size}), cannot split tensor evenly"
    )

    tensor_list = torch.split(input, dim_size // sp_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return cast(T, output)


def gather_for_sequence_parallel(input, dim: int, sp_group: dist.ProcessGroup):
    """Gathers the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be gathered.
        dim: The dimension along which the tensor should be gathered.
        sp_group: The sequence parallel process group.

    Returns:
        The gathered tensor concatenated along the specified dimension.
    """
    input = input.contiguous()
    world_size = dist.get_world_size(sp_group)
    dist.get_rank(sp_group)

    if world_size == 1:
        return input

    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    assert input.device.type == "cuda"
    dist.all_gather(tensor_list, input, group=sp_group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output
