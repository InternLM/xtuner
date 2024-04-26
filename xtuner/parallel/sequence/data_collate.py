# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .setup_distributed import get_sequence_parallel_world_size


def pad_for_sequence_parallel(tensor, padding_value, dim=-1):
    length = tensor.shape[dim]
    seq_parallel_world_size = get_sequence_parallel_world_size()
    if length % seq_parallel_world_size == 0:
        return tensor

    pad_num = seq_parallel_world_size - (length % seq_parallel_world_size)
    pad_shape = (*tensor.shape[:dim], pad_num,
                 *tensor.shape[dim + 1:]) if dim != -1 else (
                     *tensor.shape[:dim], pad_num)
    pad = torch.full(
        pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, pad], dim=dim)
    return tensor
