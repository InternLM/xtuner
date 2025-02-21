# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch


def pad_to_multiple_of(sequence, padding_value, multiple_of, dim=-1):
    length = sequence.shape[dim]
    if length % multiple_of == 0:
        return sequence

    pad_num = multiple_of - (length % multiple_of)
    pad_shape = (
        (*sequence.shape[:dim], pad_num, *sequence.shape[dim + 1 :])
        if dim != -1
        else (*sequence.shape[:dim], pad_num)
    )
    pad = torch.full(
        pad_shape, padding_value, dtype=sequence.dtype, device=sequence.device
    )
    sequence = torch.cat([sequence, pad], dim=dim)
    return sequence


def pad_to_max_length(sequence, padding_value, max_length, dim=-1):
    length = sequence.shape[dim]
    assert length <= max_length
    pad_num = max_length - length
    pad_shape = (
        (*sequence.shape[:dim], pad_num, *sequence.shape[dim + 1 :])
        if dim != -1
        else (*sequence.shape[:dim], pad_num)
    )
    pad = torch.full(
        pad_shape, padding_value, dtype=sequence.dtype, device=sequence.device
    )
    sequence = torch.cat([sequence, pad], dim=dim)
    return sequence


def unpack_sequence(packed: torch.Tensor, num_tokens: Union[torch.Tensor, List], dim=1):
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
