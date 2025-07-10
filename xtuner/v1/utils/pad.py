# Copyright (c) OpenMMLab. All rights reserved.

import torch


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
