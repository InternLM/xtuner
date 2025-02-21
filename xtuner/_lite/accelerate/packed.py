# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch


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
