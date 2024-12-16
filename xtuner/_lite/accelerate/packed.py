from contextlib import contextmanager
from typing import List, Union

import torch

from xtuner._lite import get_device
from xtuner._lite.parallel import get_sp_mesh, split_for_sequence_parallel


def unpack_sequence(packed: torch.Tensor,
                    num_tokens: Union[torch.Tensor, List],
                    dim=1):

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


@contextmanager
def packed_sequence(num_tokens, enable=True, sp_mesh=None):
    from mmengine import MessageHub
    ctx = MessageHub.get_instance('packed_sequence')

    device = get_device()
    if enable:
        num_tokens = num_tokens.to(device)
        device = num_tokens.device
        _zero_length = torch.zeros(1, device=device)
        _pad_length = torch.cat([_zero_length, num_tokens]).int()
        cumulative_lengths = torch.cumsum(_pad_length, 0).int()
        position_ids = [torch.arange(num.item()) for num in num_tokens]
        position_ids = torch.cat(position_ids, dim=0).to(device)
        position_ids = position_ids.unsqueeze(0)
        if sp_mesh:
            # `dim` is 1 as the shape of tensor is (bs, seq_len)
            position_ids = split_for_sequence_parallel(
                position_ids, dim=1, sp_mesh=sp_mesh)

        # ctx.update_info('num_tokens', num_tokens)
        ctx.update_info('position_ids', position_ids)
        ctx.update_info('cumulative_lengths', cumulative_lengths)
        ctx.update_info('max_seqlen', num_tokens.max())
        ctx.update_info('sp_mesh', sp_mesh)

    else:
        # ctx.update_info('num_tokens', None)
        ctx.update_info('position_ids', None)
        ctx.update_info('cumulative_lengths', None)
        ctx.update_info('max_seqlen', None)
        ctx.update_info('sp_mesh', None)
    yield

    # ctx.update_info('num_tokens', None)
    ctx.update_info('position_ids', None)
    ctx.update_info('cumulative_lengths', None)
    ctx.update_info('max_seqlen', None)
    ctx.update_info('sp_mesh', None)
