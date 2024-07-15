from contextlib import contextmanager

import torch


@contextmanager
def packed_sequence(num_tokens, enable=False):
    from mmengine import MessageHub
    ctx = MessageHub.get_instance('packed_sequence')

    if enable:
        device = num_tokens.device
        _zero_length = torch.zeros(1, device=device)
        _pad_length = torch.cat([_zero_length, num_tokens]).int()
        cumulative_lengths = torch.cumsum(_pad_length, 0).int()
        position_ids = [torch.arange(num.item()) for num in num_tokens]
        position_ids = torch.cat(position_ids, dim=0).to(device)
        position_ids = position_ids.unsqueeze(0)
        ctx.update_info('num_tokens', num_tokens)
        ctx.update_info('position_ids', position_ids)
        ctx.update_info('cumulative_lengths', cumulative_lengths)

    else:
        ctx.update_info('num_tokens', None)
        ctx.update_info('position_ids', None)
        ctx.update_info('cumulative_lengths', None)

    yield

    ctx.update_info('num_tokens', None)
    ctx.update_info('position_ids', None)
    ctx.update_info('cumulative_lengths', None)
