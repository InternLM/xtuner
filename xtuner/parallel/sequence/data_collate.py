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


# This function only meets the following two conditions:
# 1. use_varlen_attn = True
# 2. pack_to_max_length = True and the lengths of each sequence are different
def pad_cumulative_len_for_sequence_parallel(cumulative_len):
    assert len(cumulative_len) == 1
    seqlen = cumulative_len[0][-1]
    seq_parallel_world_size = get_sequence_parallel_world_size()
    if seqlen % seq_parallel_world_size == 0:
        return cumulative_len, None

    bs = len(cumulative_len)
    pad_len = seq_parallel_world_size - (seqlen % seq_parallel_world_size)
    seqlen_new = seqlen + pad_len
    attention_mask = torch.zeros(
        bs, seqlen_new, dtype=torch.bool, device=cumulative_len[0].device)
    attention_mask[:, :seqlen] = True

    for i, cu_len in enumerate(cumulative_len):
        pad = torch.tensor([seqlen_new],
                           device=cu_len.device,
                           dtype=cu_len.dtype)
        cumulative_len[i] = torch.cat([cu_len, pad], dim=0)

    return cumulative_len, attention_mask
