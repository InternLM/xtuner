# Copyright (c) OpenMMLab. All rights reserved.
import torch

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .setup_distributed import (get_sequence_parallel_rank,
                                get_sequence_parallel_world_size)


def pad_for_sequence_parallel(tokens,
                              labels=None,
                              position_ids=None,
                              attention_mask=None,
                              tokens_pad_index=DEFAULT_PAD_TOKEN_INDEX,
                              labels_pad_index=IGNORE_INDEX,
                              position_ids_pad_index=0,
                              attention_mask_pad_index=0):
    if labels is not None:
        assert tokens.shape == labels.shape
    if position_ids is not None:
        assert tokens.shape == position_ids.shape
    if attention_mask is not None:
        assert tokens.shape == attention_mask.shape

    bs, seq_len = tokens.shape
    seq_parallel_world_size = get_sequence_parallel_world_size()
    if seq_len % seq_parallel_world_size == 0:
        return tokens, labels, position_ids, attention_mask

    pad_num = seq_parallel_world_size - (seq_len % seq_parallel_world_size)
    pad = torch.full((bs, pad_num),
                     tokens_pad_index,
                     dtype=tokens.dtype,
                     device=tokens.device)
    tokens = torch.cat([tokens, pad], dim=1)

    if labels is not None:
        pad = torch.full((bs, pad_num),
                         labels_pad_index,
                         dtype=labels.dtype,
                         device=labels.device)
        labels = torch.cat([labels, pad], dim=1)

    if position_ids is not None:
        pad = torch.full((bs, pad_num),
                         position_ids_pad_index,
                         dtype=position_ids.dtype,
                         device=position_ids.device)
        position_ids = torch.cat([position_ids, pad], dim=1)

    if attention_mask is not None:
        pad = torch.full((bs, pad_num),
                         attention_mask_pad_index,
                         dtype=attention_mask.dtype,
                         device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, pad], dim=1)

    return tokens, labels, position_ids, attention_mask


def split_for_sequence_parallel(tokens, labels=None, position_ids=None):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    if seq_parallel_world_size == 1:
        return tokens, labels, position_ids

    seq_parallel_world_rank = get_sequence_parallel_rank()
    seq_len = tokens.size(1)
    assert seq_len % seq_parallel_world_size == 0
    sub_seq_len = seq_len // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_len
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_len

    tokens = tokens[:, sub_seq_start:sub_seq_end]
    if labels is not None:
        labels = labels[:, sub_seq_start:sub_seq_end]
    if position_ids is not None:
        position_ids = position_ids[:, sub_seq_start:sub_seq_end]

    return tokens, labels, position_ids
