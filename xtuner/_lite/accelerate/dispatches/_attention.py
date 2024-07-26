import torch
from torch.nn import functional as F

from xtuner._lite.parallel import sequence_parallel_wrapper

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         unpad_input)
    SUPPORT_FLASH2 = True
except ImportError:
    pass


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def upad_qkv(query_layer, key_layer, value_layer, attention_mask,
             query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
        attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                          head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                            head_dim), indices_k)
    if query_length == kv_seq_len:
        # Different from the origin version as sequence parallel change
        # the number of attention heads.
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, -1, head_dim),
            indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = \
            unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


@sequence_parallel_wrapper
def flash_attn_wo_mask(
        query_states,
        key_states,
        value_states,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
):
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size)
    return attn_output


@sequence_parallel_wrapper
def flash_attn_w_mask(
        query_states,  # bs, q_len, nhead, h_dim
        key_states,
        value_states,
        attention_mask,
        causal=True,
        dropout_p=0.0,
        window_size=(-1, -1),  # -1 means infinite context window
):
    batch_size, q_len = query_states.shape[:2]
    query_states, key_states, value_states, indices_q, \
        cu_seq_lens, max_seq_lens = upad_qkv(
            query_states, key_states, value_states, attention_mask, q_len)

    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size)
    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    return attn_output


@sequence_parallel_wrapper
def varlen_flash_attn(
        query_states,
        key_states,
        value_states,
        cumulative_len,
        max_seqlen,
        dropout_p=0.,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
):
    q_unpad, k_unpad, v_unpad = query_states.flatten(0, 1), key_states.flatten(
        0, 1), value_states.flatten(0, 1)
    attn_output = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cumulative_len,
        cumulative_len,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        return_attn_probs=False,
        causal=causal,
        window_size=window_size)
    attn_output = attn_output.unsqueeze(0)
    return attn_output
