from xtuner.parallel.sequence import sequence_parallel_wrapper
from .utils import upad_qkv

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
    SUPPORT_FLASH2 = True
except ImportError:
    pass


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
        softmax_scale=None,
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
        softmax_scale=softmax_scale,
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
        softmax_scale=None,
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
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        return_attn_probs=False,
        causal=causal,
        window_size=window_size)
    attn_output = attn_output.unsqueeze(0)
    return attn_output
