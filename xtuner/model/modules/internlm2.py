# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub

SUPPORT_XFORMERS = False
SUPPORT_FLASH2 = False
try:
    import xformers.ops as xops

    SUPPORT_XFORMERS = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    SUPPORT_FLASH2 = True
except ImportError:
    pass


from einops import rearrange


class InternLM2RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000000, device=None):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self.max_seq_len_cached = seq_len
            assert self.inv_freq.dtype == torch.float32
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(t.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can
    # `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def internlm2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    message_hub = MessageHub.get_instance('for_flash_attn')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    indexes = message_hub.get_info(f'indexes_rank_{rank}')
    max_seqlen = message_hub.get_info(f'max_seqlen_rank_{rank}')
    use_local_attn = cumulative_len is not None
    # Modified from https://huggingface.co/internlm/internlm-7b/blob/939a68c0dc1bd5f35b63c87d44af05ce33379061/modeling_internlm.py#L161  # noqa:E501
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1
    assert SUPPORT_FLASH2

    if use_local_attn:
        assert len(cumulative_len) == bsz and cumulative_len[0][-1] == q_len
    
    # query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
    #                                                self.head_dim).transpose(1, 2)
    # key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads,
    #                                              self.head_dim).transpose(1, 2)
    # value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads,
    #                                                self.head_dim).transpose(1, 2)
    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    
    cos, sin = self.rotary_emb(value_states, indexes.max().item() + 1) if use_local_attn else self.rotary_emb(value_states, kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, indexes if use_local_attn else position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    assert SUPPORT_FLASH2
    if SUPPORT_FLASH2:
        # q, k, v is [B, H, S, K] and flash_attn need [B, S, H, K].
        # returns [B, S, H, K]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if use_local_attn:
            q_unpad, k_unpad, v_unpad = query_states.flatten(
                0, 1), key_states.flatten(0,
                                            1), value_states.flatten(0, 1)
            cumulative_len = torch.cat(cumulative_len, dim=0)
            attn_output = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cumulative_len,
                cumulative_len,
                max_seqlen,
                max_seqlen,
                0,
                return_attn_probs=False,
                causal=True,
            )
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, causal=True)
    else:
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value
