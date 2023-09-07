# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def baichuan2_norm_head_forward(self, hidden_states):
    norm_weight = nn.functional.normalize(self.weight)
    return nn.functional.linear(hidden_states, norm_weight)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def baichuan_7b_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(
        0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads,
                                self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads,
                              self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads,
                                self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=attention_mask)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def baichuan_13b_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(
        0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads,
                                self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads,
                              self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads,
                                self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    if attention_mask is not None:
        if q_len == 1:  # inference with cache
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=attention_mask)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
