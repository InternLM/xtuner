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
import rotary_emb
from flash_attn.layers.rotary import apply_rotary_emb_func

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.scale_base = scale_base
        self.scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)
    
    def forward(self, x, indexes):
        self._update_cos_sin_cache(x, indexes)
        if isinstance(indexes, int):
            return self._cos_cached[:indexes], self._sin_cached[:indexes]
        if indexes.ndim() == 3:
            assert indexes.shape[0] == 1
            indexes = indexes.squeeze(0)
        return self._cos_cached[indexes], self._sin_cached[indexes]



class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        headdim = q.shape[-1]
        # _, three, _, headdim = qkv.shape
        # assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim == headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q1, q2 = q.chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), q1, q2, False)
        k1, k2 = k.chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            k1, k2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), k1, k2, False
        )
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        dq1, dq2 = dq.chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dq1, dq2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), dq1, dq2, True
        )
        dk1, dk2 = dk.chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dk1, dk2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), dk1, dk2, True
        )
        return dq, dk, None, None, None

apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can
    # `squeeze` them.
    # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def internlm_attn_forward(
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

    if use_local_attn:
        assert len(cumulative_len) == bsz and cumulative_len[0][-1] == q_len

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                                   self.head_dim).transpose(
                                                       1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                                 self.head_dim).transpose(
                                                     1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                                   self.head_dim).transpose(
                                                       1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    
    # if use_local_attn:
    #     # Training
    #     cos, sin = self.rotary_emb(value_states, indexes)
    # else:
    #     cos, sin = self.rotary_emb(value_states, kv_seq_len)
    # query_states = query_states.transpose(1, 2).flatten(0, 1)
    # key_states = key_states.transpose(1, 2).flatten(0, 1)
    # query_states, key_states = apply_rotary_emb_qkv_(query_states, key_states, cos, sin)
    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    if use_local_attn:
        # Training
        cos, sin = self.rotary_emb(value_states, indexes)
        query_states = query_states.transpose(1, 2).flatten(0, 1)
        key_states = key_states.transpose(1, 2).flatten(0, 1)
        # query_states, key_states = apply_rotary_pos_emb(
        #     query_states, key_states, cos, sin, indexes)
        query_states, key_states = apply_rotary_emb_qkv_(query_states, key_states, cos, sin)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    else:
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        is_context_decoding = (q_len != 1)
        seqlen_offsets = 0 if is_context_decoding else position_ids.reshape(-1)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        query_states = apply_rotary_emb_func(
            query_states, cos, sin, 
            seqlen_offsets=seqlen_offsets, interleaved=False, inplace=True
        )
        key_states = apply_rotary_emb_func(
            key_states, cos, sin, 
            seqlen_offsets=seqlen_offsets, interleaved=False, inplace=True
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        # cos, sin = self.rotary_emb(value_states, kv_seq_len)
        # cos = torch.cat((cos, cos), dim=-1)
        # sin = torch.cat((sin, sin), dim=-1)
        # query_states, key_states = apply_rotary_pos_emb(
        #     query_states, key_states, cos, sin, position_ids)

    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(
    #     query_states, key_states, cos, sin,
    #     indexes if use_local_attn else position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    assert SUPPORT_FLASH2
    if SUPPORT_FLASH2 or SUPPORT_XFORMERS:
        # q, k, v is [B, H, S, K] and xformers need [B, S, H, K].
        # returns [B, S, H, K]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if SUPPORT_FLASH2:
            if use_local_attn:
                q_unpad, k_unpad, v_unpad = query_states.flatten(
                    0, 1), key_states.flatten(0,
                                              1), value_states.flatten(0, 1)
                for i in range(1, bsz):
                    cumulative_len[i] += q_len * i
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
            attn_output = xops.memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=xops.LowerTriangularMask())
    else:
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


