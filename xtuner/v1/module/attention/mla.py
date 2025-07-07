# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import cast

import torch
from flash_attn import flash_attn_varlen_func
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.config import BaseAttnConfig, TransformerConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import State, get_logger

from ..rms_norm import RMSNorm


logger = get_logger()


class MLAConfig(BaseAttnConfig):
    num_attention_heads: int = 128
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    head_dim: int = 128
    v_head_dim: int = 128
    causal: bool = True
    o_bias: bool = False
    qkv_bias: bool = False
    dropout: bool = False


@torch.library.custom_op("xpuyu::flash_mla_decoding", mutates_args=())
def flash_mla_decoding(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    head_dim_v: int,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, num_heads // 1, 1)

    attn_output, _ = flash_mla_with_kvcache(
        query_states,
        key_cache,
        cache_seqlens=cache_seqlens,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=block_table,
        head_dim_v=head_dim_v,
    )
    return attn_output


@flash_mla_decoding.register_fake
def flash_mla_decoding_fake(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    softmax_scale: torch.Tensor,
    num_heads: int,
    head_dim_v: int,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
):
    b, s, h, d = query_states.shape
    return query_states.new_empty(b, s, h, head_dim_v)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def mla_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MultiLatentAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        if not isinstance(config.attention, MLAConfig):
            raise TypeError(f"Expected config.attention to be MLAConfig, but got {type(config.attention)}")
        self.config = config.attention
        self.layer_idx = layer_idx

        self.is_causal = self.config.causal
        self.attention_dropout = self.config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = self.config.num_attention_heads

        self.q_lora_rank = self.config.q_lora_rank
        self.qk_rope_head_dim = self.config.qk_rope_head_dim
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.qk_nope_head_dim = self.config.qk_nope_head_dim
        self.q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=self.config.qkv_bias)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            bias=self.config.qkv_bias,
        )
        self.kv_a_layernorm = RMSNorm(self.config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.config.o_bias,
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)

        rope_scaling = getattr(config, "rope_scaling", None)

        if rope_scaling is not None:
            mscale_all_dim: float = rope_scaling.mscale_all_dim if rope_scaling.mscale_all_dim is not None else 0.0
            scaling_factor: float = rope_scaling.factor
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward_training(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_meta: SequenceContext,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = position_embeddings
        # cos = torch.load('cos.pth').cuda()
        # sin = torch.load('sin.pth').cuda()
        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1
        attn_output = flash_attn_varlen_func(
            query_states.transpose(1, 2).squeeze(0),
            key_states.transpose(1, 2).squeeze(0),
            value_states.transpose(1, 2).squeeze(0),
            cu_seqlens_q=attn_meta.cu_seq_lens_q,
            cu_seqlens_k=attn_meta.cu_seq_lens_k,
            max_seqlen_q=attn_meta.max_length_q,
            max_seqlen_k=attn_meta.max_length_k,
            dropout_p=self.config.dropout,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        attn_output = cast(torch.Tensor, attn_output)
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output

    def forward_prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_meta: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states).view(
            bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim
        )
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, -1, self.qk_rope_head_dim)

        # k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b_proj(compressed_kv).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = position_embeddings
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)
        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)

        if isinstance(self.kv_b_proj.weight, DTensor):
            wkv_b = self.kv_b_proj.weight.to_local()
        else:
            wkv_b = self.kv_b_proj.weight

        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)

        query_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        assert attn_meta.block_table is not None
        bs = attn_meta.block_table.size(0)
        from lmdeploy.pytorch.kernels import fill_kv_cache

        fill_kv_cache(
            torch.cat([compressed_kv, k_pe], dim=-1),
            k_pe.new_empty(bsz, q_len, 1, 0),
            past_key_values[self.layer_idx][0],
            past_key_values[self.layer_idx][1],
            attn_meta.cu_seq_lens_q[:bs].cuda(),  # q_start_loc
            attn_meta.seq_lens_q.cuda(),  # q_seq_length
            kv_seq_length=attn_meta.seq_lens_k.cuda(),
            max_q_seq_length=attn_meta.seq_lens_q.max().cuda(),
            block_offsets=attn_meta.block_table,
        )

        attn_output = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens_q=attn_meta.cu_seq_lens_q,
            cu_seqlens_k=attn_meta.cu_seq_lens_k,
            max_seqlen_q=attn_meta.max_length_q,
            max_seqlen_k=attn_meta.max_length_k,
            dropout_p=self.config.dropout,
            softmax_scale=self.softmax_scale,
            causal=True,
        )

        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output

    # @torch.compile(fullgraph=True)
    def forward_decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_meta: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        # q_nope, q_pe = torch.split(
        #     q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        # )

        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states).view(
            bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim
        )
        # compressed_kv, k_pe = torch.split(
        #     compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        # )

        _compressed_kv = compressed_kv[..., : self.kv_lora_rank]
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        compressed_kv = _compressed_kv

        compressed_kv = self.kv_a_layernorm(compressed_kv).view(bsz, q_len, -1, self.kv_lora_rank)
        k_pe = k_pe.view(bsz, q_len, -1, self.qk_rope_head_dim)

        # k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        # kv = (
        #     self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        #     .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        #     .transpose(1, 2)
        # )

        # k_nope, value_states = torch.split(
        #     kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        # )

        cos, sin = position_embeddings

        # cos = torch.load('cos.pth').cuda()
        # sin = torch.load('sin.pth').cuda()
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)

        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)

        if isinstance(self.kv_b_proj.weight, DTensor):
            wkv_b = self.kv_b_proj.weight.to_local()
        else:
            wkv_b = self.kv_b_proj.weight

        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)

        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim])

        query_states = torch.cat([q_nope, q_pe], dim=-1)

        assert attn_meta.block_table is not None
        bs = attn_meta.block_table.size(0)

        seq_lens_k = attn_meta.seq_lens_k
        block_table = attn_meta.block_table
        block_size = past_key_values[self.layer_idx][0].size(1)
        bs = block_table.size(0)

        assert attn_meta.cu_seq_lens_k.numel() - 1 == bs, f"{attn_meta.cu_seq_lens_k.numel()}, {bs}"

        block_index = block_table[:, 0] + (seq_lens_k[:bs] - 1) // block_size
        past_key_values[self.layer_idx][0][block_index, (seq_lens_k[:bs] - 1) % block_size] = torch.cat(
            [compressed_kv, k_pe], dim=-1
        )
        # past_key_values[self.layer_idx][0][block_index, (seq_lens_k[:bs] - 1) % block_size, :, self.kv_lora_rank:] = k_pe

        attn_output = flash_mla_decoding(
            query_states.view(q_len, bsz, self.num_heads, -1),
            past_key_values[self.layer_idx][0],
            cache_seqlens=attn_meta.seq_lens_k,
            softmax_scale=self.softmax_scale,
            block_table=attn_meta.block_table,
            head_dim_v=self.kv_lora_rank,
            num_heads=self.num_heads,
        )

        attn_output = torch.einsum("bshc,hdc->bshd", attn_output, wkv_b[:, -self.v_head_dim :])

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]] | None = None,
        state: State = State.TRAINING,
    ) -> torch.Tensor:
        if state is State.PREFILLING:
            assert past_key_values is not None
            return self.forward_prefilling(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attn_meta=seq_ctx,
                past_key_values=past_key_values,
            )
        elif state is State.DECODING:
            assert past_key_values is not None
            assert seq_ctx.block_table is not None
            return self.forward_decoding(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attn_meta=seq_ctx,
                past_key_values=past_key_values,
            )
        elif state is State.TRAINING:
            return self.forward_training(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attn_meta=seq_ctx,
            )
        else:
            raise NotImplementedError
