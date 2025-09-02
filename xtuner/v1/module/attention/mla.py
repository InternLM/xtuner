# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Literal, cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.config import BaseAttnConfig, Float8Config, GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops import flash_attn_varlen_func
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from ..linear.linear import build_linear
from ..rms_norm import RMSNorm


logger = get_logger()


class MLAConfig(BaseAttnConfig["MultiLatentAttention"]):
    kv_lora_rank: int
    q_lora_rank: int | None
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "MultiLatentAttention":
        return MultiLatentAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )


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
        *,
        head_dim: int,
        hidden_size: int,
        num_heads: int = 1,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None = None,
        dropout: float = False,
        # casual: bool = True,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        o_bias: bool = False,
        rope_scaling_config: RopeScalingConfig | None = None,
        float8_cfg: Float8Config | None = None,
        generate_config: GenerateConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        sliding_window: int = -1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.dropout = dropout
        self.num_heads = num_heads
        # self.causal = casual
        self.qkv_bias = qkv_bias
        self.o_bias = o_bias
        self.qk_norm = qk_norm
        self.float8_cfg = float8_cfg
        self.generate_config = generate_config
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.layer_idx = layer_idx

        if self.q_lora_rank is None:
            self.q_proj = build_linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
                float8_cfg=self.float8_cfg,
            )
        else:
            self.q_a_proj = build_linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=self.qkv_bias,
                float8_cfg=self.float8_cfg,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = build_linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                float8_cfg=self.float8_cfg,
            )

        self.kv_a_proj_with_mqa = build_linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = build_linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
            float8_cfg=self.float8_cfg,
        )

        self.o_proj = build_linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.o_bias,
            float8_cfg=self.float8_cfg,
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)

        if rope_scaling_config is not None:
            mscale_all_dim = (
                rope_scaling_config.mscale_all_dim if rope_scaling_config.mscale_all_dim is not None else 0.0
            )
            scaling_factor = rope_scaling_config.factor
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.window_size = (-1, -1)
        if layer_type == "sliding_attention":
            self.window_size = (sliding_window, sliding_window)

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
            dropout_p=self.dropout,
            window_size=self.window_size,
            softmax_scale=self.softmax_scale,
            causal=True,
            deterministic=XTUNER_DETERMINISTIC,
        )
        attn_output = cast(torch.Tensor, attn_output)
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
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

        assert seq_ctx.block_table is not None
        bs = seq_ctx.block_table.size(0)
        from lmdeploy.pytorch.kernels import fill_kv_cache

        # TODO: support sliding attention in prefilling
        assert self.window_size == (-1, -1), "Sliding attention in prefilling is not supported yet."

        fill_kv_cache(
            torch.cat([compressed_kv, k_pe], dim=-1),
            k_pe.new_empty(bsz, q_len, 1, 0),
            past_key_values[self.layer_idx][0],
            past_key_values[self.layer_idx][1],
            seq_ctx.cu_seq_lens_q[:bs].cuda(),  # q_start_loc
            seq_ctx.seq_lens_q.cuda(),  # q_seq_length
            kv_seq_length=seq_ctx.seq_lens_k.cuda(),
            max_q_seq_length=seq_ctx.seq_lens_q.max().cuda(),
            block_offsets=seq_ctx.block_table,
        )  # type: ignore[assignment]

        attn_output: torch.Tensor = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens_q=seq_ctx.cu_seq_lens_q,
            cu_seqlens_k=seq_ctx.cu_seq_lens_k,
            max_seqlen_q=seq_ctx.max_length_q,
            max_seqlen_k=seq_ctx.max_length_k,
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            causal=True,
        )  # type: ignore[assignment]

        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
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

        assert seq_ctx.block_table is not None
        bs = seq_ctx.block_table.size(0)

        seq_lens_k = seq_ctx.seq_lens_k
        block_table = seq_ctx.block_table
        block_size = past_key_values[self.layer_idx][0].size(1)
        bs = block_table.size(0)

        assert seq_ctx.cu_seq_lens_k.numel() - 1 == bs, f"{seq_ctx.cu_seq_lens_k.numel()}, {bs}"

        block_index = block_table[:, 0] + (seq_lens_k[:bs] - 1) // block_size
        past_key_values[self.layer_idx][0][block_index, (seq_lens_k[:bs] - 1) % block_size] = torch.cat(
            [compressed_kv, k_pe], dim=-1
        )
        # past_key_values[self.layer_idx][0][block_index, (seq_lens_k[:bs] - 1) % block_size, :, self.kv_lora_rank:] = k_pe

        # TODO: support sliding attention in prefilling
        assert self.window_size == (-1, -1), "Sliding attention in prefilling is not supported yet."

        attn_output = flash_mla_decoding(
            query_states.view(q_len, bsz, self.num_heads, -1),
            past_key_values[self.layer_idx][0],
            cache_seqlens=seq_ctx.seq_lens_k,
            softmax_scale=self.softmax_scale,
            block_table=seq_ctx.block_table,
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
            cu_seqlens_q=seq_ctx.cu_seq_lens_q,
            cu_seqlens_k=seq_ctx.cu_seq_lens_k,
            max_seqlen_q=seq_ctx.max_length_q,
            max_seqlen_k=seq_ctx.max_length_k,
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        attn_output = cast(torch.Tensor, attn_output)
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        num_heads = 1

        generate_config = self.generate_config
        assert generate_config is not None, "Model configuration for generation is not set."

        max_length = max_length or generate_config.max_length
        block_size = block_size or generate_config.block_size
        max_batch_size = max_batch_size or generate_config.max_batch_size

        num_blocks = min(max_batch_size, max_length // block_size * max_batch_size)

        if generate_config.dtype == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {generate_config.dtype}")

        cache_k = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="cuda")
        cache_v = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="cuda")

        return cache_k, cache_v
