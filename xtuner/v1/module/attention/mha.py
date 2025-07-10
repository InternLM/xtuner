# Copyright (c) OpenMMLab. All rights reserved.
from typing import cast

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from xtuner.v1.config import BaseAttnConfig, TransformerConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import get_logger

from ..linear.linear import build_linear
from ..rms_norm import RMSNorm
from .kv_cache import fill_paged_kv_cache


logger = get_logger()


class MHAConfig(BaseAttnConfig):
    num_key_value_heads: int
    head_dim: int = 128
    dropout: bool = False
    causal: bool = True
    qkv_bias: bool = False
    qk_norm: bool = False
    rms_norm_eps: float = 1e-06
    o_bias: bool = False


@torch.library.custom_op("xtuner::paged_attention_decoding", mutates_args=())
def paged_attention_decoding(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    bs = block_table.size(0)
    attn_outputs = cast(
        torch.Tensor,
        flash_attn_with_kvcache(
            query_states.transpose(1, 2).transpose(0, 1)[:bs],
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        ),
    )
    return attn_outputs


@paged_attention_decoding.register_fake
def paged_attention_decoding_fake(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
):
    bs = block_table.size(0)
    return torch.empty_like(query_states.transpose(1, 2).transpose(0, 1)[:bs])


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        attn_config = config.attention
        if not isinstance(attn_config, MHAConfig):
            raise TypeError(f"Expected config to be of type MHAConfig, but got {type(attn_config)}")
        self.config = cast(MHAConfig, attn_config)
        self.model_config = config
        self.layer_idx = layer_idx

        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.dropout
        self.is_causal = self.config.causal

        self.q_proj = build_linear(
            config.hidden_size,
            self.config.num_attention_heads * self.head_dim,
            bias=self.config.qkv_bias,
            float8_cfg=config.float8_cfg,
        )
        self.k_proj = build_linear(
            config.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias=self.config.qkv_bias,
            float8_cfg=config.float8_cfg,
        )
        self.v_proj = build_linear(
            config.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias=self.config.qkv_bias,
            float8_cfg=config.float8_cfg,
        )
        self.o_proj = build_linear(
            self.config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=self.config.o_bias,
            float8_cfg=config.float8_cfg,
        )

        if self.config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.config.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        fill_paged_kv_cache(
            key_states,
            value_states,
            past_key_values[self.layer_idx][0],
            past_key_values[self.layer_idx][1],
            seq_ctx.cu_seq_lens_q,
            seq_ctx.cu_seq_lens_k,
            seq_ctx.max_length_q,
            seq_ctx.max_length_k,
            seq_ctx.block_table,
        )

        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1
        attn_output = cast(
            torch.Tensor,
            flash_attn_varlen_func(
                query_states.transpose(1, 2).squeeze(0),
                key_states.transpose(1, 2).squeeze(0),
                value_states.transpose(1, 2).squeeze(0),
                cu_seqlens_q=seq_ctx.cu_seq_lens_q,
                cu_seqlens_k=seq_ctx.cu_seq_lens_k,
                max_seqlen_q=seq_ctx.max_length_q,
                max_seqlen_k=seq_ctx.max_length_k,
                dropout_p=self.config.dropout,
                causal=True,
            ),
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        assert seq_ctx.block_table is not None
        assert self.layer_idx is not None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.config.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        seq_lens_k = seq_ctx.seq_lens_k
        block_table = seq_ctx.block_table
        block_size = past_key_values[self.layer_idx][0].size(1)
        bs = block_table.size(0)

        assert seq_ctx.cu_seq_lens_k.numel() - 1 == bs, f"{seq_ctx.cu_seq_lens_k.numel()}, {bs}"

        _key_states = key_states.transpose(1, 2).squeeze(0)
        _value_states = value_states.transpose(1, 2).squeeze(0)

        # torch.distributed.breakpoint()
        block_index = block_table[:, 0] + (seq_lens_k[:bs] - 1) // block_size
        past_key_values[self.layer_idx][0][block_index, (seq_lens_k[:bs] - 1) % block_size] = _key_states
        past_key_values[self.layer_idx][1][block_index, (seq_lens_k[:bs] - 1) % block_size] = _value_states

        attn_output = paged_attention_decoding(
            query_states,
            past_key_values[self.layer_idx][0],
            past_key_values[self.layer_idx][1],
            seq_lens_k,
            block_table,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        """Forward pass for the Multi-Head Attention module.

        This method dispatches to specific forward implementations based on the
        attention context (training, prefilling, or decoding).

        Args:
            hidden_states (torch.Tensor): The input hidden states, typically of shape
                (batch_size, seq_len, hidden_size).
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Tuple containing
                positional embedding tensors for rotary position embeddings (cos, sin).
            seq_ctx (SequenceContext): Context information about the sequences being processed,
                containing metadata like sequence lengths and attention masks.
            past_key_values (list[list[torch.Tensor]] | None, optional): Cached key and value
                states from previous forward passes. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after attention computation and projection.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.config.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            sp_size = seq_ctx.sequence_parallel_mesh.size()
            num_kv_heads = key_states.size(1)
            if sp_size > num_kv_heads:
                assert sp_size % num_kv_heads == 0
                key_states = repeat_kv(key_states, sp_size // num_kv_heads)
                value_states = repeat_kv(value_states, sp_size // num_kv_heads)

            query_states = ulysses_all_to_all(
                query_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            key_states = ulysses_all_to_all(
                key_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            value_states = ulysses_all_to_all(
                value_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )

        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1

        assert isinstance(seq_ctx.max_length_q, int)
        assert isinstance(seq_ctx.max_length_k, int)
        attn_output: torch.Tensor = cast(
            torch.Tensor,
            flash_attn_varlen_func(
                query_states.transpose(1, 2).squeeze(0),
                key_states.transpose(1, 2).squeeze(0),
                value_states.transpose(1, 2).squeeze(0),
                cu_seqlens_q=seq_ctx.cu_seq_lens_q,
                cu_seqlens_k=seq_ctx.cu_seq_lens_k,
                max_seqlen_q=seq_ctx.max_length_q,
                max_seqlen_k=seq_ctx.max_length_k,
                dropout_p=self.config.dropout,
                causal=True,
            ),
        )
        # if dist.get_rank() == 0:
        #     print("[FlashAttn Output] attn_output shape:", attn_output.shape)

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            attn_output = ulysses_all_to_all(
                attn_output,
                scatter_dim=0,
                gather_dim=1,
                mesh=seq_ctx.sequence_parallel_mesh,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.head_dim
        num_heads = self.num_key_value_heads

        generate_config = self.model_config.generate_config
        assert generate_config is not None, "Model configuration for generation is not set."

        max_length = max_length or generate_config.max_length
        block_size = block_size or generate_config.block_size
        max_batch_size = max_batch_size or generate_config.max_batch_size

        num_blocks = min(max_batch_size, max_length // block_size * max_batch_size)
        block_size = block_size or generate_config.block_size

        if generate_config.dtype == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {generate_config.dtype}")

        cache_k = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="cuda")
        cache_v = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="cuda")

        return cache_k, cache_v
