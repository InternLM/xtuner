# Copyright (c) OpenMMLab. All rights reserved.

from typing import cast, Literal
import torch
from torch import nn

from transformers.models.llama.modeling_llama import repeat_kv
from xtuner.v1.config import BaseAttnConfig, Float8Config, GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops import apply_rotary_pos_emb, flash_attn_varlen_func
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from ..linear.linear import build_linear
from ..rms_norm import RMSNorm
from .kv_cache import fill_paged_kv_cache


logger = get_logger()


class MHAConfig(BaseAttnConfig["MultiHeadAttention"]):
    num_key_value_heads: int
    dropout: bool = False
    # causal: bool = True
    qkv_bias: bool = False
    qk_norm: bool = False
    rms_norm_eps: float = 1e-06
    o_bias: bool = False

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "MultiHeadAttention":
        return MultiHeadAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )


@torch.library.custom_op("xtuner::paged_attention_decoding", mutates_args=())
def paged_attention_decoding(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache

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

    def __init__(
        self,
        *,
        head_dim: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout: float = False,
        # casual: bool = True,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        o_bias: bool = False,
        float8_cfg: Float8Config | None = None,
        generate_config: GenerateConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        sliding_window: int = -1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout
        # self.is_causal = casual
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.o_bias = o_bias
        self.generate_config = generate_config
        self.float8_cfg = float8_cfg
        self.layer_idx = layer_idx

        self.q_proj = build_linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.k_proj = build_linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.v_proj = build_linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.o_proj = build_linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=self.o_bias,
            float8_cfg=self.float8_cfg,
        )

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        self.window_size = (-1, -1)
        if layer_type == "sliding_attention":
            self.window_size = (sliding_window, sliding_window)

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

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: support sliding attention in prefilling
        assert self.window_size == (-1, -1), "Sliding attention in prefilling is not supported yet."
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
                dropout_p=self.dropout,
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

        if self.qk_norm:
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

        assert self.window_size == (-1, -1), "Sliding attention in prefilling is not supported yet."
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

        if self.qk_norm:
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
                window_size=self.window_size,
                dropout_p=self.dropout,
                causal=True,
                deterministic=XTUNER_DETERMINISTIC,
            ),
        )

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

        generate_config = self.generate_config
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
