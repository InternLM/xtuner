# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.sparse_mla import (
    DSATopKIndicesProtocol,
    SparseMLAProtocol,
    ensure_tilelang_runtime_available,
    get_dsa_topk_indices,
    get_sparse_mla,
)

from ..linear import build_linear
from .attn_outputs import AttnOutputs
from .mla import MLAConfig, MultiLatentAttention, mla_apply_rotary_pos_emb


def is_dsa_skip_topk_layer(layer_idx: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 0-indexed layer reuses another layer's top-k."""

    layer_number = layer_idx + 1
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def dsa_source_compute_layer(layer_idx: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the 0-indexed computing layer used by a shared DSA layer."""

    source = layer_idx
    while is_dsa_skip_topk_layer(source, skip_topk_offset, topk_freq):
        source -= 1
    return source


class LayerNorm(nn.Module):
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        if isinstance(self.bias, DTensor):
            bias = self.bias.to_local()
        else:
            bias = self.bias

        return torch.nn.functional.layer_norm(hidden_states, self.normalized_shape, weight, bias, self.eps)

    def init_weights(self):
        self.weight.data.fill_(1.0)
        self.bias.data.zero_()

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}"


class DSAIndexer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        indexer_backend: Literal["torch", "tilelang"] = "torch",
    ):
        super().__init__()
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.indexer_backend = indexer_backend
        self.dsa_topk_indices_func: DSATopKIndicesProtocol = get_dsa_topk_indices(indexer_backend)
        self.wq_b = build_linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = build_linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = LayerNorm(index_head_dim, eps=1e-6)
        self.weights_proj = build_linear(hidden_size, index_n_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.wq_b(q_resid).view(bsz, seq_len, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.transpose(1, 2)

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # GLM-MoE-DSA applies interleaved RoPE in the indexer, matching HF PR #46842.
        cos, sin = position_embeddings
        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2).squeeze(2)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        weights = self.weights_proj(hidden_states).float() * (self.index_n_heads**-0.5)

        with torch.no_grad():
            return self.dsa_topk_indices_func(
                q,
                k,
                weights,
                seq_ctx,
                index_head_dim=self.index_head_dim,
                index_topk=self.index_topk,
            )


class DSAMLAConfig(MLAConfig):
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 0
    indexer_rope_interleave: bool = True
    indexer_types: list[str] | None = None
    sparse_mla_backend: Literal["torch", "tilelang"] = "torch"

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "DSAMultiLatentAttention":
        if self.sparse_mla_backend == "tilelang":
            ensure_tilelang_runtime_available()

        return DSAMultiLatentAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            rope_scaling_cfg=rope_scaling_cfg,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )


class DSAMultiLatentAttention(MultiLatentAttention):
    def __init__(
        self,
        *,
        index_topk: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk_freq: int = 1,
        index_skip_topk_offset: int = 0,
        indexer_rope_interleave: bool = True,
        indexer_types: list[str] | None = None,
        sparse_mla_backend: Literal["torch", "tilelang"] = "torch",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # DSA absorbed MLA reads kv_b_proj.weight directly and reshapes it to
        # (num_heads, qk_nope + v_dim, kv_lora_rank) before two einsums. The
        # current FP8 tensor wrapper only supports views whose last two
        # dimensions are 128-aligned, while GLM-5.2 has qk_nope + v_dim = 448.
        # Keep this projection in bf16 until there is a fused FP8 absorbed MLA
        # path for the direct-weight computation below.
        if self.float8_cfg is not None:
            self.kv_b_proj = build_linear(
                self.kv_lora_rank,
                self.num_attention_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
                bias=False,
                float8_cfg=None,
            )

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk_freq = index_topk_freq
        self.index_skip_topk_offset = index_skip_topk_offset
        self.indexer_rope_interleave = indexer_rope_interleave
        self.indexer_types = indexer_types
        self.sparse_mla_backend = sparse_mla_backend
        self.sparse_mla_func: SparseMLAProtocol = get_sparse_mla(sparse_mla_backend)

        if self.q_lora_rank is None:
            raise ValueError("DSA MLA requires q_lora_rank because the indexer consumes q_a_layernorm output.")

        if self._is_skip_topk_layer():
            self.source_layer_idx = self._source_compute_layer()
            return

        self.source_layer_idx = self.layer_idx
        self.indexer = DSAIndexer(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            index_head_dim=self.index_head_dim,
            index_n_heads=self.index_n_heads,
            index_topk=self.index_topk,
            indexer_backend=self.sparse_mla_backend,
        )

    def _is_skip_topk_layer(self) -> bool:
        if self.indexer_types is not None:
            if self.layer_idx >= len(self.indexer_types):
                return True
            return self.indexer_types[self.layer_idx] == "shared"
        if self.index_topk_freq <= 1:
            return False
        return is_dsa_skip_topk_layer(self.layer_idx, self.index_skip_topk_offset, self.index_topk_freq)

    def _source_compute_layer(self) -> int:
        if self.indexer_types is not None:
            # MTP layers sit after the main layer stack in XTuner/HF naming, while
            # GLM's indexer_types only describes the main layers. They reuse the
            # nearest preceding full indexer from the main stack.
            start_idx = min(self.layer_idx, len(self.indexer_types) - 1)
            for idx in range(start_idx, -1, -1):
                if self.indexer_types[idx] == "full":
                    return idx
            raise ValueError(f"DSA shared layer {self.layer_idx} has no preceding full indexer layer.")
        return dsa_source_compute_layer(self.layer_idx, self.index_skip_topk_offset, self.index_topk_freq)

    def _get_topk_indices(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        if self._is_skip_topk_layer():
            if self.source_layer_idx not in seq_ctx.dsa_topk_indices:
                raise AssertionError(
                    "DSA index-share: skip layer "
                    f"{self.layer_idx} needs source layer {self.source_layer_idx} top-k, "
                    "but it is not present in this microbatch SequenceContext. "
                    "Cross-pipeline top-k sharing is not supported."
                )
            return seq_ctx.dsa_topk_indices[self.source_layer_idx]

        topk_indices = self.indexer(
            hidden_states.detach(),
            q_resid.detach(),
            position_embeddings,
            seq_ctx,
        )
        seq_ctx.dsa_topk_indices[self.layer_idx] = topk_indices
        return topk_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "DSA MLA training path expects packed batch size 1."
        assert self.q_lora_rank is not None

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_resid).view(bsz, q_len, self.num_attention_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_compressed = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        if isinstance(self.kv_b_proj.weight, DTensor):
            wkv_b = self.kv_b_proj.weight.to_local()
        else:
            wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_attention_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_kc, w_vc = torch.split(wkv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

        q_nope = torch.einsum("bhsd,hdm->bhsm", q_nope, w_kc)
        query_states = torch.cat([q_nope, q_pe], dim=-1).squeeze(0).transpose(0, 1).contiguous()
        key_states = torch.cat([kv_compressed, k_pe.transpose(1, 2).squeeze(2)], dim=-1)
        key_states = key_states.squeeze(0).unsqueeze(1).contiguous()

        topk_indices = self._get_topk_indices(hidden_states, q_resid, position_embeddings, seq_ctx)
        sparse_mla_outputs = self.sparse_mla_func(
            query_states,
            key_states,
            topk_indices,
            self.softmax_scale,
            value_dim=self.kv_lora_rank,
        )
        raw_output = sparse_mla_outputs.raw_output
        softmax_lse = sparse_mla_outputs.softmax_lse
        raw_output = torch.einsum("shm,hdm->shd", raw_output, w_vc)
        raw_output = raw_output.reshape(bsz, q_len, self.num_attention_heads * self.v_head_dim).contiguous()
        projected_output = self.o_proj(raw_output)

        return {
            "raw_output": raw_output,
            "projected_output": projected_output,
            "softmax_lse": softmax_lse,
        }
