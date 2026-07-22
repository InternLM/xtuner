# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.comm import gather_for_sequence_parallel
from xtuner.v1.ops.sparse_mla import (
    DSATopKIndicesProtocol,
    SparseMLAProtocol,
    ensure_cudnn_dsa_runtime_available,
    ensure_tilelang_runtime_available,
    get_dsa_topk_indices,
    get_sparse_mla,
)

from ..linear import build_linear
from .attn_outputs import AttnOutputs
from .dsa_topk_sharing import build_dsa_topk_release_plan, dsa_topk_source_layer, get_dsa_topk_sharing_runtime
from .mla import MLAConfig, MultiLatentAttention, mla_apply_rotary_pos_emb


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
        indexer_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch",
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

    @torch.no_grad()
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

        # Top-k 索引是整数，不需要梯度，所以整个 indexer 都放在 no_grad 下。
        # 这解释了 Case 1 为什么只在 compile 下显错：
        #   eager COMPUTE: indexer 不产生槽位 -> SparseMLA 保存 [A, B, C]
        #   eager REUSE:   cache read 不产生槽位 -> SparseMLA 保存 [A, B, C]
        # original/replay 虽然走了不同分支，但 checkpoint 看到的保存清单仍能对齐。
        # compile 会把 indexer 周围的可求导计算按 compiled block 打包；COMPUTE 与
        # REUSE 经过不同 graph break 后，可能分别保存 [A, B, C, D] 和
        # [A, X, C, D]，同一槽位的 metadata 不同才触发 CheckpointError。
        # 这里的字母只表示保存槽位，不表示真实变量或 Tensor 数值。
        # Index Q 按 query token 保持分片，只有 K 需要全局 gather。
        k = gather_for_sequence_parallel(k, dim=1, sp_mesh=seq_ctx.sequence_parallel_mesh)
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
    sparse_mla_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch"

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "DSAMultiLatentAttention":
        if self.sparse_mla_backend in ("tilelang", "cudnn_dsa"):
            ensure_tilelang_runtime_available()
        if self.sparse_mla_backend == "cudnn_dsa":
            ensure_cudnn_dsa_runtime_available()

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
        sparse_mla_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch",
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
        if indexer_types is None:
            self.dsa_topk_last_use, self.dsa_topk_recompute_release = {}, {}
        else:
            release_plan = build_dsa_topk_release_plan(
                num_main_layers=len(indexer_types),
                num_mtp_layers=0,
                indexer_types=indexer_types,
                index_skip_topk_offset=index_skip_topk_offset,
                index_topk_freq=index_topk_freq,
            )
            self.dsa_topk_last_use = release_plan.forward_last_use
            self.dsa_topk_recompute_release = release_plan.recompute_release

        if self.q_lora_rank is None:
            raise ValueError("DSA MLA requires q_lora_rank because the indexer consumes q_a_layernorm output.")

        self.source_layer_idx = dsa_topk_source_layer(
            layer_idx=self.layer_idx,
            indexer_types=self.indexer_types,
            index_skip_topk_offset=self.index_skip_topk_offset,
            index_topk_freq=self.index_topk_freq,
        )
        if self.source_layer_idx != self.layer_idx:
            return

        self.indexer = DSAIndexer(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            index_head_dim=self.index_head_dim,
            index_n_heads=self.index_n_heads,
            index_topk=self.index_topk,
            indexer_backend=self.sparse_mla_backend,
        )

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

        # Keep queries sequence-sharded instead of using MHA's Ulysses layout.
        # DSA has only one compressed KV group, so head-to-sequence all-to-all
        # would first have to replicate that group and would not reduce KV memory.
        # Its top-k is also head-independent: every head shard would need the full
        # [global_seq, 1, topk] cache, plus query/output all-to-all. Gathering only
        # the small compressed KV keeps all heads and the large top-k cache local.
        key_states = gather_for_sequence_parallel(key_states, dim=0, sp_mesh=seq_ctx.sequence_parallel_mesh)

        topk_indices = get_dsa_topk_sharing_runtime().get_or_compute(
            layer=self,
            seq_ctx=seq_ctx,
            compute_source_topk=lambda: self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                seq_ctx,
            ),
        )
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
