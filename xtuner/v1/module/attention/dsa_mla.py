# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, NamedTuple, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
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
    dsa_indexer_kl_loss,
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


class DSAIndexerTrainingConfig(BaseModel):
    """Source-layer-only sparse indexer training configuration.

    ``None`` on :class:`DSAMLAConfig` remains the strict frozen baseline.  This
    implementation excludes IndexShare supervision, sequence parallelism,
    activation checkpoint replay, and MTP indexers.

    ``indexer_only`` is a diagnostic overfit mode: GLM freezes the teacher and
    every non-indexer parameter, leaving only main-stack source indexers
    trainable.  ``debug_interval`` prints per-source teacher/student
    distribution statistics without changing the loss.
    """

    model_config = ConfigDict(extra="forbid")
    loss_coeff: float = Field(default=1.0, ge=0.0)
    supervision: Literal["source_layer"] = "source_layer"
    train_mtp_indexer: Literal[False] = False
    indexer_only: bool = False
    debug_interval: int = Field(default=0, ge=0)


class DSAIndexerFeatures(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    selection_weights: torch.Tensor
    training_weights: torch.Tensor


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
        trainable: bool = False,
    ):
        super().__init__()
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.indexer_backend = indexer_backend
        self.dsa_topk_indices_func: DSATopKIndicesProtocol = get_dsa_topk_indices(indexer_backend)
        # wq_b.weight: [index_n_heads * index_head_dim, q_lora_rank]
        self.wq_b = build_linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        # wk.weight: [index_head_dim, hidden_size]
        self.wk = build_linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = LayerNorm(index_head_dim, eps=1e-6)
        # weights_proj.weight: [index_n_heads, hidden_size]
        self.weights_proj = build_linear(hidden_size, index_n_heads, bias=False)
        # ``trainable=False`` is the historical and strict frozen baseline.
        # Top-k selection itself remains no-grad even when sparse KL training is
        # enabled; only ``project_features`` participates in autograd.
        if not trainable:
            self.requires_grad_(False)

    def project_features(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> DSAIndexerFeatures:
        """Project indexer features while preserving an optional autograd graph.

        Shapes use ``S`` for the local sequence length and ``S_g`` for the
        SP-gathered global KV length. Numbers below follow GLM-5.2 defaults
        (``q_lora_rank=2048``, ``hidden_size=6144``, ``index_n_heads=32``,
        ``index_head_dim=128``, ``index_topk=K``).

        Data flow::

            q_resid (1,S,2048) ──wq_b──► q (1,S,32,128) ──RoPE(pe)──► q
            hidden  (1,S,6144) ──wk+norm► k (1,S,128)   ──RoPE(pe)──► k
                                                                      │
                                                                      ├─SP gather──► k (1,S_g,128)
            hidden ──────────weights_proj─────────────────────────────► weights (1,S,32)
                                                                      │
                                                           dsa_topk(q, k, weights)
                                                                      │
                                                                      ▼
                                                           topk_indices (S, 1, K)
        """
        # hidden_states: [bsz, S, hidden_size]
        # q_resid: [bsz, S, q_lora_rank]
        bsz, seq_len, _ = hidden_states.shape

        # q: [bsz, S, Ni, Di]
        q = self.wq_b(q_resid).view(bsz, seq_len, self.index_n_heads, self.index_head_dim)
        # q_pe: [bsz, S, Ni, Dr] -> [bsz, Ni, S, Dr]; q_nope: [bsz, S, Ni, Di - Dr]
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.transpose(1, 2)

        # k: [bsz, S, Di]
        k = self.k_norm(self.wk(hidden_states))
        # k_pe: [bsz, S, Dr] -> [bsz, 1, S, Dr]; k_nope: [bsz, S, Di - Dr]
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # GLM-MoE-DSA applies interleaved RoPE in the indexer, matching HF PR #46842.
        cos, sin = position_embeddings
        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        # q_pe: [bsz, S, Ni, Dr]; k_pe: [bsz, S, Dr]
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2).squeeze(2)

        # q: [bsz, S, Ni, Di]; k: [bsz, S, Di]
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        raw_weights = self.weights_proj(hidden_states)
        # The selection backends apply ``index_head_dim**-0.5`` internally.
        # cuDNN indexer backward uses ``sm_scale=1`` and therefore consumes the
        # complete effective scaling in its BF16 weights tensor.
        selection_weights = raw_weights.float() * (self.index_n_heads**-0.5)
        training_weights = raw_weights * ((self.index_n_heads * self.index_head_dim) ** -0.5)

        # Top-k 索引是整数，不需要梯度，所以 selection 始终放在 no_grad 下。
        # 这解释了 Case 1 为什么只在 compile 下显错：
        #   eager COMPUTE: indexer 不产生槽位 -> SparseMLA 保存 [A, B, C]
        #   eager REUSE:   cache read 不产生槽位 -> SparseMLA 保存 [A, B, C]
        # original/replay 虽然走了不同分支，但 checkpoint 看到的保存清单仍能对齐。
        # compile 会把 indexer 周围的可求导计算按 compiled block 打包；COMPUTE 与
        # REUSE 经过不同 graph break 后，可能分别保存 [A, B, C, D] 和
        # [A, X, C, D]，同一槽位的 metadata 不同才触发 CheckpointError。
        # 这里的字母只表示保存槽位，不表示真实变量或 Tensor 数值。
        # Index Q 按 query token 保持分片，只有 K 需要全局 gather。
        # k: [bsz, S_g, Di]
        k = gather_for_sequence_parallel(k, dim=1, sp_mesh=seq_ctx.sequence_parallel_mesh)
        return DSAIndexerFeatures(q, k, selection_weights, training_weights)

    @torch.no_grad()
    def select_topk(self, features: DSAIndexerFeatures, seq_ctx: SequenceContext) -> torch.Tensor:
        """Select integer sparse IDs without retaining the indexer graph."""

        # returns topk_indices: [S, 1, K]
        return self.dsa_topk_indices_func(
            features.q.detach(),
            features.k.detach(),
            features.selection_weights.detach(),
            seq_ctx,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        features = self.project_features(hidden_states, q_resid, position_embeddings, seq_ctx)
        return self.select_topk(features, seq_ctx)


class DSAMLAConfig(MLAConfig):
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 0
    indexer_rope_interleave: bool = True
    indexer_types: list[str] | None = None
    sparse_mla_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch"
    indexer_training: DSAIndexerTrainingConfig | None = None

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
        indexer_training: DSAIndexerTrainingConfig | dict | None = None,
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
        self.indexer_training = (
            None if indexer_training is None else DSAIndexerTrainingConfig.model_validate(indexer_training)
        )
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
            trainable=self.indexer_training is not None,
        )

    def disable_indexer_training(self) -> None:
        """Restore strict frozen behavior, used by physical MTP layers."""

        self.indexer_training = None
        if hasattr(self, "indexer"):
            self.indexer.requires_grad_(False)
    def get_muon_split_sizes(self) -> dict[nn.Parameter, tuple[int, ...]]:
        """Return the logical row blocks used by GLM MuonSplit."""
        return {
            cast(nn.Parameter, self.q_b_proj.weight): (self.qk_nope_head_dim, self.qk_rope_head_dim)
            * self.num_attention_heads,
            cast(nn.Parameter, self.kv_a_proj_with_mqa.weight): (self.kv_lora_rank, self.qk_rope_head_dim),
            cast(nn.Parameter, self.kv_b_proj.weight): (self.qk_nope_head_dim, self.v_head_dim)
            * self.num_attention_heads,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        """Absorbed DSA-MLA forward for packed training (``bsz == 1``).

        Shapes use ``S`` for the local sequence length and ``S_g`` for the
        SP-gathered global KV length. Numbers below follow GLM-5.2 defaults
        (``hidden_size=6144``, ``N=64``, ``Rq=2048``, ``Rkv=512``,
        ``Dn=192``, ``Dr=64``, ``Dv=256``, ``Dq=256``, ``K=index_topk``).

        Data flow::

            hidden (1,S,6144)
               ├─ Q-LoRA → q_nope(1,64,S,192), q_pe(1,64,S,64)
               │              │ absorb(w_kc)
               │              └─→ query (S,64,576)
               │
               ├─ KV-LoRA → kv_c(1,S,512), k_pe(1,1,S,64)
               │              └─→ key (S,1,576) ─SP gather→ (S_g,1,576)
               │
               └─ Indexer(q_resid, hidden) ─→ topk (S,1,K)
                                                  │
                                SparseMLA(q, key, topk) → (S,64,512)
                                                  │ absorb^{-1}(w_vc)
                                                  └─→ raw (1,S,16384) → o_proj → (1,S,6144)
        """
        # hidden_states: [bsz, S, hidden_size]
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "DSA MLA training path expects packed batch size 1."
        assert self.q_lora_rank is not None

        # q_a_proj.weight: [Rq, hidden_size]; q_resid: [bsz, S, Rq]
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        # q_b_proj.weight: [N * Dq, Rq]; q: [bsz, N, S, Dq]
        q = self.q_b_proj(q_resid).view(bsz, q_len, self.num_attention_heads, self.q_head_dim).transpose(1, 2)
        # q_nope: [bsz, N, S, Dn]; q_pe: [bsz, N, S, Dr]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # kv_a_proj_with_mqa.weight: [Rkv + Dr, hidden_size]
        # compressed_kv: [bsz, S, Rkv + Dr]
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # kv_compressed: [bsz, S, Rkv]; k_pe: [bsz, 1, S, Dr]
        kv_compressed = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q_pe, k_pe = mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # kv_b_proj.weight: [N * (Dn + Dv), Rkv]
        if isinstance(self.kv_b_proj.weight, DTensor):
            wkv_b = self.kv_b_proj.weight.to_local()
        else:
            wkv_b = self.kv_b_proj.weight
        # wkv_b: [N, Dn + Dv, Rkv]
        wkv_b = wkv_b.view(self.num_attention_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        # w_kc: [N, Dn, Rkv]; w_vc: [N, Dv, Rkv]
        w_kc, w_vc = torch.split(wkv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

        # q_nope: [bsz, N, S, Rkv]
        q_nope = torch.einsum("bhsd,hdm->bhsm", q_nope, w_kc)
        # query_states: [S, N, Rkv + Dr]
        query_states = torch.cat([q_nope, q_pe], dim=-1).squeeze(0).transpose(0, 1).contiguous()
        # key_states: [bsz, S, Rkv + Dr] -> [S, 1, Rkv + Dr]
        key_states = torch.cat([kv_compressed, k_pe.transpose(1, 2).squeeze(2)], dim=-1)
        key_states = key_states.squeeze(0).unsqueeze(1).contiguous()

        # Keep queries sequence-sharded instead of using MHA's Ulysses layout.
        # DSA has only one compressed KV group, so head-to-sequence all-to-all
        # would first have to replicate that group and would not reduce KV memory.
        # Its top-k is also head-independent: every head shard would need the full
        # [global_seq, 1, topk] cache, plus query/output all-to-all. Gathering only
        # the small compressed KV keeps all heads and the large top-k cache local.
        # key_states: [S_g, 1, Rkv + Dr]
        key_states = gather_for_sequence_parallel(key_states, dim=0, sp_mesh=seq_ctx.sequence_parallel_mesh)

        indexer_features: DSAIndexerFeatures | None = None
        indexer_loss_enabled = (
            self.indexer_training is not None
            and self.indexer_training.loss_coeff > 0
            and self.source_layer_idx == self.layer_idx
        )
        if indexer_loss_enabled and self.training and not torch.is_grad_enabled():
            raise RuntimeError(
                "DSA indexer training does not support activation checkpointing; set recompute_ratio=0."
            )
        if (
            indexer_loss_enabled
            and seq_ctx.sequence_parallel_mesh is not None
            and seq_ctx.sequence_parallel_mesh.size() > 1
        ):
            raise RuntimeError("DSA indexer training requires sequence parallel size 1.")
        train_source_indexer = indexer_loss_enabled and torch.is_grad_enabled()
        if train_source_indexer:
            # The indexer learns from attention, but must not inject an extra
            # gradient path into the transformer hidden/Q-LoRA activations.
            indexer_features = self.indexer.project_features(
                hidden_states.detach(),
                q_resid.detach(),
                position_embeddings,
                seq_ctx,
            )
            topk_indices = get_dsa_topk_sharing_runtime().get_or_compute(
                layer=self,
                seq_ctx=seq_ctx,
                compute_source_topk=lambda: self.indexer.select_topk(indexer_features, seq_ctx),
            )
        else:
            # ``loss_coeff=0`` follows this no-grad path so optimizer-visible
            # indexer parameters retain ``grad is None`` rather than zero grads.
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
        # raw_output: [S, N, Rkv]; softmax_lse: [S, N]
        raw_output = sparse_mla_outputs.raw_output
        softmax_lse = sparse_mla_outputs.softmax_lse
        if train_source_indexer:
            assert indexer_features is not None
            valid_query_rows = q_len - seq_ctx.num_padding
            if valid_query_rows <= 0:
                raise ValueError("DSA indexer training requires at least one attention-valid query row.")
            valid_query_mask = torch.arange(q_len, device=query_states.device).unsqueeze(0) < valid_query_rows
            indexer_loss = dsa_indexer_kl_loss(
                indexer_features.q,
                indexer_features.k,
                indexer_features.training_weights,
                query_states.unsqueeze(0),
                key_states.squeeze(1).unsqueeze(0),
                softmax_lse.unsqueeze(0),
                topk_indices.squeeze(1).unsqueeze(0),
                softmax_scale=self.softmax_scale,
                row_coefficient=self.indexer_training.loss_coeff / valid_query_rows,
                valid_query_mask=valid_query_mask,
                debug_name=f"layer{self.layer_idx}",
                debug_interval=self.indexer_training.debug_interval,
            )
            seq_ctx.dsa_topk_cache.indexer_losses.append(indexer_loss)
        # raw_output: [S, N, Dv] -> [bsz, S, N * Dv]
        raw_output = torch.einsum("shm,hdm->shd", raw_output, w_vc)
        raw_output = raw_output.reshape(bsz, q_len, self.num_attention_heads * self.v_head_dim).contiguous()
        # o_proj.weight: [hidden_size, N * Dv]; projected_output: [bsz, S, hidden_size]
        projected_output = self.o_proj(raw_output)

        return {
            "raw_output": raw_output,
            "projected_output": projected_output,
            "softmax_lse": softmax_lse,
        }
