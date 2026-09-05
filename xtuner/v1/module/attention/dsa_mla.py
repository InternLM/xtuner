# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, NamedTuple, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.loss import dense_dsa_indexer_kl_loss
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
from .mla import MLAConfig, MultiLatentAttention, mla_apply_rotary_pos_emb, mla_apply_rotary_pos_emb_non_interleaved


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
    """Two-stage DSA indexer distillation configuration.

    ``None`` on :class:`DSAMLAConfig` remains the strict frozen baseline.  This
    ``dense_warmup`` keeps the dense attention teacher frozen and trains new
    source indexers with a query-blocked PyTorch KL. ``sparse`` runs the
    selected sparse backend. Each source indexer is supervised by the attention
    layer that owns it; shared consumer layers only reuse its discrete top-k.

    Sequence parallelism and decoder activation checkpoint replay are not
    supported while indexer loss is active. ``train_mtp_indexer`` additionally
    enables the checkpoint-backed physical MTP source indexer; when one
    physical MTP layer is reused for multiple prediction depths, only the
    first (index-compute) depth is supervised.

    ``indexer_only`` is a diagnostic overfit mode: GLM freezes the teacher and
    every non-indexer parameter, leaving only the selected source indexers
    trainable. ``debug_interval`` prints per-source teacher/student
    distribution statistics without changing the loss.
    """

    model_config = ConfigDict(extra="forbid")
    stage: Literal["dense_warmup", "sparse"] = "sparse"
    loss_coeff: float = Field(default=1.0, ge=0.0)
    train_mtp_indexer: bool = False
    indexer_only: bool = False
    dense_query_block_size: int = Field(default=256, ge=1)
    debug_interval: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_stage(self) -> "DSAIndexerTrainingConfig":
        if self.stage == "dense_warmup" and not self.indexer_only:
            raise ValueError("dense_warmup requires indexer_only=True so the full-attention teacher stays frozen.")
        return self


class DSAIndexerFeatures(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    weights: torch.Tensor


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
        rope_interleave: bool = True,
        indexer_backend: Literal["torch", "tilelang", "cudnn_dsa"] = "torch",
        trainable: bool = False,
    ):
        super().__init__()
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.rope_interleave = rope_interleave
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

        # GLM-5.2 uses interleaved RoPE. GLM-4.7 follows its checkpoint
        # setting for both the dense teacher and newly initialized indexer.
        cos, sin = position_embeddings
        rope_fn = mla_apply_rotary_pos_emb if self.rope_interleave else mla_apply_rotary_pos_emb_non_interleaved
        q_pe, k_pe = rope_fn(q_pe, k_pe, cos, sin)
        # q_pe: [bsz, S, Ni, Dr]; k_pe: [bsz, S, Dr]
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2).squeeze(2)

        # q: [bsz, S, Ni, Di]; k: [bsz, S, Di]
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        weights = self.weights_proj(hidden_states)
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
        return DSAIndexerFeatures(q, k, weights)

    @torch.no_grad()
    def select_topk(self, features: DSAIndexerFeatures, seq_ctx: SequenceContext) -> torch.Tensor:
        """Select integer sparse IDs without retaining the indexer graph."""

        # returns topk_indices: [S, 1, K]
        return self.dsa_topk_indices_func(
            features.q.detach(),
            features.k.detach(),
            features.weights.detach().float() * (self.index_n_heads**-0.5),
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
            rope_interleave=self.indexer_rope_interleave,
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

    def _validate_indexer_training_runtime(self, seq_ctx: SequenceContext) -> None:
        if self.training and not torch.is_grad_enabled():
            raise RuntimeError(
                "DSA indexer training does not support activation checkpointing; set recompute_ratio=0."
            )
        if seq_ctx.sequence_parallel_mesh is not None and seq_ctx.sequence_parallel_mesh.size() > 1:
            raise RuntimeError("DSA indexer training requires sequence parallel size 1.")

    @torch.no_grad()
    def _project_dense_teacher_states(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Q-LoRA residual and explicit dense teacher Q/K states."""

        bsz, q_len, _ = hidden_states.shape
        assert self.q_lora_rank is not None
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_resid).view(bsz, q_len, self.num_attention_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz,
            q_len,
            self.num_attention_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope = kv[..., : self.qk_nope_head_dim]

        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        rope_fn = mla_apply_rotary_pos_emb if self.rope_interleave else mla_apply_rotary_pos_emb_non_interleaved
        q_pe, k_pe = rope_fn(q_pe, k_pe, cos, sin)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2).expand(-1, -1, self.num_attention_heads, -1)
        return q_resid, torch.cat([q_nope, q_pe], dim=-1), torch.cat([k_nope, k_pe], dim=-1)

    def _forward_dense_warmup(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        """Run original dense MLA while distilling only the DSA indexers."""

        dense_outputs = MultiLatentAttention.forward(self, hidden_states, position_embeddings, seq_ctx)
        training_cfg = self.indexer_training
        if training_cfg is None or training_cfg.loss_coeff == 0 or not self.training:
            return dense_outputs
        if self.source_layer_idx != self.layer_idx:
            return dense_outputs

        self._validate_indexer_training_runtime(seq_ctx)
        q_resid, teacher_q, teacher_k = self._project_dense_teacher_states(hidden_states.detach(), position_embeddings)
        features = self.indexer.project_features(
            hidden_states.detach(),
            q_resid.detach(),
            position_embeddings,
            seq_ctx,
        )

        indexer_loss = dense_dsa_indexer_kl_loss(
            features.q,
            features.k,
            features.weights * ((self.index_n_heads * self.index_head_dim) ** -0.5),
            teacher_q,
            teacher_k,
            seq_ctx,
            softmax_scale=self.softmax_scale,
            loss_coefficient=training_cfg.loss_coeff,
            query_block_size=training_cfg.dense_query_block_size,
        )
        seq_ctx.dsa_topk_cache.indexer_losses.append(indexer_loss)
        return dense_outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        training_cfg = self.indexer_training
        if training_cfg is not None and training_cfg.stage == "dense_warmup":
            return self._forward_dense_warmup(hidden_states, position_embeddings, seq_ctx)
        return self._forward_sparse(hidden_states, position_embeddings, seq_ctx)

    def _forward_sparse(
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
        rope_fn = mla_apply_rotary_pos_emb if self.rope_interleave else mla_apply_rotary_pos_emb_non_interleaved
        q_pe, k_pe = rope_fn(q_pe, k_pe, cos, sin)

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

        training_cfg = self.indexer_training
        indexer_features: DSAIndexerFeatures | None = None
        indexer_loss_enabled = (
            training_cfg is not None
            and training_cfg.stage == "sparse"
            and training_cfg.loss_coeff > 0
            and self.source_layer_idx == self.layer_idx
            and self.training
        )
        if indexer_loss_enabled:
            self._validate_indexer_training_runtime(seq_ctx)
        topk_runtime = get_dsa_topk_sharing_runtime()
        # GLM-5.2 runs the physical MTP layer once as an index-compute layer,
        # then reuses its discrete top-k for later logical MTP depths. Mirror
        # that contract during training: later depths must neither rerun the
        # indexer projections nor add duplicate KL terms.
        reuse_mtp_iteration_topk = topk_runtime.reuses_mtp_iteration_topk(layer=self, seq_ctx=seq_ctx)
        train_source_indexer = indexer_loss_enabled and torch.is_grad_enabled() and not reuse_mtp_iteration_topk
        if train_source_indexer:
            # The indexer learns from attention, but must not inject an extra
            # gradient path into the transformer hidden/Q-LoRA activations.
            indexer_features = self.indexer.project_features(
                hidden_states.detach(),
                q_resid.detach(),
                position_embeddings,
                seq_ctx,
            )
            topk_indices = topk_runtime.get_or_compute(
                layer=self,
                seq_ctx=seq_ctx,
                compute_source_topk=lambda: self.indexer.select_topk(indexer_features, seq_ctx),
            )
        else:
            # ``loss_coeff=0`` follows this no-grad path so optimizer-visible
            # indexer parameters retain ``grad is None`` rather than zero grads.
            topk_indices = topk_runtime.get_or_compute(
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
                # Packed data may legitimately produce an all-padding shard.
                # Keep every indexer parameter in the FSDP backward graph while
                # contributing no supervision on that rank.
                indexer_loss = (
                    indexer_features.q.sum() + indexer_features.k.sum() + indexer_features.weights.sum()
                ) * 0.0
            else:
                valid_query_mask = torch.arange(q_len, device=query_states.device).unsqueeze(0) < valid_query_rows
                attn_q_for_loss = query_states.unsqueeze(0)
                attn_lse_for_loss = softmax_lse.unsqueeze(0)
                # The cuDNN score-recompute MMA requires an attention-head count
                # divisible by 8. Zero Q with +inf LSE contributes exactly zero to
                # its head-summed teacher before the existing L1 normalization.
                pad_heads = (-self.num_attention_heads) % 8
                if pad_heads:
                    attn_q_for_loss = torch.nn.functional.pad(attn_q_for_loss, (0, 0, 0, pad_heads))
                    attn_lse_for_loss = torch.nn.functional.pad(
                        attn_lse_for_loss, (0, pad_heads), value=float("inf")
                    )
                indexer_loss = dsa_indexer_kl_loss(
                    indexer_features.q,
                    indexer_features.k,
                    indexer_features.weights * ((self.index_n_heads * self.index_head_dim) ** -0.5),
                    attn_q_for_loss,
                    key_states.squeeze(1).unsqueeze(0),
                    attn_lse_for_loss,
                    topk_indices.squeeze(1).unsqueeze(0),
                    softmax_scale=self.softmax_scale,
                    row_coefficient=training_cfg.loss_coeff / valid_query_rows,
                    valid_query_mask=valid_query_mask,
                    debug_name=f"layer{self.layer_idx}",
                    debug_interval=training_cfg.debug_interval,
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
