# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash's NoPE DSA (design doc F5): absorbed MLA with no rotary tail,
using the KPool indexer instead of GLM-5.2's per-token indexer.

``qk_rope_head_dim=0`` makes the base :class:`MultiLatentAttention` naturally settle on
``q_head_dim = qk_nope_head_dim = 256`` and ``softmax_scale = 256 ** -0.5`` -- no NoPE-specific
override needed there. What *is* NoPE-specific: no rotary embeddings are applied anywhere in
this layer (``position_embeddings`` is accepted and ignored, matching KDA's NoPE convention),
and there is no cross-layer ``dsa_topk_ids`` sharing -- the published config's ``indexer_types``
are all ``"full"`` (design doc 3.4/F5), so every layer runs its own indexer independently,
unlike GLM-5.2's ``dsa_topk_source_layer``/``GLM52MTPBlock`` machinery.
"""

from typing import Literal, cast

import torch
from pydantic import ConfigDict, model_validator
from torch import nn
from torch.nn import functional as F
from typing_extensions import overload

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.attention.attn_outputs import AttnOutputs
from xtuner.v1.module.attention.mla import MLAConfig, MultiLatentAttention
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rms_norm import LayerNorm
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.comm import gather_for_sequence_parallel
from xtuner.v1.ops.sparse_mla import (
    KPoolIndexerBackend,
    SparseMLABackend,
    SparseMLAProtocol,
    get_kpool_topk_indices,
    get_sparse_mla,
)
from xtuner.v1.utils.dtensor import materialize_full
from xtuner.v1.utils.init_weight import init_params


# Output-width alignment per SparseMLA forward backend (design doc F5.a/F5.b): the padded
# topk-index buffer width must match what that backend's kernel expects.
_SPARSE_MLA_ALIGNMENT: dict[str, int] = {
    "torch": 1,
    "flash_mla_cudnn": 512,
    "tilelang": 64,
}


class KPoolIndexer(nn.Module):
    """KPool DSA indexer.

    Parameter names mirror the published checkpoint's
    ``self_attn.indexer.*`` keys: ``wq_b`` / ``wk`` / ``k_norm`` / ``weights_proj`` /
    ``index_kpool_compress_ape`` / ``index_kpool_compress_gate``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        index_kpool: int,
        index_kpool_always_select_tail: bool,
        indexer_backend: KPoolIndexerBackend,
        alignment: int,
        topk_query_chunk_size: int | None = None,
    ):
        super().__init__()
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.always_select_tail = index_kpool_always_select_tail
        self.indexer_backend = indexer_backend
        self.alignment = alignment
        self.topk_query_chunk_size = topk_query_chunk_size
        # Resolved once here (not per forward call), mirroring GLM-5.2's
        # get_dsa_topk_indices(indexer_backend) precomputation.
        self._topk_indices_fn = get_kpool_topk_indices(indexer_backend)

        self.wq_b = build_linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = build_linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = LayerNorm(index_head_dim, eps=1e-6)
        self.weights_proj = build_linear(hidden_size, index_n_heads, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(index_kpool, index_head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(index_head_dim, hidden_size))

    def init_weights(self) -> None:
        """Initialize the parameters ``default_init_weights`` cannot reach by
        name.

        Both are zero-initialized, matching ``__init__``: a zero ``compress_gate`` makes the
        intra-pool softmax uniform, which is the neutral starting point for an indexer that is
        frozen by default and otherwise loaded from a checkpoint.
        """
        init_params(self.index_kpool_compress_ape, torch.nn.init.zeros_)
        init_params(self.index_kpool_compress_gate, torch.nn.init.zeros_)

    def forward(self, hidden_states: torch.Tensor, q_resid: torch.Tensor, seq_ctx: SequenceContext) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        gate_weight = materialize_full(self.index_kpool_compress_gate, name="indexer.index_kpool_compress_gate")
        kpool_ape = materialize_full(self.index_kpool_compress_ape, name="indexer.index_kpool_compress_ape")

        q = self.wq_b(q_resid).view(bsz, seq_len, self.index_n_heads, self.index_head_dim)
        k = self.k_norm(self.wk(hidden_states))
        gate_scores = F.linear(hidden_states, gate_weight)
        weights = self.weights_proj(hidden_states).float()

        topk_ids = self._topk_indices_fn(
            q,
            k,
            gate_scores,
            weights,
            kpool_ape,
            seq_ctx,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            index_kpool=self.index_kpool,
            always_select_tail=self.always_select_tail,
            alignment=self.alignment,
            query_chunk_size=self.topk_query_chunk_size,
        )
        return topk_ids.to(torch.int32).contiguous()


class NoPEDSAMLAConfig(MLAConfig):
    """NoPE + KPool DSA config.

    ``qk_rope_head_dim=0`` makes the base class automatically derive the correct
    ``q_head_dim = 256`` and ``softmax_scale = 256 ** -0.5``; ``kv_a_proj_with_mqa`` outputs
    exactly ``kv_lora_rank`` (no rope tail), so the base class needs no NoPE-specific override.
    """

    # Backends are commonly selected from the environment *after* construction (see
    # examples/v1/config/sft_glm53.py); without this the model validator below would not re-run
    # and an unsupported choice would surface far away, inside a kernel.
    model_config = ConfigDict(validate_assignment=True)

    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    indexer_types: list[str] | None = None
    # Production default: FlashMLA fwd (native 512 head_dim) + cuDNN bwd (no dim hardcoding) --
    # equivalent to Automodel's cudnn_sparse_attention; doesn't need the TileLang tail_dim=0 fix
    # (design doc 3.5.2). "tilelang" needs that kernel fix and is not implemented yet (F5.b).
    sparse_mla_backend: Literal["torch", "flash_mla_cudnn", "tilelang"] = "flash_mla_cudnn"
    # Independent of `sparse_mla_backend` above, which is `flash_mla_cudnn` here and is not
    # even a valid indexer backend -- inheriting it would never have worked.
    indexer_backend: KPoolIndexerBackend = "tilelang"
    # Bounds the transient [query_chunk, num_pools] logits tile the selector materializes.
    # ``None`` keeps a single launch; set it for long context, where that tile is the
    # indexer's memory peak. Pool-space keys already make it index_kpool times smaller than
    # the per-token DSA indexer's.
    indexer_topk_query_chunk_size: int | None = None
    freeze_dsa_indexer: bool = True

    @model_validator(mode="after")
    def _check(self) -> "NoPEDSAMLAConfig":
        if self.qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3-Flash's main attention is NoPE: qk_rope_head_dim must be 0.")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")
        if self.indexer_types is not None and "shared" in self.indexer_types:
            raise ValueError(
                "GLM-5.3-Flash's published config has indexer_types all 'full' -- IndexShare "
                "('shared') is not supported."
            )
        if self.sparse_mla_backend == "tilelang":
            raise NotImplementedError(
                "sparse_mla_backend='tilelang' needs the tail_dim=0 TileLang kernel fix (design "
                "doc 3.5.2), which is not implemented yet. Use 'flash_mla_cudnn' (production) or "
                "'torch' (reference)."
            )
        return self

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "NoPEDSAMultiLatentAttention":
        del layer_type, rope_scaling_cfg, generate_config  # NoPE: no rotary, unused.
        return NoPEDSAMultiLatentAttention(
            **self.model_dump(), hidden_size=hidden_size, layer_idx=layer_idx, float8_cfg=float8_cfg
        )


class NoPEDSAMultiLatentAttention(MultiLatentAttention):
    """Absorbed NoPE-MLA: query/key are both 512-dim latents, ``softmax_scale = 256 ** -0.5``.

    Data flow::

        hidden (1,S,4096)
           |- q_a_proj -> q_a_layernorm -> q_resid (1,S,1536)
           |     |- q_b_proj -> q (1,64,S,256) --absorb(w_kc)--> query (S,64,512)
           |     `- KPoolIndexer(hidden, q_resid) ------------> topk (S,1,width)
           |- kv_a_proj_with_mqa -> kv_a_layernorm -> key (S,1,512) --SP gather--> (S_g,1,512)
           `- SparseMLA(query, key, topk) -> (S,64,512) --absorb^-1(w_vc)--> (1,S,16384) -> o_proj
    """

    def __init__(
        self,
        *,
        index_topk: int,
        index_head_dim: int,
        index_n_heads: int,
        index_kpool: int,
        index_kpool_always_select_tail: bool,
        indexer_types: list[str] | None,
        sparse_mla_backend: SparseMLABackend,
        indexer_backend: KPoolIndexerBackend,
        indexer_topk_query_chunk_size: int | None,
        freeze_dsa_indexer: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.q_lora_rank is None:
            raise ValueError("NoPE DSA MLA requires q_lora_rank because the indexer consumes q_a_layernorm output.")

        # Absorbed MLA reads kv_b_proj.weight directly and folds it into w_kc / w_vc with a
        # view + split (see _absorb_weights). Under FSDP's FP8 runtime that weight becomes a
        # Float8Tensor, which does not implement aten.split_with_sizes, so the forward dies at
        # the first layer. Keep this one projection in high precision until there is a fused FP8
        # absorbed-MLA path; GLM-5.2's DSA excludes the same projection, for the analogous
        # reason (its view is not 128-aligned).
        if self.float8_cfg is not None:
            self.kv_b_proj = build_linear(
                self.kv_lora_rank,
                self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                float8_cfg=None,
            )

        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.indexer_types = indexer_types
        self.sparse_mla_backend = sparse_mla_backend
        self.freeze_dsa_indexer = freeze_dsa_indexer
        self.sparse_mla_func: SparseMLAProtocol = get_sparse_mla(sparse_mla_backend)
        self.alignment = _SPARSE_MLA_ALIGNMENT[sparse_mla_backend]

        # No cute_dsl backend here, unlike GLM-5.2's per-token indexer: that kernel is
        # specialized to topk in {1024, 2048} (radix bin width, candidate tiles and compaction
        # merges are all tuned per value), while KPool selects over pools and asks for
        # index_topk // index_kpool = 512. Supporting it means adding and tuning a third
        # specialization, not wiring up a backend -- see xtuner/v1/ops/sparse_mla/
        # cute_dsl_indexer_topk.py. TileLang covers the production path meanwhile.
        self.indexer = KPoolIndexer(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            index_kpool=index_kpool,
            index_kpool_always_select_tail=index_kpool_always_select_tail,
            indexer_backend=indexer_backend,
            alignment=self.alignment,
            topk_query_chunk_size=indexer_topk_query_chunk_size,
        )
        if freeze_dsa_indexer:
            self.indexer.requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        seq_ctx: SequenceContext | None = None,
        **kwargs,
    ) -> AttnOutputs:
        # NoPE: no rotary embeddings anywhere in this layer.
        del position_embeddings, kwargs
        assert seq_ctx is not None
        assert hidden_states.size(0) == 1, "packed training path expects batch size 1"
        assert self.q_lora_rank is not None
        bsz, seq_len, _ = hidden_states.shape

        w_kc, w_vc = self._absorb_weights()

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self._absorbed_query(q_resid, w_kc, bsz, seq_len)
        key_states = self._gathered_key(hidden_states, seq_ctx)

        # Branches on freeze_dsa_indexer like GLM-5.2's per-token indexer, ahead of a future
        # differentiable indexer output; today's kpool_topk_indices/torch_kpool_topk_indices
        # only ever return an int32 index tensor, so this doesn't yet change what's trainable.
        if self.freeze_dsa_indexer:
            with torch.no_grad():
                topk_ids = self.indexer(hidden_states, q_resid, seq_ctx)
        else:
            topk_ids = self.indexer(hidden_states, q_resid, seq_ctx)

        out = self.sparse_mla_func(query_states, key_states, topk_ids, self.softmax_scale, value_dim=self.kv_lora_rank)
        raw_output = torch.einsum("shm,hdm->shd", out.raw_output, w_vc)
        raw_output = raw_output.reshape(bsz, seq_len, self.num_attention_heads * self.v_head_dim).contiguous()
        projected_output = self.o_proj(raw_output)

        return {
            "raw_output": raw_output,
            "projected_output": projected_output,
            "softmax_lse": out.softmax_lse,
        }

    def _absorb_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``kv_b_proj`` into the key and value halves the absorbed form
        contracts with.

        Absorbed MLA never materializes K/V: the key half folds into the query and the value half unfolds the attention
        output, so this one weight plays both roles and has to be reshaped per head first.
        """
        wkv_b = materialize_full(cast(torch.Tensor, self.kv_b_proj.weight), name="kv_b_proj.weight")
        wkv_b = wkv_b.view(self.num_attention_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_kc, w_vc = torch.split(wkv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=1)
        return w_kc, w_vc

    def _absorbed_query(self, q_resid: torch.Tensor, w_kc: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        """Project and absorb the query into latent space: ``[S, N, kv_lora_rank]``."""
        q_nope = self.q_b_proj(q_resid).view(bsz, seq_len, self.num_attention_heads, self.qk_nope_head_dim)
        q_nope = q_nope.transpose(1, 2)  # [1, N, S, Dn]
        return torch.einsum("bhsd,hdm->bhsm", q_nope, w_kc).squeeze(0).transpose(0, 1).contiguous()

    def _gathered_key(self, hidden_states: torch.Tensor, seq_ctx: SequenceContext) -> torch.Tensor:
        """Compressed KV for the whole sequence: ``[S, 1, kv_lora_rank]``.

        DSA has a single compressed-KV group, so gathering this (rather than an all-to-all on
        query/output) is the cheaper SP path -- the same choice KDA and GLM-5.2 make.
        """
        kv_compressed = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))  # [1, S, Rkv]
        key_states = kv_compressed.squeeze(0).unsqueeze(1).contiguous()
        return gather_for_sequence_parallel(key_states, dim=0, sp_mesh=seq_ctx.sequence_parallel_mesh)

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        seq_ctx: SequenceContext | None = None,
    ) -> AttnOutputs: ...

    __call__ = nn.Module.__call__
