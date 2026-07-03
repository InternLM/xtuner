# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.rope import RopeScalingConfig

from ..linear import build_linear
from .attn_outputs import AttnOutputs
from .mla import MLAConfig, MultiLatentAttention, mla_apply_rotary_pos_emb


def torch_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correctness-first PyTorch SparseMLA backend.

    The GLM-5.2 sparse kernel uses ``-1`` to pad invalid top-k slots. Short smoke
    tests can have many invalid slots, so trim columns that are invalid for the
    whole microbatch before the gather to keep the fallback cheap.
    """

    _, heads, dim_plus_tail_dim = q.shape
    _, kv_group, _ = kv.shape
    head_kv = heads // kv_group
    value_dim = value_dim if value_dim is not None else dim_plus_tail_dim
    scale = float(scaling) if scaling is not None else dim_plus_tail_dim**-0.5

    outputs = []
    lses = []
    for group_idx in range(kv_group):
        group_indices = indices[:, group_idx, :]
        valid = group_indices != -1

        valid_cols = valid.any(dim=0)
        if valid_cols.any().item():
            last_col = int(valid_cols.nonzero()[-1].item()) + 1
            group_indices = group_indices[:, :last_col]
            valid = valid[:, :last_col]

        safe_indices = group_indices.clamp(min=0).to(torch.long)
        gathered_kv = kv[:, group_idx, :][safe_indices]
        q_group = q[:, group_idx * head_kv : (group_idx + 1) * head_kv, :]

        scores = torch.einsum("shd,skd->shk", q_group.float(), gathered_kv.float())
        scores = scores.mul(scale).masked_fill(~valid[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("shk,skd->shd", probs, gathered_kv[..., :value_dim].float())

        outputs.append(out.to(q.dtype))
        lses.append(torch.logsumexp(scores, dim=-1))

    return torch.cat(outputs, dim=1), torch.cat(lses, dim=1)


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
    ):
        super().__init__()
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
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
        scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * (self.index_head_dim**-0.5)
        scores = torch.relu(scores)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        packed_mask = _packed_causal_mask(seq_ctx, seq_len, hidden_states.device)
        index_scores = index_scores.masked_fill(~packed_mask.unsqueeze(0), float("-inf"))

        topk = min(self.index_topk, seq_len)
        topk_scores, topk_indices = index_scores.topk(topk, dim=-1)
        topk_indices = topk_indices.masked_fill(topk_scores == -torch.inf, -1)
        return topk_indices.squeeze(0).unsqueeze(1)


def _packed_causal_mask(seq_ctx: SequenceContext, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    cu_seq_lens = seq_ctx.cu_seq_lens_q.to(device)
    for seq_idx in range(cu_seq_lens.numel() - 1):
        start = int(cu_seq_lens[seq_idx].item())
        end = int(cu_seq_lens[seq_idx + 1].item())
        rows = torch.arange(start, end, device=device)
        cols = torch.arange(start, end, device=device)
        mask[start:end, start:end] = cols.unsqueeze(0) <= rows.unsqueeze(1)
    return mask


class DSAMLAConfig(MLAConfig):
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 0
    indexer_rope_interleave: bool = True
    indexer_types: list[str] | None = None

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "DSAMultiLatentAttention":
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk_freq = index_topk_freq
        self.index_skip_topk_offset = index_skip_topk_offset
        self.indexer_rope_interleave = indexer_rope_interleave
        self.indexer_types = indexer_types

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
        )

    def _is_skip_topk_layer(self) -> bool:
        if self.indexer_types is not None and self.layer_idx < len(self.indexer_types):
            return self.indexer_types[self.layer_idx] == "shared"
        if self.index_topk_freq <= 1:
            return False
        return is_dsa_skip_topk_layer(self.layer_idx, self.index_skip_topk_offset, self.index_topk_freq)

    def _source_compute_layer(self) -> int:
        if self.indexer_types is not None:
            for idx in range(self.layer_idx, -1, -1):
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
        if seq_ctx.dsa_topk_indices is None:
            seq_ctx.dsa_topk_indices = {}

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
        # TODO: refactor below as MHA's `self.attn_impl_func: Callable[..., AttnOpOutputs] = get_attn_impl_fn(attn_impl)`
        raw_output, softmax_lse = torch_sparse_mla(
            query_states,
            key_states,
            topk_indices,
            self.softmax_scale,
            value_dim=self.kv_lora_rank,
        )
        raw_output = torch.einsum("shm,hdm->shd", raw_output, w_vc)
        raw_output = raw_output.reshape(bsz, q_len, self.num_attention_heads * self.v_head_dim).contiguous()
        projected_output = self.o_proj(raw_output)

        return {
            "raw_output": raw_output,
            "projected_output": projected_output,
            "softmax_lse": softmax_lse,
        }
