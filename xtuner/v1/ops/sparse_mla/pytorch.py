# Copyright (c) OpenMMLab. All rights reserved.

import torch

from xtuner.v1.data_proto import SequenceContext

from .protocol import SparseMLAOutputs


def torch_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    """Correctness-first PyTorch SparseMLA backend.

    The GLM-5.2 sparse kernel uses ``-1`` to pad invalid top-k slots. Keep the
    fallback fully tensorized so it can run inside GLM fullgraph compile.
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

        safe_indices = group_indices.clamp(min=0).to(torch.long)
        gathered_kv = kv[:, group_idx, :][safe_indices]
        q_group = q[:, group_idx * head_kv : (group_idx + 1) * head_kv, :]

        scores = torch.einsum("shd,skd->shk", q_group.float(), gathered_kv.float())
        scores = scores.mul(scale).masked_fill(~valid[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("shk,skd->shd", probs, gathered_kv[..., :value_dim].float())

        outputs.append(out.to(q.dtype))
        lses.append(torch.logsumexp(scores, dim=-1))

    return SparseMLAOutputs(raw_output=torch.cat(outputs, dim=1), softmax_lse=torch.cat(lses, dim=1))


def torch_dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    _, seq_len, _, _ = q.shape
    scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * (index_head_dim**-0.5)
    scores = torch.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

    packed_mask = _packed_causal_mask(seq_ctx, seq_len, q.device)
    index_scores = index_scores.masked_fill(~packed_mask.unsqueeze(0), float("-inf"))

    topk = min(index_topk, seq_len)
    topk_scores, topk_indices = index_scores.topk(topk, dim=-1)
    topk_indices = topk_indices.masked_fill(topk_scores == -torch.inf, -1)
    return topk_indices.squeeze(0).unsqueeze(1)


def _packed_causal_mask(seq_ctx: SequenceContext, seq_len: int, device: torch.device) -> torch.Tensor:
    cu_seq_lens = seq_ctx.cu_seq_lens_q.to(device)
    token_indices = torch.arange(seq_len, device=device)
    seq_indices = torch.searchsorted(cu_seq_lens, token_indices, right=True) - 1
    row_starts = cu_seq_lens[seq_indices]

    # Keep this vectorized: GLM DSA compiles the indexer inside fullgraph
    # decoder-layer compile, so Python scalar reads from cu_seq_lens would break.
    rows = token_indices[:, None]
    cols = token_indices[None, :]
    return (cols >= row_starts[:, None]) & (cols <= rows)
