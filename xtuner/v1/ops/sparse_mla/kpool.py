# Copyright (c) OpenMMLab. All rights reserved.
"""KPool indexer top-k selection for GLM-5.3-Flash's NoPE DSA (design doc
F5.a).

GLM-5.3-Flash's indexer scores *pools* of ``index_kpool`` (4) consecutive tokens instead of
individual tokens, then expands the selected pools back into raw token ids and appends the
current (still-incomplete) trailing pool as a "tail". The production path below has three
segments:

  A. :func:`build_pools`    -- ``O(S * Di)`` pool construction, pure PyTorch.
  B. the per-query top-k over pools -- reuses
     :func:`xtuner.v1.ops.sparse_mla.tilelang._tilelang_dsa_topk_indices_from_ranges`
     unmodified: that kernel's key-sequence length is fully dynamic (``T.dynamic("seq_len_kv")``
     in ``tl_indexer_fwd_impl``), so passing pool keys in place of token keys and pool-space
     causal ranges in place of token-space ones needs no kernel change (design doc 3.5.2/F5.a).
  C. :func:`expand_pools_and_tail` -- one gather to expand pool ids back to token ids, plus the
     tail append.

Ground truth is HF's ``transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextIndexer``
(``get_pooled_states`` / ``get_visible_tokens`` / ``append_visible_tail``). The one deliberate
difference: HF assumes a single left-padded ``[B, S]`` batch and starts pooling at each row's
first real token; XTuner runs bsz=1 packed multi-document sequences, so pool numbering (and the
causal/tail windows) restarts at *every* document boundary from ``seq_ctx.cu_seq_lens_q``,
not just once per batch row. This keeps pools from straddling documents.

Coordinate system under sequence parallelism: ``SequenceContext.split`` leaves
``cu_seq_lens_q`` **global** and records the local shard's offset in ``shard_start``, so every
token id in this module is global. Concretely that splits the work in two:

- **pool side is whole-sequence** -- ``k``/``gate_scores`` are gathered across the SP mesh
  *before* :func:`build_pools`, because a pool can straddle the shard seam (documents start at
  arbitrary offsets) and no rank can build such a pool alone;
- **query side stays sharded** -- :func:`pool_causal_ranges` and :func:`_visible_tail_tokens`
  map the local queries onto that global grid by adding ``shard_start``, exactly like
  :meth:`SequenceContext.packed_causal_query_ranges`.
"""

import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.comm import gather_for_sequence_parallel

from .tilelang import tilelang_indexer_topk_from_ranges


def kpool_output_width(index_topk: int, index_kpool: int, alignment: int) -> int:
    """Output buffer width: semantic width ``index_topk + index_kpool - 1``, padded up to the
    selected SparseMLA forward's alignment requirement (design doc: FlashMLA -> 512 -> 2560;
    TileLang -> block_I=64 -> 2112). Padding slots are filled with ``-1``.

    The FlashMLA path therefore carries 509 dead slots per query -- ~20% of a
    ``[S, 1, 2560]`` int32 buffer, 168 MB at pack=16384, plus the invalid-slot branches the
    kernel takes on them. Worth revisiting if a 64-aligned FlashMLA path appears.
    """
    width = index_topk + index_kpool - 1
    return -(-width // alignment) * alignment  # ceil_div(width, alignment) * alignment


def _doc_pool_layout(cu_seq_lens_q: Tensor, index_kpool: int) -> tuple[Tensor, Tensor]:
    """Per-document pool bookkeeping shared by :func:`build_pools` and
    :func:`pool_causal_ranges`/:func:`_visible_tail_tokens`.

    Returns:
        tuple[Tensor, Tensor]: ``(num_pools_per_doc, doc_pool_start)``, both ``[D]`` int64,
        where ``doc_pool_start`` is the exclusive cumulative sum (first global pool id of
        each document).
    """
    cu = cu_seq_lens_q.to(torch.int64)
    doc_lens = cu[1:] - cu[:-1]
    num_pools_per_doc = (doc_lens + index_kpool - 1) // index_kpool
    doc_pool_start = torch.cumsum(num_pools_per_doc, dim=0) - num_pools_per_doc
    return num_pools_per_doc, doc_pool_start


def _token_doc_layout(cu_seq_lens_q: Tensor, start: int, count: int, device) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token document assignment for ``count`` tokens starting at
    **global** position ``start``.

    ``start`` is the only coordinate-system conversion point in this module: under sequence
    parallelism ``SequenceContext.split`` keeps ``cu_seq_lens_q`` global and records the local
    shard's offset in ``shard_start``, so query-side callers must pass that offset here exactly
    like :meth:`SequenceContext.packed_causal_query_ranges` does. Pool-side callers pass ``0``
    because pools are built over the whole (gathered) sequence.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``(token_ids, doc_of_token, local_pos)``, all ``[count]``
        int64, with ``token_ids`` in global coordinates.
    """
    cu = cu_seq_lens_q.to(device=device, dtype=torch.int64)
    token_ids = torch.arange(count, device=device, dtype=torch.int64) + start
    doc_of_token = torch.searchsorted(cu, token_ids, right=True) - 1
    local_pos = token_ids - cu[doc_of_token]
    return token_ids, doc_of_token, local_pos


@torch.no_grad()
def build_pool_index(seq_ctx: SequenceContext, seq_len: int, index_kpool: int, device) -> Tensor:
    """Assign every token to a pool, restarting pool numbering at each document
    boundary.

    Returns:
        Tensor: ``pool_index``, shape ``[P, index_kpool]`` int64. ``pool_index[p, slot]`` is
            the global token id at that slot, or ``-1`` if the pool is incomplete there (its
            document ended before filling the pool).
    """
    cu = seq_ctx.cu_seq_lens_q.to(device=device)
    num_pools_per_doc, doc_pool_start = _doc_pool_layout(cu, index_kpool)
    # start=0: pools always cover the whole sequence, never a shard (see build_pools).
    token_ids, doc_of_token, local_pos = _token_doc_layout(cu, 0, seq_len, device)

    pool_id = doc_pool_start[doc_of_token] + local_pos // index_kpool
    slot = local_pos % index_kpool

    num_pools = int(num_pools_per_doc.sum().item())
    pool_index = torch.full((num_pools, index_kpool), -1, device=device, dtype=torch.int64)
    pool_index[pool_id, slot] = token_ids
    return pool_index


@torch.no_grad()
def build_pools(
    k: Tensor,
    gate_scores: Tensor,
    kpool_ape: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_kpool: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build per-pool keys by a softmax(gate + positional-ape)-weighted average
    over each pool's (up to ``index_kpool``) tokens.

    Pool construction is a **whole-sequence** operation, so ``k``/``gate_scores`` must already
    span every token of the packed sequence. Under sequence parallelism that means the caller
    gathers them first: pools restart at document boundaries and documents start at arbitrary
    offsets, so a pool can straddle the shard seam and no rank can build it alone. (Requiring
    ``shard_size % index_kpool == 0`` would *not* help -- e.g. document lengths ``[5, 11]`` with
    ``sp_size=2`` still puts pool ``[5,6,7,8]`` across the seam.)

    Args:
        k (Tensor): Indexer key features for the whole sequence, shape ``[S, index_head_dim]``.
        gate_scores (Tensor): ``index_kpool_compress_gate(hidden_states)``, whole sequence,
            shape ``[S, index_head_dim]``.
        kpool_ape (Tensor): Per-slot positional bias, shape ``[index_kpool, index_head_dim]``.
        seq_ctx (SequenceContext): Packed-sequence metadata (document boundaries).
        index_kpool (int): Pool size.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - ``pool_key`` ``[P, index_head_dim]``, same dtype as ``k``.
            - ``pool_index`` ``[P, index_kpool]`` int32 (global token ids, ``-1`` if invalid).
            - ``pool_complete`` ``[P]`` bool (all ``index_kpool`` slots valid).
    """
    seq_len, device = k.shape[0], k.device
    global_len = int(seq_ctx.cu_seq_lens_q[-1].item())
    if seq_len != global_len:
        raise RuntimeError(
            f"build_pools needs key features for the whole sequence ({global_len} tokens) but got "
            f"{seq_len}; gather them across the sequence-parallel mesh before calling."
        )
    pool_index = build_pool_index(seq_ctx, seq_len, index_kpool, device)

    valid = pool_index >= 0
    safe = pool_index.clamp(min=0)
    grouped_k = k[safe]
    grouped_gate = gate_scores[safe]

    logits = grouped_gate.float() + kpool_ape.float().unsqueeze(0)
    logits = logits.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    # A fully-invalid pool never occurs in practice (every document has >= 1 token, so its
    # first pool always has >= 1 valid slot), but nan_to_num matches HF's defensive guard.
    probs = torch.nan_to_num(logits.softmax(dim=1)).to(grouped_k.dtype)
    pool_key = (probs * grouped_k).sum(dim=1)
    pool_complete = valid.all(dim=-1)

    return pool_key, pool_index.to(torch.int32), pool_complete


def pool_causal_ranges(
    seq_ctx: SequenceContext,
    pool_index: Tensor,
    query_len: int,
    device,
) -> tuple[Tensor, Tensor]:
    """Per-query causal visibility window in **pool-index space**.

    A pool is selectable by a query iff it is complete and its (necessarily last, since pools
    are built in token order within a document) token is causally visible. Because pools are
    assigned sequentially per document with no gaps, the set of visible+complete pools for a
    query is always a prefix of that document's pool range -- i.e. a contiguous
    ``[doc_pool_start, doc_pool_start + num_complete_visible)`` interval, with
    ``num_complete_visible = (local_pos + 1) // index_kpool``. Completeness is therefore implied
    by the range itself and needs no separate mask.

    Args:
        seq_ctx (SequenceContext): Packed-sequence metadata; ``shard_start`` locates this rank's
            queries inside the global sequence.
        pool_index (Tensor): From :func:`build_pools`, shape ``[P, index_kpool]``, covering the
            **whole** sequence (pool ids returned here index into it).
        query_len (int): Number of local queries on this rank.
        device: Device to build the range tensors on.

    Returns:
        tuple[Tensor, Tensor]: ``(starts, ends)``, each ``[query_len]`` int32, half-open
        ``[start, end)`` ranges in global pool-index space -- the same contract
        :meth:`SequenceContext.packed_causal_query_ranges` uses in token space.
    """
    index_kpool = pool_index.shape[1]
    cu = seq_ctx.cu_seq_lens_q.to(device=device)
    _, doc_pool_start = _doc_pool_layout(cu, index_kpool)
    _, doc_of_query, local_pos = _token_doc_layout(cu, seq_ctx.shard_start, query_len, device)

    starts = doc_pool_start[doc_of_query]
    ends = starts + (local_pos + 1) // index_kpool
    return starts.to(torch.int32), ends.to(torch.int32)


def _visible_tail_tokens(seq_ctx: SequenceContext, query_len: int, index_kpool: int, device) -> Tensor:
    """The <= ``index_kpool - 1`` most recent tokens in the query's document that haven't yet
    formed a complete pool, most-recent-last, ``-1``-padded. Shape ``[query_len, index_kpool - 1]``
    int32. Mirrors HF's ``Glm5NextTextIndexer.append_visible_tail``."""
    max_tail_width = index_kpool - 1
    if max_tail_width == 0:
        return torch.empty((query_len, 0), device=device, dtype=torch.int32)

    cu = seq_ctx.cu_seq_lens_q.to(device=device)
    token_ids, _, local_pos = _token_doc_layout(cu, seq_ctx.shard_start, query_len, device)
    tail_count = (local_pos + 1) % index_kpool

    tail_offsets = torch.arange(max_tail_width, device=device, dtype=torch.int64)
    tail_start = token_ids - tail_count + 1
    tail_ids = tail_start.unsqueeze(-1) + tail_offsets.unsqueeze(0)
    tail_valid = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(-1)
    return tail_ids.masked_fill(~tail_valid, -1).to(torch.int32)


@torch.no_grad()
def expand_pools_and_tail(
    selected_pool_ids: Tensor,
    pool_index: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_topk: int,
    index_kpool: int,
    always_select_tail: bool,
    alignment: int,
) -> Tensor:
    """Expand selected pool ids back into raw token ids and append the visible
    tail.

    Args:
        selected_pool_ids (Tensor): Top-k pool ids from the indexer kernel, shape
            ``[S, 1, index_topk // index_kpool]``, ``-1`` for unfilled slots.
        pool_index (Tensor): From :func:`build_pools`, shape ``[P, index_kpool]``.
        seq_ctx (SequenceContext): Packed-sequence metadata.
        index_topk (int): Semantic token budget (e.g. 2048).
        index_kpool (int): Pool size.
        always_select_tail (bool): Whether to append the incomplete trailing pool.
        alignment (int): Output width alignment, from the chosen SparseMLA backend.

    Returns:
        Tensor: ``[S, 1, kpool_output_width(...)]`` int32 token ids, ``-1``-padded.
    """
    seq_len, device = selected_pool_ids.shape[0], selected_pool_ids.device
    width = kpool_output_width(index_topk, index_kpool, alignment)
    out = selected_pool_ids.new_full((seq_len, width), -1, dtype=torch.int32)

    selected = selected_pool_ids.squeeze(1).to(torch.int64)  # [S, index_topk // index_kpool]
    valid_sel = selected >= 0
    safe_sel = selected.clamp(min=0)
    expanded = pool_index[safe_sel].masked_fill(~valid_sel.unsqueeze(-1), -1)  # [S, select_k, kpool]
    expanded = expanded.reshape(seq_len, -1)  # [S, index_topk] (select_k * kpool == index_topk)
    out[:, : expanded.shape[-1]] = expanded

    if always_select_tail:
        tail = _visible_tail_tokens(seq_ctx, seq_len, index_kpool, device)
        out[:, index_topk : index_topk + index_kpool - 1] = tail

    return out.unsqueeze(1)


@torch.no_grad()
def kpool_topk_indices(
    q: Tensor,
    k: Tensor,
    gate_scores: Tensor,
    weights: Tensor,
    kpool_ape: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
    index_kpool: int = 4,
    always_select_tail: bool = True,
    alignment: int = 512,
    query_chunk_size: int | None = None,
) -> Tensor:
    """Production KPool indexer top-k: build pools (A), score+top-k over pools by reusing the
    existing DSA indexer kernel unmodified (B), expand back to token ids + tail (C).

    Args:
        q (Tensor): Index query, shape ``[1, S, index_n_heads, index_head_dim]``, bf16.
        k (Tensor): Index key, shape ``[1, S, index_head_dim]``, bf16.
        gate_scores (Tensor): ``index_kpool_compress_gate(hidden_states)``, shape
            ``[1, S, index_head_dim]``.
        weights (Tensor): ``weights_proj(hidden_states)``, shape ``[1, S, index_n_heads]``,
            **unscaled** raw projection output (this function applies the
            ``index_n_heads**-0.5 * index_head_dim**-0.5`` scaling).
        kpool_ape (Tensor): ``index_kpool_compress_ape``, shape ``[index_kpool, index_head_dim]``.
        seq_ctx (SequenceContext): Packed-sequence and SP metadata.
        index_head_dim (int): Indexer head dimension (scaling only; kernel infers dims from q/k).
        index_topk (int): Semantic token budget (e.g. 2048); must be divisible by ``index_kpool``.
        index_kpool (int): Pool size.
        always_select_tail (bool): Whether to append the incomplete trailing pool.
        alignment (int): Output width alignment (FlashMLA: 512; TileLang: 64).

    Returns:
        Tensor: ``[S, 1, kpool_output_width(index_topk, index_kpool, alignment)]`` int32.
    """
    assert index_topk % index_kpool == 0
    # SP: queries stay sharded; the key side gathers *before* pooling, because a pool can
    # straddle the shard seam (see build_pools). This costs 2 x [S, index_head_dim] per layer
    # instead of the [P, index_head_dim] a post-pooling gather would move -- the price of
    # pooling being a whole-sequence operation.
    sp_mesh = seq_ctx.sequence_parallel_mesh
    k = gather_for_sequence_parallel(k.squeeze(0), dim=0, sp_mesh=sp_mesh)
    gate_scores = gather_for_sequence_parallel(gate_scores.squeeze(0), dim=0, sp_mesh=sp_mesh)

    pool_key, pool_index, _ = build_pools(k, gate_scores, kpool_ape, seq_ctx, index_kpool=index_kpool)

    starts, ends = pool_causal_ranges(seq_ctx, pool_index, q.shape[1], q.device)

    q_local = q.squeeze(0).contiguous()
    index_n_heads = q_local.shape[1]
    scaled_weights = (weights.squeeze(0) * (index_n_heads**-0.5) * (index_head_dim**-0.5)).contiguous()

    select_k = index_topk // index_kpool
    selected_pool_ids = tilelang_indexer_topk_from_ranges(
        q_local,
        pool_key.contiguous(),
        scaled_weights,
        starts,
        ends,
        select_k,
        query_chunk_size=query_chunk_size,
    )

    return expand_pools_and_tail(
        selected_pool_ids,
        pool_index,
        seq_ctx,
        index_topk=index_topk,
        index_kpool=index_kpool,
        always_select_tail=always_select_tail,
        alignment=alignment,
    )


@torch.no_grad()
def torch_kpool_topk_indices(
    q: Tensor,
    k: Tensor,
    gate_scores: Tensor,
    weights: Tensor,
    kpool_ape: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
    index_kpool: int = 4,
    always_select_tail: bool = True,
    alignment: int = 512,
    query_chunk_size: int | None = None,
) -> Tensor:
    """Reference-only KPool top-k (correctness anchor for tests / small shapes
    / CPU).

    **Not for production**: the ``einsum`` materializes a dense ``[S, index_n_heads, P]`` fp32
    score tensor -- 8.6 GB at ``S=16384``. Uses the same pool-space causal-range logic as the
    production path but computes scores densely instead of via the DSA indexer kernel.

    ``query_chunk_size`` is accepted for signature parity with the production path (both are
    reached through the same protocol) and ignored: this path exists to be obviously correct,
    and bounding its memory is what the production path is for.
    """
    del query_chunk_size
    assert index_topk % index_kpool == 0
    sp_mesh = seq_ctx.sequence_parallel_mesh
    k = gather_for_sequence_parallel(k.squeeze(0), dim=0, sp_mesh=sp_mesh)
    gate_scores = gather_for_sequence_parallel(gate_scores.squeeze(0), dim=0, sp_mesh=sp_mesh)

    pool_key, pool_index, _ = build_pools(k, gate_scores, kpool_ape, seq_ctx, index_kpool=index_kpool)

    q_local = q.squeeze(0)  # [S, Ni, Di]
    starts, ends = pool_causal_ranges(seq_ctx, pool_index, q_local.shape[0], q_local.device)

    # scores: [S, Ni, P]
    scores = torch.relu(torch.einsum("shd,pd->shp", q_local.float(), pool_key.float()) * (index_head_dim**-0.5))
    index_n_heads = q_local.shape[1]
    scaled_weights = weights.squeeze(0).float() * (index_n_heads**-0.5)  # [S, Ni]
    index_scores = torch.einsum("shp,sh->sp", scores, scaled_weights)  # [S, P]

    num_pools = pool_key.shape[0]
    pool_ids = torch.arange(num_pools, device=q_local.device).unsqueeze(0)  # [1, P]
    visible = (pool_ids >= starts.unsqueeze(-1)) & (pool_ids < ends.unsqueeze(-1))  # [S, P]
    index_scores = index_scores.masked_fill(~visible, float("-inf"))

    select_k = min(index_topk // index_kpool, num_pools)
    selected = index_scores.topk(select_k, dim=-1)
    selected_pool_ids = selected.indices.masked_fill(selected.values == float("-inf"), -1)
    selected_pool_ids = selected_pool_ids.to(torch.int32).unsqueeze(1)  # [S, 1, select_k]

    return expand_pools_and_tail(
        selected_pool_ids,
        pool_index,
        seq_ctx,
        index_topk=index_topk,
        index_kpool=index_kpool,
        always_select_tail=always_select_tail,
        alignment=alignment,
    )
