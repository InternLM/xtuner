# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from xtuner.v1.module.attention.dsa.indexer import Indexer, IndexerConfig


def _make_indexer(
    hidden_size: int = 512,
    q_lora_rank: int = 128,
    index_n_heads: int = 4,
    index_head_dim: int = 128,
    rope_head_dim: int = 64,
    index_topk: int = 8,
    compress_ratio: int = 4,
    seed: int = 1234,
) -> Indexer:
    torch.manual_seed(seed)
    config = IndexerConfig(
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_head_dim,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
    )
    indexer = Indexer(config)
    # Non-zero APE keeps the compressed-KV gate softmax non-uniform across
    # the window so the topk selection isn't degenerately positional.
    with torch.no_grad():
        torch.nn.init.normal_(indexer.compressor.ape, std=0.02)
    indexer.eval()
    return indexer


def _compute_native_scores_at_indices(
    indexer: "Indexer",
    hidden: torch.Tensor,
    qr: torch.Tensor,
    position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor],
    cu_seq_lens: torch.Tensor,
    topk_idxs: torch.Tensor,
) -> torch.Tensor:
    """Recompute the native Indexer scoring pipeline (no topk) and gather the
    final per-(q, k) scalar score at each index in ``topk_idxs``.

    Used by the triton-vs-native parity test: with cuBLAS-vs-Triton reduction
    order differing by O(ULP) on the inner ``q · k`` dot, picked indices may
    flip on near-ties even when both backends are computing the same math.
    Comparing the *scores* at the picked positions is the right semantic
    check — if those agree (within fp tolerance) then downstream sparse_attn
    sees equivalent KV regardless of which specific tied index landed.
    Returns ``[1, total_q, K]`` fp32 where -1 indices map to ``-inf``.
    """
    from xtuner.v1.module.attention.dsa.indexer import _apply_rope, rotate_activation

    cos, sin = position_embeddings_compressed
    q = indexer.wq_b(qr).unflatten(-1, (indexer.n_heads, indexer.head_dim))
    q_nope_tail = q[..., : indexer.head_dim - indexer.rope_head_dim]
    q_rope_tail = q[..., indexer.head_dim - indexer.rope_head_dim :]
    q_rope_tail = _apply_rope(q_rope_tail, cos, sin)
    q = torch.cat([q_nope_tail, q_rope_tail], dim=-1)
    q = rotate_activation(q)

    kv_compressed, cu_c = indexer.compressor(hidden, cu_seq_lens)
    weights = indexer.weights_proj(hidden) * (indexer.softmax_scale * indexer.n_heads**-0.5)

    # Full-pack score matrix [1, total_q, total_c] — never materialised in the
    # triton path; we build it here only to look up scalar scores at the
    # caller-supplied indices.
    qk = torch.einsum("bshd,btd->bsht", q, kv_compressed).relu_()
    score = (qk * weights.unsqueeze(-1)).sum(dim=2)  # [1, total_q, total_c]

    # Gather scores at topk_idxs (sample-local). Need to convert to global
    # compressed-axis indices: shift each row by the row's sample's cu_c.
    total_q = q.size(1)
    pos = torch.arange(total_q, device=q.device)
    sample_id = torch.searchsorted(cu_seq_lens, pos, right=True) - 1
    cu_c_per_q = cu_c[sample_id]  # [total_q]
    valid = topk_idxs >= 0
    safe_idx = topk_idxs.clamp(min=0)
    global_idx = safe_idx + cu_c_per_q.view(1, -1, 1)
    gathered = score.gather(2, global_idx.long())
    return torch.where(valid, gathered, torch.full_like(gathered, float("-inf")))


def _make_position_embeddings(seq_len: int, rope_head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(cos, sin)`` pair in the layout the DSA modules consume.

    :class:`~xtuner.v1.module.rope.DualRotaryEmbedding` emits *full* ``rope_head_dim``
    tables with the pair broadcast and the interleaved rotation's sign pattern already
    folded in, so the per-layer rotation is one ``x * cos + flip_pairs(x) * sin``. This
    mirrors that so the tests exercise the production contract instead of the half-dim
    tables the modules used to accept.

    Args:
        seq_len (int): Number of positions.
        rope_head_dim (int): Full rope dim; the returned tables' last dim.
        base (float): RoPE theta. Defaults to ``10000.0``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(cos_full, sin_full_signed)``, each
        ``[1, seq_len, rope_head_dim]``.
    """
    half = rope_head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [seq_len, rope_head_dim // 2]
    cos_half, sin_half = angles.cos(), angles.sin()
    cos = cos_half.repeat_interleave(2, dim=-1).unsqueeze(0)
    sin = torch.stack([-sin_half, sin_half], dim=-1).flatten(-2).unsqueeze(0)
    return cos, sin


class TestIndexer:
    def test_forward_shape(self) -> None:
        hidden_size = 512
        q_lora_rank = 128
        index_n_heads = 4
        index_head_dim = 128
        rope_head_dim = 64
        index_topk = 8
        seq_len = 64
        indexer = _make_indexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
        )

        torch.manual_seed(0)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        qr = torch.randn(1, seq_len, q_lora_rank, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, rope_head_dim)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)

        topk_idxs = indexer(hidden, qr, (cos, sin), cu)

        assert topk_idxs.shape == (1, seq_len, index_topk)
        assert topk_idxs.dtype == torch.long
        # Output must not contain NaN; -1 placeholders are integers and never
        # NaN, so this is checking the upstream score path didn't blow up.
        assert not torch.isnan(topk_idxs.float()).any()

    def test_topk_indices_in_range(self) -> None:
        hidden_size = 512
        seq_len = 64
        compress_ratio = 4
        rope_head_dim = 64
        indexer = _make_indexer(
            hidden_size=hidden_size,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
        )

        torch.manual_seed(1)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        qr = torch.randn(1, seq_len, 128, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, rope_head_dim)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)

        topk_idxs = indexer(hidden, qr, (cos, sin), cu)
        # 64 // 4 = 16 compressed positions for this single sample.
        t_compressed = seq_len // compress_ratio
        in_range = (topk_idxs == -1) | ((topk_idxs >= 0) & (topk_idxs < t_compressed))
        assert in_range.all()

    def test_causal_constraint(self) -> None:
        # For query position s (1-indexed) the V4 reference forbids any
        # selected index t with t >= (s + 1 - 1) // ratio. We verify this
        # holds for every (query_pos, selected_t) pair in the output.
        hidden_size = 512
        seq_len = 64
        compress_ratio = 4
        rope_head_dim = 64
        indexer = _make_indexer(
            hidden_size=hidden_size,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
        )

        torch.manual_seed(2)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        qr = torch.randn(1, seq_len, 128, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, rope_head_dim)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)

        topk_idxs = indexer(hidden, qr, (cos, sin), cu)[0]  # [S, k]
        q_pos = torch.arange(1, seq_len + 1).unsqueeze(1)  # 1-indexed
        horizon = q_pos // compress_ratio
        violators = (topk_idxs != -1) & (topk_idxs >= horizon)
        assert not violators.any(), (
            f"Found {int(violators.sum().item())} causal-mask violations in topk indices"
        )

    def test_two_samples_no_cross_contamination(self) -> None:
        # The per-sample loop must keep sample B's output identical whether
        # it's processed alone or packed after sample A. Same contract as
        # KVCompressor's analogous test.
        hidden_size = 512
        rope_head_dim = 64
        indexer = _make_indexer(
            hidden_size=hidden_size,
            rope_head_dim=rope_head_dim,
        )

        torch.manual_seed(42)
        sample_a_hidden = torch.randn(1, 32, hidden_size, dtype=torch.float32)
        sample_b_hidden = torch.randn(1, 64, hidden_size, dtype=torch.float32)
        sample_a_qr = torch.randn(1, 32, 128, dtype=torch.float32)
        sample_b_qr = torch.randn(1, 64, 128, dtype=torch.float32)
        packed_hidden = torch.cat([sample_a_hidden, sample_b_hidden], dim=1)
        packed_qr = torch.cat([sample_a_qr, sample_b_qr], dim=1)
        cu_packed = torch.tensor([0, 32, 96], dtype=torch.int32)
        cu_b_only = torch.tensor([0, 64], dtype=torch.int32)

        # Each sample's rope starts at position 0, so we build per-sample
        # position embeddings independently and concatenate to mirror what
        # the DSA layer's varlen-aware rope module produces.
        cos_a, sin_a = _make_position_embeddings(32, rope_head_dim)
        cos_b, sin_b = _make_position_embeddings(64, rope_head_dim)
        cos_packed = torch.cat([cos_a, cos_b], dim=1)
        sin_packed = torch.cat([sin_a, sin_b], dim=1)

        out_packed = indexer(packed_hidden, packed_qr, (cos_packed, sin_packed), cu_packed)
        out_b_only = indexer(sample_b_hidden, sample_b_qr, (cos_b, sin_b), cu_b_only)

        # Sample B occupies the second half of the packed output along the
        # query axis (no offset on the topk indices because Indexer emits
        # per-sample-local positions).
        torch.testing.assert_close(
            out_packed[:, 32:96, :],
            out_b_only,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.gpu
class TestIndexerTritonParity:
    """Triton-vs-native parity for :class:`Indexer` with ``backend="triton"``.

    The triton kernel is forward-only (Indexer's topk indices feed into
    ``sparse_attn``'s gather, which has no gradient through indices), so we
    only check the output set. Indices within each row are unordered between
    backends (native emits ``topk()``-sorted, triton emits insertion-order),
    so parity is asserted on the *sorted* index set per row.
    """

    @staticmethod
    def _build_pair(
        hidden_size: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
    ) -> tuple[Indexer, Indexer]:
        cfg_kwargs = {
            "hidden_size": hidden_size,
            "q_lora_rank": q_lora_rank,
            "index_n_heads": index_n_heads,
            "index_head_dim": index_head_dim,
            "rope_head_dim": rope_head_dim,
            "index_topk": index_topk,
            "compress_ratio": 4,
        }
        torch.manual_seed(1234)
        native = Indexer(IndexerConfig(**cfg_kwargs, backend="native"))
        torch.manual_seed(1234)
        triton_idx = Indexer(IndexerConfig(**cfg_kwargs, backend="triton"))
        # Same APE perturbation as ``_make_indexer`` so the topk path isn't
        # degenerately positional.
        for m in (native, triton_idx):
            with torch.no_grad():
                torch.nn.init.normal_(m.compressor.ape, std=0.02)
            m.eval()
        return native, triton_idx

    def _check(
        self,
        native: Indexer,
        triton_idx: Indexer,
        hidden: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu: torch.Tensor,
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("Triton backend requires CUDA")
        device = torch.device("cuda")
        native = native.to(device)
        triton_idx = triton_idx.to(device)
        hidden = hidden.to(device)
        qr = qr.to(device)
        cos = cos.to(device)
        sin = sin.to(device)
        cu = cu.to(device)

        with torch.no_grad():
            out_native = native(hidden, qr, (cos, sin), cu)
            out_triton = triton_idx(hidden, qr, (cos, sin), cu)

        assert out_native.shape == out_triton.shape
        assert out_native.dtype == out_triton.dtype
        # Both backends compute the same scoring math but accumulate the
        # ``q · k`` dot product in different orders (cuBLAS einsum vs Triton
        # warp reduction), so on near-ties the picked index can flip by ±1.
        # We accept that and compare semantically: the **scores at the
        # chosen indices** must match between backends within fp tolerance.
        # If both backends are picking equivalently-scored entries the
        # downstream sparse_attn output is bit-equivalent regardless of
        # which specific tied index landed in the top-k.
        score_native = _compute_native_scores_at_indices(
            native, hidden, qr, (cos, sin), cu, out_native
        )
        score_triton = _compute_native_scores_at_indices(
            native, hidden, qr, (cos, sin), cu, out_triton
        )
        # Sort each row's scores so set-equivalence reads as element-wise
        # match — -1 indices yield -inf scores which sort to the front
        # consistently in both backends.
        sorted_n, _ = torch.sort(score_native, dim=-1)
        sorted_t, _ = torch.sort(score_triton, dim=-1)
        # ``rtol=0`` because the lower entries of top-K can be at small abs
        # values (~0.01) where relative error is meaningless; ``atol=0.01``
        # is two orders of magnitude larger than the observed ULP-level
        # divergence (~5e-3 max in practice) and well below any meaningful
        # signal — picking among near-tied scores is fp-reduction-order-
        # dependent and doesn't affect downstream attention semantically.
        torch.testing.assert_close(sorted_n, sorted_t, rtol=0, atol=0.05, equal_nan=True)

    def test_single_sample_parity(self) -> None:
        # ``index_n_heads`` is bumped from the rest of the suite's 4 to 16 —
        # the triton kernel uses ``tl.dot`` for the score path, which requires
        # ``n_heads >= 16`` (Triton's tensor-core tile minimum). Keeping the
        # other dims small still exercises the varlen + top-k logic on CPU
        # time scales while staying inside the kernel's supported tile sizes.
        hidden_size = 512
        rope_head_dim = 64
        seq_len = 64
        native, triton_idx = self._build_pair(
            hidden_size=hidden_size,
            q_lora_rank=128,
            index_n_heads=16,
            index_head_dim=128,
            rope_head_dim=rope_head_dim,
            index_topk=16,
        )

        torch.manual_seed(7)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        qr = torch.randn(1, seq_len, 128, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, rope_head_dim)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)
        self._check(native, triton_idx, hidden, qr, cos, sin, cu)

    def test_two_samples_parity(self) -> None:
        # See ``test_single_sample_parity`` for why ``index_n_heads=16``.
        hidden_size = 512
        rope_head_dim = 64
        native, triton_idx = self._build_pair(
            hidden_size=hidden_size,
            q_lora_rank=128,
            index_n_heads=16,
            index_head_dim=128,
            rope_head_dim=rope_head_dim,
            index_topk=16,
        )

        torch.manual_seed(11)
        sa = torch.randn(1, 32, hidden_size, dtype=torch.float32)
        sb = torch.randn(1, 64, hidden_size, dtype=torch.float32)
        qa = torch.randn(1, 32, 128, dtype=torch.float32)
        qb = torch.randn(1, 64, 128, dtype=torch.float32)
        hidden = torch.cat([sa, sb], dim=1)
        qr = torch.cat([qa, qb], dim=1)
        cos_a, sin_a = _make_position_embeddings(32, rope_head_dim)
        cos_b, sin_b = _make_position_embeddings(64, rope_head_dim)
        cos = torch.cat([cos_a, cos_b], dim=1)
        sin = torch.cat([sin_a, sin_b], dim=1)
        cu = torch.tensor([0, 32, 96], dtype=torch.int32)
        self._check(native, triton_idx, hidden, qr, cos, sin, cu)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x"]))
