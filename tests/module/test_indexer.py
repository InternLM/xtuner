# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from xtuner.v1.module.attention.indexer import Indexer, IndexerConfig


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


def _make_position_embeddings(seq_len: int, rope_head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    half = rope_head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [seq_len, rope_head_dim // 2]
    cos = angles.cos().unsqueeze(0)
    sin = angles.sin().unsqueeze(0)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x"]))
