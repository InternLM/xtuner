# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from xtuner.v1.module.attention.kv_compressor import KVCompressor


def _make_compressor(
    hidden_size: int = 512,
    head_dim: int = 128,
    compress_ratio: int = 4,
    overlap: bool = True,
    seed: int = 1234,
) -> KVCompressor:
    torch.manual_seed(seed)
    compressor = KVCompressor(
        hidden_size=hidden_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    # Initialize APE with a non-zero pattern so the gate softmax is not
    # uniform across positions; this exercises the overlap_transform path
    # rather than collapsing it to a plain mean.
    with torch.no_grad():
        torch.nn.init.normal_(compressor.ape, std=0.02)
    compressor.eval()
    return compressor


class TestKVCompressor:
    def test_single_sample_compress_4(self) -> None:
        hidden_size = 512
        head_dim = 128
        seq_len = 64
        compressor = _make_compressor(
            hidden_size=hidden_size,
            head_dim=head_dim,
            compress_ratio=4,
            overlap=True,
        )
        torch.manual_seed(0)
        x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)

        out, cu_out = compressor(x, cu)

        assert out.shape == (1, seq_len // 4, head_dim)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert torch.equal(cu_out, torch.tensor([0, seq_len // 4], dtype=torch.int32))

    def test_single_sample_compress_128(self) -> None:
        hidden_size = 512
        head_dim = 128
        seq_len = 512
        compressor = _make_compressor(
            hidden_size=hidden_size,
            head_dim=head_dim,
            compress_ratio=128,
            overlap=False,
        )
        torch.manual_seed(1)
        x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)

        out, cu_out = compressor(x, cu)

        assert out.shape == (1, 4, head_dim)
        assert torch.isfinite(out).all()
        assert torch.equal(cu_out, torch.tensor([0, 4], dtype=torch.int32))

    def test_two_samples_no_cross_contamination(self) -> None:
        # The contract that matters here is sample isolation: compressing
        # sample B alone must produce the same bits as compressing the
        # packed (A, B) sequence and slicing out B's region. If the overlap
        # window leaked across the cu_seq_lens boundary, sample B's first
        # compressed token would depend on the tail of sample A.
        hidden_size = 512
        head_dim = 128
        compressor = _make_compressor(
            hidden_size=hidden_size,
            head_dim=head_dim,
            compress_ratio=4,
            overlap=True,
        )
        torch.manual_seed(42)
        sample_a = torch.randn(1, 64, hidden_size, dtype=torch.float32)
        sample_b = torch.randn(1, 128, hidden_size, dtype=torch.float32)
        packed = torch.cat([sample_a, sample_b], dim=1)
        cu_packed = torch.tensor([0, 64, 64 + 128], dtype=torch.int32)
        cu_b_only = torch.tensor([0, 128], dtype=torch.int32)

        out_packed, cu_out_packed = compressor(packed, cu_packed)
        out_b_only, cu_out_b_only = compressor(sample_b, cu_b_only)

        # Sample A occupies the first 64 // 4 = 16 compressed tokens; sample
        # B occupies the next 128 // 4 = 32. The B slice in the packed run
        # should equal the standalone B run bit-for-bit.
        a_compressed = 64 // 4
        b_compressed = 128 // 4
        torch.testing.assert_close(
            out_packed[:, a_compressed : a_compressed + b_compressed, :],
            out_b_only,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.equal(
            cu_out_packed,
            torch.tensor([0, a_compressed, a_compressed + b_compressed], dtype=torch.int32),
        )
        assert torch.equal(cu_out_b_only, torch.tensor([0, b_compressed], dtype=torch.int32))

    def test_padding_when_not_divisible(self) -> None:
        hidden_size = 512
        head_dim = 128
        seq_len = 65
        compressor = _make_compressor(
            hidden_size=hidden_size,
            head_dim=head_dim,
            compress_ratio=4,
            overlap=True,
        )
        torch.manual_seed(7)
        x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)

        out, cu_out = compressor(x, cu)

        # 65 -> padded to 68 -> 17 compressed tokens. The 17th is a partial
        # group (1 real token + 3 zero-padded tokens). We do not assert its
        # exact value, only that the pipeline produced finite numbers.
        assert out.shape == (1, 17, head_dim)
        assert torch.isfinite(out).all()
        assert torch.equal(cu_out, torch.tensor([0, 17], dtype=torch.int32))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x"]))
