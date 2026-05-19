# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from xtuner.v1.module.attention.sparse_attn import sparse_attn


def _dense_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    # Pure dense reference matching sparse_attn's MQA contract: kv is shared
    # across heads and an extra per-head sink slot absorbs probability mass
    # but contributes zero to the output.
    bsz, seq, num_heads, head_dim = q.shape
    t_total = kv.size(1)
    q_f = q.float()
    kv_f = kv.float()
    sink = attn_sink.float().view(1, 1, num_heads, 1).expand(bsz, seq, num_heads, 1)
    logits = torch.einsum("bshd,btd->bsht", q_f, kv_f) * softmax_scale
    logits_with_sink = torch.cat([logits, sink], dim=-1)
    weights = torch.softmax(logits_with_sink, dim=-1)
    kv_weights = weights[..., :t_total]
    return torch.einsum("bsht,btd->bshd", kv_weights, kv_f).to(q.dtype)


class TestSparseAttn:
    def test_full_topk_matches_dense(self) -> None:
        # When topk_idxs covers the entire KV (no -1, no causal cuts), the
        # sparse path must reproduce dense attention bit-for-bit (modulo
        # tiny gather/einsum reorderings).
        torch.manual_seed(0)
        bsz, seq, num_heads, head_dim = 1, 8, 2, 16
        t_total = seq
        q = torch.randn(bsz, seq, num_heads, head_dim, dtype=torch.float32)
        kv = torch.randn(bsz, t_total, head_dim, dtype=torch.float32)
        attn_sink = torch.randn(num_heads, dtype=torch.float32)
        softmax_scale = head_dim**-0.5
        topk_idxs = torch.arange(t_total).view(1, 1, t_total).expand(1, seq, t_total).contiguous().long()
        cu = torch.tensor([0, seq], dtype=torch.int32)

        out_sparse = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, cu)
        out_dense = _dense_attention(q, kv, attn_sink, softmax_scale)

        torch.testing.assert_close(out_sparse, out_dense, rtol=1e-5, atol=1e-5)

    def test_sink_absorbs_prob_mass(self) -> None:
        # If every q-KV logit is extremely negative compared to the sink,
        # softmax mass flows almost entirely onto the sink slot, which is
        # dropped from the output -> result must be ~zero.
        torch.manual_seed(1)
        bsz, seq, num_heads, head_dim = 1, 4, 2, 16
        t_total = seq
        # Drive q and kv to near-zero so q·K is tiny; raise the sink very
        # high so softmax routes essentially all weight to the sink slot.
        q = torch.zeros(bsz, seq, num_heads, head_dim, dtype=torch.float32)
        kv = torch.zeros(bsz, t_total, head_dim, dtype=torch.float32)
        attn_sink = torch.full((num_heads,), 100.0, dtype=torch.float32)
        softmax_scale = head_dim**-0.5
        topk_idxs = torch.arange(t_total).view(1, 1, t_total).expand(1, seq, t_total).contiguous().long()
        cu = torch.tensor([0, seq], dtype=torch.int32)

        out = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, cu)
        # KV is exactly zero so the output must be exactly zero — this also
        # proves the sink slot is never multiplied back into the output (a
        # bug there would surface as a non-zero contribution from the sink
        # "value", which we explicitly disallow).
        assert torch.equal(out, torch.zeros_like(out))

        # Repeat with non-zero KV but a q dotted negatively into KV so every
        # q·K logit is uniformly very negative. We construct q = -1e3 * kv
        # per token so q·K = -1e3 * ||k||^2 << sink logit for every (s, t).
        kv_nz = torch.rand(bsz, t_total, head_dim, dtype=torch.float32) + 0.5
        # q must broadcast to [B, S, H, D]; we use the same neg-kv pattern
        # across all heads to make the bookkeeping easy.
        q_aligned = -1e3 * kv_nz.unsqueeze(2).expand(bsz, t_total, num_heads, head_dim).contiguous()
        out_nz = sparse_attn(q_aligned, kv_nz, attn_sink, topk_idxs, softmax_scale, cu)
        # Sink logit 100 against q·K ≈ -1e3 * head_dim * 0.75 / sqrt(d) so
        # softmax routes essentially all mass to the sink slot and the
        # output is ~zero.
        assert out_nz.abs().max() < 1e-3

    def test_masked_minus_one_ignored(self) -> None:
        # Marking half of topk_idxs as -1 should produce the same output as
        # running sparse_attn with the unmasked-half-only index list.
        torch.manual_seed(2)
        bsz, seq, num_heads, head_dim = 1, 4, 2, 16
        t_total = 8
        q = torch.randn(bsz, seq, num_heads, head_dim, dtype=torch.float32)
        kv = torch.randn(bsz, t_total, head_dim, dtype=torch.float32)
        attn_sink = torch.randn(num_heads, dtype=torch.float32)
        softmax_scale = head_dim**-0.5
        cu = torch.tensor([0, seq], dtype=torch.int32)

        topk_full = torch.arange(t_total).view(1, 1, t_total).expand(1, seq, t_total).contiguous().long()
        # Mask the second half: indices 4..7 -> -1.
        topk_masked = topk_full.clone()
        topk_masked[..., t_total // 2 :] = -1

        # Reference run uses only the first half of indices.
        topk_half = topk_full[..., : t_total // 2].contiguous()
        out_masked = sparse_attn(q, kv, attn_sink, topk_masked, softmax_scale, cu)
        out_half = sparse_attn(q, kv, attn_sink, topk_half, softmax_scale, cu)

        torch.testing.assert_close(out_masked, out_half, rtol=1e-5, atol=1e-5)

    def test_shapes_passthrough(self) -> None:
        torch.manual_seed(3)
        bsz, seq, num_heads, head_dim = 1, 12, 4, 32
        t_total = 16
        k = 5
        q = torch.randn(bsz, seq, num_heads, head_dim, dtype=torch.float32)
        kv = torch.randn(bsz, t_total, head_dim, dtype=torch.float32)
        attn_sink = torch.randn(num_heads, dtype=torch.float32)
        topk_idxs = torch.randint(0, t_total, (bsz, seq, k), dtype=torch.long)
        cu = torch.tensor([0, seq], dtype=torch.int32)

        out = sparse_attn(q, kv, attn_sink, topk_idxs, head_dim**-0.5, cu)
        assert out.shape == q.shape
        assert out.dtype == q.dtype
        assert torch.isfinite(out).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x"]))
