# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for :func:`xtuner.v1.module.decoder_layer.deepseek_v4.hc_sinkhorn.hc_split_sinkhorn`."""

import torch

from xtuner.v1.module.decoder_layer import hc_split_sinkhorn


HC_MULT = 4
MIX_DIM = (2 + HC_MULT) * HC_MULT


def _make_inputs(batch: int = 1, seq: int = 8, hc_mult: int = HC_MULT, *, dtype: torch.dtype = torch.float32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mix_dim = (2 + hc_mult) * hc_mult
    mixes = torch.randn(batch, seq, mix_dim, generator=g, dtype=dtype)
    # Use a non-zero scale so the `comb` block actually carries information; init pattern
    # mirrors the production "scale[0]=1, rest=0" but bumps scale[2] to inject signal.
    hc_scale = torch.tensor([1.0, 0.5, 0.5], dtype=dtype)
    hc_base = torch.randn(mix_dim, generator=g, dtype=dtype) * 0.1
    return mixes, hc_scale, hc_base


class TestHCSplitSinkhorn:
    def test_sinkhorn_doubly_stochastic(self):
        mixes, hc_scale, hc_base = _make_inputs()
        _, _, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        # Row and column sums of a doubly-stochastic matrix are 1.
        row_sums = comb.sum(dim=-1)
        col_sums = comb.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=1e-3, rtol=1e-3)

    def test_bf16_rounded_inputs_stay_finite(self):
        """bf16-quantised values must not drive the 20-iteration Sinkhorn to NaN.

        The kernel itself takes fp32 (``hc_pre`` casts the bf16 Linear output up, and the
        HC parameters are fp32 by construction) — repeated ``/ (sum + eps)`` rounds are
        bf16-unstable, which is exactly why the contract is fp32. What still reaches it is
        bf16-*rounded* data, so round-trip the inputs through bf16 and check the loop
        survives the coarser values.
        """
        mixes, hc_scale, hc_base = _make_inputs(dtype=torch.float32)
        rounded = [t.to(torch.bfloat16).float() for t in (mixes, hc_scale, hc_base)]
        pre, post, comb = hc_split_sinkhorn(*rounded, hc_mult=HC_MULT, iters=20, eps=1e-6)
        for name, t in (("pre", pre), ("post", post), ("comb", comb)):
            assert t.dtype == torch.float32, f"{name} dtype mismatch: {t.dtype}"
            assert torch.isfinite(t).all(), f"{name} contains NaN or Inf"

    def test_deterministic(self):
        mixes, hc_scale, hc_base = _make_inputs(seed=42)
        out_a = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        out_b = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        for a, b in zip(out_a, out_b):
            assert torch.equal(a, b), "hc_split_sinkhorn should be bit-deterministic for identical input"

    def test_zero_init_degenerate(self):
        # With all-zero hc_fn upstream we get mixes=0; combined with zero base, the
        # sinkhorn must collapse to a uniform doubly-stochastic comb (1/H).
        batch, seq = 2, 4
        mixes = torch.zeros(batch, seq, MIX_DIM, dtype=torch.float32)
        hc_scale = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        hc_base = torch.zeros(MIX_DIM, dtype=torch.float32)

        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)

        uniform = torch.full_like(comb, 1.0 / HC_MULT)
        torch.testing.assert_close(comb, uniform, atol=1e-4, rtol=1e-4)
        # pre = sigmoid(0) + eps = 0.5 + eps; post = 2 * sigmoid(0) = 1.0.
        torch.testing.assert_close(pre, torch.full_like(pre, 0.5 + 1e-6), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(post, torch.ones_like(post), atol=1e-5, rtol=1e-5)
