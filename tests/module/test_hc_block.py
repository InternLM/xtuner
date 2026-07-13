# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for :func:`xtuner.v1.module.decoder_layer.deepseek_v4.hc_block.hc_pre` and
:func:`hc_post`.

These tests exercise the HC residual-mix math directly as functions; the
previous test surface (``HCDecoderLayer`` wrapping a mock attn+ffn inner
block) was removed when the V4 layer was consolidated and the class no
longer exists. The math is the same; the test fixture now constructs the
HC parameters explicitly and calls ``hc_pre`` / ``hc_post`` in the same
order :class:`xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer.forward`
does.
"""

import torch
import torch.nn as nn

from xtuner.v1.module.decoder_layer import HCWrapperConfig, hc_post, hc_pre


class _MockSubBlock(nn.Module):
    """Minimal ``[B, S, D] -> [B, S, D]`` callable used as a stand-in for the
    attn / ffn sub-block inside the HC residual pattern."""

    def __init__(self, hidden_size: int, *, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.norm = _RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.square().mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


def _make_hc_params(hc_mult: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (hc_fn, hc_scale, hc_base) at the V4DecoderLayer's documented
    degenerate init: zeros plus ``scale[0] = 1`` so ``hc_pre`` produces a
    uniform mean over the streams instead of softmax noise."""
    mix_dim = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size
    hc_fn = torch.zeros(mix_dim, hc_dim, dtype=torch.float32)
    hc_base = torch.zeros(mix_dim, dtype=torch.float32)
    hc_scale = torch.zeros(3, dtype=torch.float32)
    hc_scale[0] = 1.0
    return hc_fn, hc_scale, hc_base


def _apply_hc_pair(
    x: torch.Tensor,
    sub_block: nn.Module,
    cfg: HCWrapperConfig,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> torch.Tensor:
    """Run one ``hc_pre`` → sub_block → ``hc_post`` pass, the same pattern
    :class:`V4DecoderLayer.forward` uses for both the attn and ffn halves."""
    residual = x
    x_reduced, post, comb = hc_pre(
        x, hc_fn, hc_scale, hc_base, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
    )
    out = sub_block(x_reduced)
    return hc_post(out, residual, post, comb)


class TestHCResidualMath:
    def test_zero_init_passes_through(self) -> None:
        """With the documented degenerate init, ``hc_pre`` produces a uniform
        ``0.5 + eps`` weight per stream and ``hc_post`` produces a per-stream
        ``post=1, comb=1/H`` mix. Closed form: each output stream =
        ``sub_block((0.5+eps) * H * mean_h x[h]) + mean_h x[h]``."""
        hidden = 16
        hc_mult = 4
        torch.manual_seed(2024)
        sub_block = _MockSubBlock(hidden_size=hidden, seed=3)
        cfg = HCWrapperConfig(hc_mult=hc_mult)
        hc_fn, hc_scale, hc_base = _make_hc_params(hc_mult, hidden)

        # Feed uniform streams so the closed form is tractable: streams are
        # identical so ``mean_h x[h] == x[:, :, 0, :]`` and ``sum_h x[h] == H * x[:, :, 0, :]``.
        x_single = torch.randn(1, 4, 1, hidden, dtype=torch.float32)
        x_uniform = x_single.expand(1, 4, hc_mult, hidden).contiguous()

        out = _apply_hc_pair(x_uniform, sub_block, cfg, hc_fn, hc_scale, hc_base)

        assert out.shape == x_uniform.shape
        assert torch.isfinite(out).all()

        # Closed-form prediction. ``hc_pre`` weight per stream is ``0.5 + eps``,
        # so the reduced input is ``(0.5+eps) * H * x_single``. After hc_post:
        # each output stream = ``sub_block_out + mean_h(x_uniform) = sub_block_out + x_single``.
        x_reduced_expected = (0.5 + cfg.hc_eps) * hc_mult * x_single.squeeze(-2)
        sub_out = sub_block(x_reduced_expected)
        expected_stream = sub_out + x_single.squeeze(-2)

        for h in range(hc_mult):
            torch.testing.assert_close(out[:, :, h, :], expected_stream, atol=1e-4, rtol=1e-4)

    def test_forward_shapes(self) -> None:
        hidden = 128
        hc_mult = 4
        torch.manual_seed(0)
        sub_block = _MockSubBlock(hidden_size=hidden, seed=11)
        cfg = HCWrapperConfig(hc_mult=hc_mult)
        hc_fn, hc_scale, hc_base = _make_hc_params(hc_mult, hidden)

        x = torch.randn(1, 4, hc_mult, hidden, dtype=torch.float32)
        out = _apply_hc_pair(x, sub_block, cfg, hc_fn, hc_scale, hc_base)

        assert out.shape == (1, 4, hc_mult, hidden)
        assert torch.isfinite(out).all()

    def test_grad_flows(self) -> None:
        """Push the HC mix params off the zero attractor and verify gradients
        flow back into ``hc_fn`` through the ``hc_pre`` + sub_block + ``hc_post``
        chain."""
        hidden = 32
        hc_mult = 4
        torch.manual_seed(0)
        sub_block = _MockSubBlock(hidden_size=hidden, seed=5)
        cfg = HCWrapperConfig(hc_mult=hc_mult)
        hc_fn, hc_scale, hc_base = _make_hc_params(hc_mult, hidden)
        hc_fn = (hc_fn + 0.01 * torch.randn_like(hc_fn)).requires_grad_(True)
        hc_scale = hc_scale.requires_grad_(True)
        hc_base = hc_base.requires_grad_(True)

        x = torch.randn(1, 4, hc_mult, hidden, dtype=torch.float32, requires_grad=True)
        out = _apply_hc_pair(x, sub_block, cfg, hc_fn, hc_scale, hc_base)
        out.sum().backward()

        assert hc_fn.grad is not None
        assert torch.isfinite(hc_fn.grad).all()
        assert hc_fn.grad.abs().sum().item() > 0.0
