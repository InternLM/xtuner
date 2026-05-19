# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for :class:`xtuner.v1.module.decoder_layer.hc_block.HCDecoderLayer`."""

import torch
import torch.nn as nn

from xtuner.v1.module.decoder_layer import HCDecoderLayer, HCWrapperConfig


class MockBlock(nn.Module):
    """Minimal inner-block stub that satisfies the ``HCDecoderLayer`` contract.

    Exposes ``attn_block`` and ``ffn_block`` as ``[B, S, D]`` → ``[B, S, D]``
    callables that include an internal RMS-style norm followed by a linear
    projection — enough to exercise the wrapper without dragging in
    ``MoEDecoderLayer`` and its dispatcher / EP dependencies.
    """

    def __init__(self, hidden_size: int, *, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.input_layernorm = _RMSNorm(hidden_size)
        self.post_attention_layernorm = _RMSNorm(hidden_size)
        self.self_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def attn_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(self.input_layernorm(x))

    def ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.post_attention_layernorm(x))


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.square().mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class TestHCDecoderLayer:
    def test_hc_mult_1_equals_plain_residual(self):
        """``hc_mult=1`` must structurally degenerate to a plain pre-norm residual block."""
        hidden = 32
        torch.manual_seed(123)
        inner = MockBlock(hidden_size=hidden, seed=7)
        cfg = HCWrapperConfig(hc_mult=1)
        wrapper = HCDecoderLayer(inner=inner, hc_cfg=cfg, hidden_size=hidden)

        x_single = torch.randn(2, 4, hidden, dtype=torch.float32)
        x_hc = x_single.unsqueeze(-2)  # [B, S, 1, D]

        wrapper_out = wrapper(x_hc).squeeze(-2)

        # Reference plain pre-norm residual computed without HC machinery.
        ref = x_single + inner.attn_block(x_single)
        ref = ref + inner.ffn_block(ref)

        torch.testing.assert_close(wrapper_out, ref, atol=1e-5, rtol=1e-5)

    def test_zero_init_passes_through(self):
        """With the documented degenerate init the wrapper output stays finite and matches
        the analytic ``post=1, pre=0.5+eps, comb=1/H`` HC-mean prediction."""
        hidden = 16
        hc_mult = 4
        torch.manual_seed(2024)
        inner = MockBlock(hidden_size=hidden, seed=3)
        cfg = HCWrapperConfig(hc_mult=hc_mult)
        wrapper = HCDecoderLayer(inner=inner, hc_cfg=cfg, hidden_size=hidden)

        # Sanity: documented init left hc_*_fn / hc_*_base at 0 and only scale[0]=1.
        assert torch.equal(wrapper.hc_attn_fn, torch.zeros_like(wrapper.hc_attn_fn))
        assert torch.equal(wrapper.hc_attn_base, torch.zeros_like(wrapper.hc_attn_base))
        assert torch.equal(wrapper.hc_ffn_fn, torch.zeros_like(wrapper.hc_ffn_fn))
        assert torch.equal(wrapper.hc_ffn_base, torch.zeros_like(wrapper.hc_ffn_base))
        torch.testing.assert_close(wrapper.hc_attn_scale, torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(wrapper.hc_ffn_scale, torch.tensor([1.0, 0.0, 0.0]))

        x = torch.randn(1, 4, hc_mult, hidden, dtype=torch.float32)
        out = wrapper(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

        # Closed-form prediction under zero init: pre=0.5+eps (uniform), post=1, comb=1/H.
        # hc_pre output `y = (0.5+eps) * sum_h x[:,:,h]`; in degenerate state all h
        # streams are identical (we will check that explicitly below by feeding a uniform x).
        x_uniform = torch.randn(1, 4, 1, hidden, dtype=torch.float32).expand(1, 4, hc_mult, hidden).contiguous()
        out_uniform = wrapper(x_uniform)
        x_single = x_uniform[:, :, 0, :]
        # With uniform streams: sum_h x[:,:,h] = H * x_single, pre=0.5+eps -> y = (0.5+eps)*H*x_single.
        y_attn = (0.5 + cfg.hc_eps) * hc_mult * x_single
        attn_out = inner.attn_block(y_attn)
        # After hc_post (post=1, comb=1/H), each output stream = attn_out + mean_h(x_uniform) = attn_out + x_single.
        post_attn_stream = attn_out + x_single
        # Repeat for FFN.
        # New uniform residual: each stream equals post_attn_stream.
        y_ffn = (0.5 + cfg.hc_eps) * hc_mult * post_attn_stream
        ffn_out = inner.ffn_block(y_ffn)
        expected_stream = ffn_out + post_attn_stream

        for h in range(hc_mult):
            torch.testing.assert_close(out_uniform[:, :, h, :], expected_stream, atol=1e-4, rtol=1e-4)

    def test_forward_shapes(self):
        hidden = 128
        hc_mult = 4
        torch.manual_seed(0)
        inner = MockBlock(hidden_size=hidden, seed=11)
        cfg = HCWrapperConfig(hc_mult=hc_mult)
        wrapper = HCDecoderLayer(inner=inner, hc_cfg=cfg, hidden_size=hidden)

        x = torch.randn(1, 4, hc_mult, hidden, dtype=torch.float32)
        out = wrapper(x)
        assert out.shape == (1, 4, hc_mult, hidden)
        assert torch.isfinite(out).all()

    def test_grad_flows(self):
        hidden = 32
        hc_mult = 4
        torch.manual_seed(0)
        inner = MockBlock(hidden_size=hidden, seed=5)
        cfg = HCWrapperConfig(hc_mult=hc_mult)
        wrapper = HCDecoderLayer(inner=inner, hc_cfg=cfg, hidden_size=hidden)

        # Push hc_attn_fn off the zero attractor so its gradient is non-trivial.
        with torch.no_grad():
            wrapper.hc_attn_fn.add_(0.01 * torch.randn_like(wrapper.hc_attn_fn))

        x = torch.randn(1, 4, hc_mult, hidden, dtype=torch.float32, requires_grad=True)
        out = wrapper(x)
        out.sum().backward()

        assert wrapper.hc_attn_fn.grad is not None
        assert torch.isfinite(wrapper.hc_attn_fn.grad).all()
        assert wrapper.hc_attn_fn.grad.abs().sum().item() > 0.0
