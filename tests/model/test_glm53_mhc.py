"""GLM-5.3-Flash 的 mHC 四流残差原语，见 doc/xtuner_glm5p3flash_design.md F4。

TestHCSplitSinkhorn
    test_sinkhorn_doubly_stochastic              comb 迭代后行列和均为 1
    test_comb_is_not_symmetric                   comb 非对称，故 hc_post 的转置是语义需要
    test_hc_mult_1_degenerates_to_identity_comb  hc_mult=1 退化成普通残差
    test_deterministic                           同输入两次调用逐位一致
TestHCPreHCPostMatchesHF
    test_hc_pre_hc_post_matches_hf_hyper_connection  fp32 下与 HF 实现逐值一致
    test_hc_pre_hc_post_matches_hf_bf16              bf16 下与 HF 实现一致
    test_zero_init_produces_uniform_stream_collapse  零初始化时四流等权塌缩
    test_grad_flows_through_hc_pre_and_hc_post       梯度能穿过 hc_pre/hc_post
"""

import torch
import torch.nn as nn

from xtuner.v1.module.decoder_layer.mhc import hc_post, hc_pre, hc_split_sinkhorn


HC_MULT = 4
HIDDEN = 32
MIX_DIM = (2 + HC_MULT) * HC_MULT


def _hf_glm53_hyper_connection(hidden_size: int, hc_mult: int, hc_eps: float, hc_sinkhorn_iters: int):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    config = Glm5NextTextConfig(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        hc_eps=hc_eps,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        rms_norm_eps=1e-6,
    )
    module = Glm5NextTextHyperConnection(config)
    # __init__ uses torch.empty (weights normally come from _init_weights /
    # from_pretrained, which this standalone unit test doesn't run) -- fill with a fixed
    # random init so `fn`/`base`/`scale` aren't NaN/garbage.
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.02)
    return module


class TestHCSplitSinkhorn:
    def _make_inputs(self, batch=1, seq=8, hc_mult=HC_MULT, *, dtype=torch.float32, seed=0):
        g = torch.Generator().manual_seed(seed)
        mix_dim = (2 + hc_mult) * hc_mult
        mixes = torch.randn(batch, seq, mix_dim, generator=g, dtype=dtype)
        hc_scale = torch.tensor([1.0, 0.5, 0.5], dtype=dtype)
        hc_base = torch.randn(mix_dim, generator=g, dtype=dtype) * 0.1
        return mixes, hc_scale, hc_base

    def test_sinkhorn_doubly_stochastic(self):
        # Sinkhorn 迭代后 comb 必须行和列和都为 1，这是 mHC 的流量守恒前提。
        mixes, hc_scale, hc_base = self._make_inputs()
        _, _, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        row_sums = comb.sum(dim=-1)
        col_sums = comb.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=1e-3, rtol=1e-3)

    def test_comb_is_not_symmetric(self):
        """Regression guard: comb is doubly-stochastic but NOT symmetric, so hc_post must
        reduce over comb's *first* axis (transpose), not just matmul it directly."""
        # comb 不是对称阵，所以 hc_post 里的 transpose 是语义必需而非性能重排。
        mixes, hc_scale, hc_base = self._make_inputs(seed=7)
        _, _, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        assert not torch.allclose(comb, comb.transpose(-1, -2), atol=1e-4)

    def test_hc_mult_1_degenerates_to_identity_comb(self):
        """hc_mult=1 makes comb a 1x1 doubly-stochastic matrix, i.e. exactly 1.0 -- the mHC
        math then degenerates to a plain pre-norm residual (pre/post act as pure scalars)."""
        # hc_mult=1 时 mHC 必须退化成普通 pre-norm 残差，作为结构正确性锚点。
        mixes, hc_scale, hc_base = self._make_inputs(hc_mult=1, seed=3)
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=1, iters=20, eps=1e-6)
        torch.testing.assert_close(comb, torch.ones_like(comb), atol=1e-6, rtol=1e-6)

    def test_deterministic(self):
        # 同一输入两次调用必须逐位一致，Sinkhorn 不能引入非确定性。
        mixes, hc_scale, hc_base = self._make_inputs(seed=42)
        out_a = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        out_b = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=HC_MULT, iters=20, eps=1e-6)
        for a, b in zip(out_a, out_b):
            assert torch.equal(a, b)


class TestHCPreHCPostMatchesHF:
    def test_hc_pre_hc_post_matches_hf_hyper_connection(self):
        """hc_pre + sub_block + hc_post, run in fp32, must match HF's
        Glm5NextTextHyperConnection + Glm5NextTextDecoderLayer's residual expression
        closely: at fp32 the bf16-fast-path casts inside hc_pre become no-ops, so this
        exercises the exact same arithmetic HF's reference uses."""
        # fp32 下 hc_pre/hc_post 组合必须与 HF Glm5NextTextHyperConnection 逐值一致。
        torch.manual_seed(0)
        hf_hc = _hf_glm53_hyper_connection(HIDDEN, HC_MULT, hc_eps=1e-6, hc_sinkhorn_iters=20)

        streams = torch.randn(2, 5, HC_MULT, HIDDEN, dtype=torch.float32)
        sub_block = nn.Linear(HIDDEN, HIDDEN, bias=False)

        residual = streams
        hf_post, hf_comb, hf_collapsed = hf_hc(streams)
        hf_sub_out = sub_block(hf_collapsed)
        hf_out = hf_post.unsqueeze(-1) * hf_sub_out.unsqueeze(-2) + torch.matmul(hf_comb.transpose(-1, -2), residual)

        xtuner_collapsed, xtuner_post, xtuner_comb = hc_pre(
            streams, hf_hc.fn, hf_hc.scale, hf_hc.base, HC_MULT, iters=20, eps=1e-6, norm_eps=1e-6
        )
        torch.testing.assert_close(xtuner_collapsed, hf_collapsed, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(xtuner_post, hf_post, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(xtuner_comb, hf_comb, atol=1e-5, rtol=1e-5)

        xtuner_sub_out = sub_block(xtuner_collapsed)
        xtuner_out = hc_post(xtuner_sub_out, residual, xtuner_post, xtuner_comb)
        torch.testing.assert_close(xtuner_out, hf_out, atol=1e-4, rtol=1e-4)

    def test_hc_pre_hc_post_matches_hf_bf16(self):
        """Same check at the production bf16 dtype; wider tolerance since hc_pre's fast
        path (bf16 Linear instead of HF's fp32-throughout) and hc_post's fp32-accumulate
        eager path both differ from HF by a bf16 rounding boundary or two."""
        # bf16 下同样要与 HF 一致，覆盖训练实际使用的精度。
        torch.manual_seed(1)
        hf_hc = _hf_glm53_hyper_connection(HIDDEN, HC_MULT, hc_eps=1e-6, hc_sinkhorn_iters=20)

        streams = torch.randn(2, 5, HC_MULT, HIDDEN, dtype=torch.bfloat16)
        sub_block = nn.Linear(HIDDEN, HIDDEN, bias=False).to(torch.bfloat16)

        residual = streams
        hf_post, hf_comb, hf_collapsed = hf_hc(streams)
        hf_sub_out = sub_block(hf_collapsed)
        dtype = streams.dtype
        hf_out = hf_post.to(dtype).unsqueeze(-1) * hf_sub_out.unsqueeze(-2) + torch.matmul(
            hf_comb.to(dtype).transpose(-1, -2), residual
        )

        xtuner_collapsed, xtuner_post, xtuner_comb = hc_pre(
            streams, hf_hc.fn, hf_hc.scale, hf_hc.base, HC_MULT, iters=20, eps=1e-6, norm_eps=1e-6
        )
        xtuner_sub_out = sub_block(xtuner_collapsed)
        xtuner_out = hc_post(xtuner_sub_out, residual, xtuner_post, xtuner_comb)

        diff = (xtuner_out.float() - hf_out.float()).abs()
        assert diff.max().item() < 5e-2, f"max abs diff {diff.max().item():.3e}"
        assert diff.mean().item() < 5e-3, f"mean abs diff {diff.mean().item():.3e}"

    def test_zero_init_produces_uniform_stream_collapse(self):
        """With hc_fn=0/hc_base=0/hc_scale=[1,0,0] (the documented degenerate init), hc_pre
        collapses to a uniform mean-like weighting and hc_post's comb is uniform 1/H."""
        # 零初始化时四条流应等权塌缩，等价于未训练时的普通残差。
        hc_mult = HC_MULT
        mix_dim = (2 + hc_mult) * hc_mult
        hc_fn = torch.zeros(mix_dim, hc_mult * HIDDEN, dtype=torch.float32)
        hc_base = torch.zeros(mix_dim, dtype=torch.float32)
        hc_scale = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        x_single = torch.randn(1, 4, 1, HIDDEN, dtype=torch.float32)
        x_uniform = x_single.expand(1, 4, hc_mult, HIDDEN).contiguous()

        collapsed, post, comb = hc_pre(x_uniform, hc_fn, hc_scale, hc_base, hc_mult, iters=20, eps=1e-6)

        uniform_comb = torch.full_like(comb, 1.0 / hc_mult)
        torch.testing.assert_close(comb, uniform_comb, atol=1e-4, rtol=1e-4)
        expected_collapsed = (0.5 + 1e-6) * hc_mult * x_single.squeeze(-2)
        torch.testing.assert_close(collapsed, expected_collapsed, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(post, torch.ones_like(post), atol=1e-5, rtol=1e-5)

    def test_grad_flows_through_hc_pre_and_hc_post(self):
        # 梯度必须穿过 hc_pre/hc_post 到达输入与三组 hc 参数。
        torch.manual_seed(0)
        hc_mult = HC_MULT
        mix_dim = (2 + hc_mult) * hc_mult
        hc_fn = (0.01 * torch.randn(mix_dim, hc_mult * HIDDEN, dtype=torch.float32)).requires_grad_(True)
        hc_base = torch.zeros(mix_dim, dtype=torch.float32, requires_grad=True)
        hc_scale = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)

        x = torch.randn(1, 4, hc_mult, HIDDEN, dtype=torch.float32, requires_grad=True)
        sub_block = nn.Linear(HIDDEN, HIDDEN, bias=False)

        collapsed, post, comb = hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, iters=20, eps=1e-6)
        out = hc_post(sub_block(collapsed), x, post, comb)
        out.sum().backward()

        assert hc_fn.grad is not None
        assert torch.isfinite(hc_fn.grad).all()
        assert hc_fn.grad.abs().sum().item() > 0.0
        assert x.grad is not None and torch.isfinite(x.grad).all()
