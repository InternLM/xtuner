"""GLM-5.3-Flash 的 mHC decoder 层，见 doc/xtuner_glm5p3flash_design.md F4。

用 KDA 作为两类层的替身注意力：mHC 的包装逻辑与注意力类型无关
（Glm53MoEDecoderLayer 只覆盖 MoEDecoderLayer 的 _pre/_post_moe_forward 接缝）。

TestGlm53DenseDecoderLayer
    test_forward_finite_and_shape_preserving             前向保持四流形状且数值有限
    test_mhc_cfg_none_matches_plain_dense_decoder_layer  mhc_cfg=None 等价于基类
    test_grad_flows_through_hc_params                    梯度能到达 hc_* 参数
TestGlm53MoEDecoderLayer
    test_forward_finite_and_shape_preserving             同上，MoE 版
    test_mhc_cfg_none_matches_plain_moe_decoder_layer    同上，MoE 版
    test_grad_flows_through_hc_params_and_experts        梯度能到达 hc_* 与专家
TestGlm53DecoderLayerCompile
    test_dense_layer_forward_compiles_with_dynamic_cu_seqlens  动态 cu_seqlens 下可编译
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53.decoder_layer import Glm53DenseDecoderLayer, Glm53MoEDecoderLayer
from xtuner.v1.module import KDAConfig
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.mhc import MHCConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEDecoderLayer
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


HIDDEN = 256
HC_MULT = 4


def _kda_attention_config(head_dim=32, num_heads=8):
    return KDAConfig(num_heads=num_heads, head_dim=head_dim, conv_kernel_size=4, gate_lower_bound=-5.0)


def _seq_ctx(seq_len: int, device: str) -> SequenceContext:
    return SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device=device)


def _init_uninitialized_moe_params(layer) -> None:
    """``MoEGate.weight`` and ``GroupedLinear.weight`` (the routed-expert weights) are left
    ``torch.empty`` -- real models initialize them through the top-level
    ``MoE.init_weights()`` pass, which these standalone decoder-layer tests don't run.
    ``torch.empty`` doesn't consume the RNG stream, so two independently-constructed layers
    (even from the same seed) get *different* uninitialized garbage here -- fill
    deterministically so forward doesn't propagate that garbage, and so the two-layer
    equivalence tests compare like-for-like weights."""
    with torch.no_grad():
        layer.gate.weight.normal_(mean=0.0, std=0.02)
        layer.experts.fused_w1w3.weight.normal_(mean=0.0, std=0.02)
        layer.experts.fused_w2.weight.normal_(mean=0.0, std=0.02)


def _dense_kwargs(mhc_cfg):
    return dict(
        hidden_size=HIDDEN,
        intermediate_size=512,
        hidden_act="silu",
        attention_config=_kda_attention_config(),
        mhc_cfg=mhc_cfg,
    )


def _moe_kwargs(mhc_cfg):
    return dict(
        hidden_size=HIDDEN,
        intermediate_size=512,
        moe_intermediate_size=128,
        hidden_act="silu",
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        attention_config=_kda_attention_config(),
        router_config=NoAuxRouterConfig(
            scoring_func="sigmoid", router_scaling_factor=1.0, norm_topk_prob=True, n_group=1, topk_group=1
        ),
        moe_act_fn_cfg=MoEActFnConfig(act_type="swiglu"),
        dispatcher=None,
        ep_mesh=None,
        mhc_cfg=mhc_cfg,
    )


class TestGlm53DenseDecoderLayer:
    @pytest.mark.gpu
    def test_forward_finite_and_shape_preserving(self):
        # mHC 包装后前向仍保持 [B,S,hc_mult,D] 形状且数值有限。
        torch.manual_seed(0)
        mhc_cfg = MHCConfig(hc_mult=HC_MULT, hc_eps=1e-6, hc_sinkhorn_iters=20)
        layer = Glm53DenseDecoderLayer(**_dense_kwargs(mhc_cfg), layer_idx=0).cuda()

        streams = torch.randn(1, 17, HC_MULT, HIDDEN, device="cuda")
        seq_ctx = _seq_ctx(17, "cuda")
        out = layer(hidden_states=streams, position_embeddings=(None, None), seq_ctx=seq_ctx)

        assert out["hidden_states"].shape == streams.shape
        assert torch.isfinite(out["hidden_states"]).all()

    @pytest.mark.gpu
    def test_mhc_cfg_none_matches_plain_dense_decoder_layer(self):
        """use_mhc=False must be a true no-op passthrough to the base class."""
        # mhc_cfg=None 必须是对基类的无损透传（MTP 层就用这条路径）。
        torch.manual_seed(0)
        glm_layer = Glm53DenseDecoderLayer(**_dense_kwargs(None), layer_idx=0).cuda()
        torch.manual_seed(0)
        base_layer = DenseDecoderLayer(
            **{k: v for k, v in _dense_kwargs(None).items() if k != "mhc_cfg"}, layer_idx=0
        ).cuda()

        x = torch.randn(1, 11, HIDDEN, device="cuda")
        seq_ctx = _seq_ctx(11, "cuda")
        glm_out = glm_layer(hidden_states=x, position_embeddings=(None, None), seq_ctx=seq_ctx)
        base_out = base_layer(hidden_states=x, position_embeddings=(None, None), seq_ctx=seq_ctx)
        torch.testing.assert_close(glm_out["hidden_states"], base_out["hidden_states"])

    @pytest.mark.gpu
    def test_grad_flows_through_hc_params(self):
        # 梯度必须到达每层两组 hc_* 参数。
        torch.manual_seed(0)
        mhc_cfg = MHCConfig(hc_mult=HC_MULT)
        layer = Glm53DenseDecoderLayer(**_dense_kwargs(mhc_cfg), layer_idx=0).cuda()

        streams = torch.randn(1, 9, HC_MULT, HIDDEN, device="cuda")
        seq_ctx = _seq_ctx(9, "cuda")
        out = layer(hidden_states=streams, position_embeddings=(None, None), seq_ctx=seq_ctx)
        out["hidden_states"].sum().backward()

        assert layer.hc_attn_fn.grad is not None and torch.isfinite(layer.hc_attn_fn.grad).all()
        assert layer.hc_ffn_fn.grad is not None and torch.isfinite(layer.hc_ffn_fn.grad).all()


class TestGlm53MoEDecoderLayer:
    # The routed-expert grouped-GEMM kernel only supports bf16/fp16 inputs, so these tests
    # (unlike the dense-layer ones, which stay fp32) build the layer directly in bf16 --
    # matching the actual production dtype for this sub-block.
    @pytest.mark.gpu
    def test_forward_finite_and_shape_preserving(self):
        torch.manual_seed(0)
        mhc_cfg = MHCConfig(hc_mult=HC_MULT, hc_eps=1e-6, hc_sinkhorn_iters=20)
        layer = Glm53MoEDecoderLayer(**_moe_kwargs(mhc_cfg), layer_idx=0).cuda().to(torch.bfloat16)
        _init_uninitialized_moe_params(layer)

        streams = torch.randn(1, 13, HC_MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
        seq_ctx = _seq_ctx(13, "cuda")
        out = layer(hidden_states=streams, position_embeddings=(None, None), seq_ctx=seq_ctx)

        assert out["hidden_states"].shape == streams.shape
        assert torch.isfinite(out["hidden_states"]).all()

    @pytest.mark.gpu
    def test_mhc_cfg_none_matches_plain_moe_decoder_layer(self):
        """use_mhc=False must be a true no-op passthrough to the base class -- this is the
        code path the MTP block (F6) reuses, since checkpoint layer 45 has no hc_* params."""
        # MoE 版的 mhc_cfg=None 同样必须等价于基类。
        torch.manual_seed(0)
        glm_layer = Glm53MoEDecoderLayer(**_moe_kwargs(None), layer_idx=0).cuda().to(torch.bfloat16)
        _init_uninitialized_moe_params(glm_layer)
        torch.manual_seed(0)
        base_layer = (
            MoEDecoderLayer(**{k: v for k, v in _moe_kwargs(None).items() if k != "mhc_cfg"}, layer_idx=0)
            .cuda()
            .to(torch.bfloat16)
        )
        _init_uninitialized_moe_params(base_layer)

        x = torch.randn(1, 15, HIDDEN, device="cuda", dtype=torch.bfloat16)
        seq_ctx = _seq_ctx(15, "cuda")
        glm_out = glm_layer(hidden_states=x, position_embeddings=(None, None), seq_ctx=seq_ctx)
        base_out = base_layer(hidden_states=x, position_embeddings=(None, None), seq_ctx=seq_ctx)
        torch.testing.assert_close(glm_out["hidden_states"], base_out["hidden_states"])

    @pytest.mark.gpu
    def test_grad_flows_through_hc_params_and_experts(self):
        # 梯度必须同时到达 hc_* 参数与路由专家。
        torch.manual_seed(0)
        mhc_cfg = MHCConfig(hc_mult=HC_MULT)
        layer = Glm53MoEDecoderLayer(**_moe_kwargs(mhc_cfg), layer_idx=0).cuda().to(torch.bfloat16)
        _init_uninitialized_moe_params(layer)

        streams = torch.randn(1, 7, HC_MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
        seq_ctx = _seq_ctx(7, "cuda")
        out = layer(hidden_states=streams, position_embeddings=(None, None), seq_ctx=seq_ctx)
        out["hidden_states"].sum().backward()

        assert layer.hc_attn_fn.grad is not None and torch.isfinite(layer.hc_attn_fn.grad).all()
        assert layer.hc_ffn_fn.grad is not None and torch.isfinite(layer.hc_ffn_fn.grad).all()
        assert layer.experts.fused_w1w3.weight.grad is not None


class TestGlm53DecoderLayerCompile:
    """torch.compile 下含 KDA 的 decoder 层。"""

    @pytest.mark.gpu
    def test_dense_layer_forward_compiles_with_dynamic_cu_seqlens(self):
        # 训练把 cu_seq_lens 标记为 dynamic 以复用计算图，这让 FLA 内部的
        # cu_seqlens.tolist() 成为数据依赖算子、inductor 无法 lower；KDA 的 FLA 入口
        # 必须对 dynamo 不可见，否则任何含 KDA 的编译区都会整块编译失败。
        layer = Glm53DenseDecoderLayer(**_dense_kwargs(MHCConfig(hc_mult=HC_MULT, hc_sinkhorn_iters=2)), layer_idx=0)
        layer = layer.cuda().to(torch.bfloat16)
        seq_len = 128
        streams = torch.randn(1, seq_len, HC_MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
        seq_ctx = _seq_ctx(seq_len, "cuda")
        torch._dynamo.mark_dynamic(seq_ctx.cu_seq_lens_q, 0)
        torch._dynamo.mark_dynamic(seq_ctx.cu_seq_lens_k, 0)

        compiled = torch.compile(Glm53DenseDecoderLayer._forward, fullgraph=False)
        out = compiled(layer, streams, position_embeddings=(None, None), seq_ctx=seq_ctx)

        assert out.shape == streams.shape
        assert torch.isfinite(out).all()
