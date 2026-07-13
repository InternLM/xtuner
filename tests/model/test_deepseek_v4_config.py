# Copyright (c) OpenMMLab. All rights reserved.
"""Regression tests for :meth:`DeepSeekV4Config.from_hf` field derivation.

Every test here guards one field that XTuner has to *derive* rather than copy, because the HF
config either does not expose it (router grouping, YaRN ``truncate``) or exposes it under a
schema that changed between transformers releases (``compress_ratios``, the nested
``rope_parameters``). All of them share one failure mode: the derivation silently falls back to
XTuner's own default, which happens to equal the released DeepSeek-V4-Flash value — so the bug
is invisible on the release checkpoint and only shows on a config with different hyper-params.
The fixtures below therefore deliberately use values that differ from those defaults.

Pure CPU: nothing is instantiated beyond configs and the rotary buffers.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from xtuner.v1.model.moe.deepseek_v4 import DeepSeekV4Config
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.rope.rope import DualRotaryEmbedding


# Deliberately unlike DeepSeek-V4-Flash: 4 layers instead of 43, a layer schedule that is not a
# prefix of ``_DEFAULT_COMPRESS_RATIOS``, and YaRN parameters that differ from the V4 defaults.
# A derivation that silently falls back to a default is then observable.
_LAYER_TYPES = ["compressed_sparse_attention", "sliding_attention", "heavily_compressed_attention", "sliding_attention"]
_MLP_LAYER_TYPES = ["hash_moe", "hash_moe", "moe", "moe"]
_EXPECTED_COMPRESS_RATIOS = [4, 0, 128, 0]

_HEAD_DIM = 32
_QK_ROPE = 16
_MAX_POS = 2048
_YARN_FACTOR = 8.0
_YARN_ORIGINAL_MAX_POS = 256
_SWIGLU_LIMIT = 3.0
_ROUTED_SCALING_FACTOR = 2.5


def _build_hf_config() -> DeepseekV4Config:
    """Build a toy HF DeepSeek-V4 config with non-default V4 hyper-parameters.

    Returns:
        DeepseekV4Config: Config carrying the modern (transformers >= 5.9) schema.
    """
    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=len(_LAYER_TYPES),
        num_attention_heads=4,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=_QK_ROPE / _HEAD_DIM,
        layer_types=list(_LAYER_TYPES),
        mlp_layer_types=list(_MLP_LAYER_TYPES),
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=8,
        q_lora_rank=8,
        index_topk=8,
        index_head_dim=8,
        index_n_heads=4,
        sliding_window=8,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": _YARN_FACTOR,
            "original_max_position_embeddings": _YARN_ORIGINAL_MAX_POS,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        max_position_embeddings=_MAX_POS,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
        scoring_func="sqrtsoftplus",
        swiglu_limit=_SWIGLU_LIMIT,
        num_nextn_predict_layers=1,
    )


@pytest.fixture(scope="module")
def hf_cfg() -> DeepseekV4Config:
    return _build_hf_config()


@pytest.fixture(scope="module")
def xtuner_cfg(hf_cfg: DeepseekV4Config, tmp_path_factory: pytest.TempPathFactory) -> DeepSeekV4Config:
    """``DeepSeekV4Config.from_hf`` applied to the toy config serialised by transformers.

    Going through ``save_pretrained`` (rather than hand-writing a legacy ``config.json``) is the
    point: it is the modern schema in which ``compress_ratios`` and ``num_hash_layers`` no longer
    exist and the YaRN parameters live under ``rope_parameters["compress"]``.
    """
    hf_dir: Path = tmp_path_factory.mktemp("v4_hf")
    hf_cfg.save_pretrained(hf_dir)
    return DeepSeekV4Config.from_hf(hf_dir)


class TestDeepSeekV4ConfigFromHF:
    """Field-by-field derivation from an HF config that shares no defaults with V4-Flash."""

    def test_compress_ratios_recovered_from_layer_types(self, hf_cfg: DeepseekV4Config, xtuner_cfg) -> None:
        """``compress_ratios`` survives transformers' rewrite into ``layer_types``.

        ``DeepseekV4Config.__post_init__`` consumes the flat ``compress_ratios`` vector and
        re-expresses it as ``layer_types`` + a ``compress_rates`` lookup. Reading the popped key
        alone yields ``None``, and the caller then keeps ``_DEFAULT_COMPRESS_RATIOS`` — which
        assigns the wrong attention mode to every layer.
        """
        assert not hasattr(hf_cfg, "compress_ratios"), "fixture must exercise the post-`__post_init__` schema"
        assert xtuner_cfg.rope_parameters_cfg.compress_ratios == _EXPECTED_COMPRESS_RATIOS

    def test_router_does_no_group_limited_routing(self, xtuner_cfg) -> None:
        """V4 selects experts with a plain top-k, so the group-routing knobs must be neutral.

        The V4 reference ``Gate.forward`` is ``scores.topk(topk)`` over all experts, and HF's
        ``DeepseekV4TopKRouter`` matches; the reference's ``n_groups`` attribute is ``o_groups``,
        the attention O-LoRA grouping. Pinning ``n_group=8 / topk_group=4`` confines every token
        to half the experts.
        """
        for cfg in (xtuner_cfg, DeepSeekV4Config()):
            assert cfg.router.n_group == 1
            assert cfg.router.topk_group == 1

    def test_yarn_parameters_read_from_nested_rope_parameters(self, xtuner_cfg) -> None:
        """YaRN scaling is read out of ``rope_parameters["compress"]``, not left at its default.

        transformers >= 5.9 nests V4's rope config by rope-type label (``main`` / ``compress``).
        A flat reader finds no ``factor`` / ``original_max_position_embeddings`` at the top level
        and keeps XTuner's V4 defaults (16 / 65536), which are right only for V4-Flash.
        """
        rope_cfg = xtuner_cfg.rope_parameters_cfg
        assert rope_cfg.rope_type == "yarn"
        assert rope_cfg.factor == _YARN_FACTOR
        assert rope_cfg.original_max_position_embeddings == _YARN_ORIGINAL_MAX_POS
        # HF configs never serialise `truncate`; HF's own YaRN helper defaults it to True and the
        # V4 reference floors / ceils its correction range, so V4 must not inherit
        # `RopeParametersConfig`'s `False` default.
        assert rope_cfg.truncate is True
        # `rope_theta` is the uncompressed base (`main`); `compress_rope_theta` the compressed one.
        assert rope_cfg.rope_theta == 10000.0
        assert rope_cfg.compress_rope_theta == 160000.0

    def test_scalar_fields_copied_from_hf(self, hf_cfg: DeepseekV4Config, xtuner_cfg) -> None:
        """Fields that are a direct copy stay a direct copy under the modern schema."""
        assert xtuner_cfg.num_hidden_layers == hf_cfg.num_hidden_layers
        assert xtuner_cfg.num_hash_layers == _MLP_LAYER_TYPES.count("hash_moe")
        assert xtuner_cfg.attention.qk_rope_head_dim == _QK_ROPE
        assert xtuner_cfg.attention.o_groups == hf_cfg.o_groups
        assert xtuner_cfg.attention.sliding_window == hf_cfg.sliding_window
        assert xtuner_cfg.router.router_scaling_factor == _ROUTED_SCALING_FACTOR
        assert xtuner_cfg.swiglu_limit == _SWIGLU_LIMIT
        assert xtuner_cfg.moe_act_fn_cfg.clip_limit == _SWIGLU_LIMIT
        assert xtuner_cfg.hc_cfg.hc_mult == hf_cfg.hc_mult
        assert xtuner_cfg.mtp_config is not None
        assert xtuner_cfg.mtp_config.num_layers == hf_cfg.num_nextn_predict_layers


class TestDeepSeekV4DualRope:
    """The two rope bases must reproduce HF's ``main`` / ``compress`` buffers exactly."""

    def test_inv_freq_matches_hf(self, hf_cfg: DeepseekV4Config, xtuner_cfg) -> None:
        """Dense base is plain RoPE, compressed base is YaRN, neither rescales cos/sin.

        Three independent regressions live here:

        * YaRN was applied to the dense base too. The reference builds sliding-window layers with
          ``original_seq_len=0`` (interpolation off) and HF emits ``rope_type="default"`` for
          ``main``; interpolating there moves ``inv_freq`` by up to ~85% on the low-frequency dims.
        * ``attention_scaling`` took YaRN's ``0.1·ln(factor)+1``. The reference returns
          unit-modulus ``polar(ones, freqs)`` and HF pins ``attention_factor=1.0``.
        * ``truncate`` defaulted to ``False``. Both the reference (``floor`` / ``ceil`` in
          ``find_correction_range``) and HF (``truncate`` defaults to ``True``) truncate, so the
          compressed base landed on a different interpolation ramp.
        """
        hf_rope = DeepseekV4RotaryEmbedding(hf_cfg)
        xt_rope = DualRotaryEmbedding(xtuner_cfg, device="cpu")

        assert xt_rope.attention_scaling == 1.0
        assert hf_rope.main_attention_scaling == 1.0
        assert hf_rope.compress_attention_scaling == 1.0
        torch.testing.assert_close(xt_rope.inv_freq_dense.cpu(), hf_rope.main_inv_freq)
        torch.testing.assert_close(xt_rope.inv_freq_compressed.cpu(), hf_rope.compress_inv_freq)

    def test_dense_base_is_not_interpolated(self, xtuner_cfg) -> None:
        """Guard the dense base directly, so the test still bites if HF's rope buffers move."""
        xt_rope = DualRotaryEmbedding(xtuner_cfg, device="cpu")
        plain = 1.0 / (10000.0 ** (torch.arange(0, _QK_ROPE, 2, dtype=torch.float32) / _QK_ROPE))
        torch.testing.assert_close(xt_rope.inv_freq_dense.cpu(), plain)


class TestDeepSeekV4ActFn:
    """V4's expert activation is a clamped SwiGLU, not gpt-oss's clipped SwiGLU."""

    def test_clamped_swiglu_matches_reference(self, xtuner_cfg) -> None:
        """``silu(clamp(gate)) * clamp(up)`` — no gpt-oss ``up + 1`` residual term.

        V4's ``Expert.forward`` and HF's ``DeepseekV4MLP.forward`` both apply the asymmetric
        ``swiglu_limit`` clamp and then multiply. ``clipped_swiglu`` shares the clamp but scales
        by ``up + 1``, which silently adds a whole extra ``silu(gate)`` to every expert output.
        """
        assert xtuner_cfg.moe_act_fn_cfg.act_type == "clamped_swiglu"
        act_fn = xtuner_cfg.moe_act_fn_cfg.build()

        torch.manual_seed(0)
        gate, up = torch.randn(2, 4, 16).unbind(0)
        limit = _SWIGLU_LIMIT
        expected = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)

        torch.testing.assert_close(act_fn(torch.cat([gate, up], dim=-1)), expected)

    def test_clipped_swiglu_still_carries_gpt_oss_bias(self) -> None:
        """The gpt-oss variant is untouched — the two activations stay distinct."""
        act_fn = MoEActFnConfig(act_type="clipped_swiglu", clip_alpha=1.702, clip_limit=7.0).build()

        torch.manual_seed(0)
        gate, up = torch.randn(2, 4, 16).unbind(0)
        gate_c, up_c = gate.clamp(max=7.0), up.clamp(min=-7.0, max=7.0)
        expected = (up_c + 1) * gate_c * torch.sigmoid(gate_c * 1.702)

        torch.testing.assert_close(act_fn(torch.cat([gate, up], dim=-1)), expected)


class TestDeepSeekV4LegacyConfigSchema:
    """The released V4-Flash ``config.json`` predates transformers' V4 support."""

    def test_legacy_flat_schema_agrees_with_modern_schema(
        self, hf_cfg: DeepseekV4Config, xtuner_cfg, tmp_path: Path
    ) -> None:
        """A flat legacy ``config.json`` and the modern nested one must derive the same config.

        The legacy schema ships ``compress_ratios`` / ``num_hash_layers`` as top-level keys and a
        flat ``rope_scaling``; the modern one ships ``layer_types`` / ``mlp_layer_types`` and a
        rope-type-keyed ``rope_parameters``. Both must land on the same XTuner config, otherwise
        the release checkpoint and a re-saved one train differently.
        """
        legacy = {
            "model_type": "deepseek_v4",
            "vocab_size": hf_cfg.vocab_size,
            "hidden_size": hf_cfg.hidden_size,
            "moe_intermediate_size": hf_cfg.moe_intermediate_size,
            "num_hidden_layers": hf_cfg.num_hidden_layers,
            "num_attention_heads": hf_cfg.num_attention_heads,
            "num_key_value_heads": 1,
            "head_dim": _HEAD_DIM,
            "qk_rope_head_dim": _QK_ROPE,
            "q_lora_rank": hf_cfg.q_lora_rank,
            "o_lora_rank": hf_cfg.o_lora_rank,
            "o_groups": hf_cfg.o_groups,
            "sliding_window": hf_cfg.sliding_window,
            "index_topk": hf_cfg.index_topk,
            "index_head_dim": hf_cfg.index_head_dim,
            "index_n_heads": hf_cfg.index_n_heads,
            "n_routed_experts": hf_cfg.n_routed_experts,
            "n_shared_experts": hf_cfg.n_shared_experts,
            "num_experts_per_tok": hf_cfg.num_experts_per_tok,
            "num_hash_layers": _MLP_LAYER_TYPES.count("hash_moe"),
            "compress_ratios": list(_EXPECTED_COMPRESS_RATIOS),
            "compress_rope_theta": hf_cfg.compress_rope_theta,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": _YARN_FACTOR,
                "original_max_position_embeddings": _YARN_ORIGINAL_MAX_POS,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            "max_position_embeddings": _MAX_POS,
            "hc_mult": hf_cfg.hc_mult,
            "hc_eps": hf_cfg.hc_eps,
            "hc_sinkhorn_iters": hf_cfg.hc_sinkhorn_iters,
            "rms_norm_eps": hf_cfg.rms_norm_eps,
            "scoring_func": hf_cfg.scoring_func,
            "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
            "norm_topk_prob": True,
            "swiglu_limit": _SWIGLU_LIMIT,
            "hidden_act": hf_cfg.hidden_act,
            "num_nextn_predict_layers": hf_cfg.num_nextn_predict_layers,
            "tie_word_embeddings": False,
            "eos_token_id": 1,
            "bos_token_id": 0,
        }
        (tmp_path / "config.json").write_text(json.dumps(legacy), encoding="utf-8")

        legacy_cfg = DeepSeekV4Config.from_hf(tmp_path)

        assert legacy_cfg.rope_parameters_cfg == xtuner_cfg.rope_parameters_cfg
        assert legacy_cfg.router == xtuner_cfg.router
        assert legacy_cfg.attention == xtuner_cfg.attention
        assert legacy_cfg.moe_act_fn_cfg == xtuner_cfg.moe_act_fn_cfg
