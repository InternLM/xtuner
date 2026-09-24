# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash text model: 45-layer KDA/NoPE-DSA stack with mHC four-stream
residual.

See doc/xtuner_glm5p3flash_design.md F6. Layer schedule (checkpoint ``text_config``):
``layer_types`` alternates ``[KDA, KDA, KDA, DSA]`` x 11 + a final KDA layer (45 layers total);
``mlp_layer_types`` is dense for the first 3 (``first_k_dense_replace``) and MoE (288 routed
experts) for the rest. Every main-stack layer -- KDA or DSA, dense or MoE -- is mHC-wrapped
(design doc F4); only the MTP layer (checkpoint ``layers.45``) has no ``hc_*`` params and runs
a plain pre-norm residual, achieved simply by building it with ``mhc_cfg=None`` (see
``xtuner/v1/model/moe/glm53/decoder_layer.py`` module docstring). Because every layer's
``indexer_types`` is ``"full"`` (12 independent indexers: 11 main-stack DSA layers + 1 MTP,
confirmed against the real checkpoint), GLM-5.3-Flash needs none of GLM-5.2's cross-layer
``dsa_topk_ids`` IndexShare machinery -- the base ``MoE._call_decoder_layer`` /
``MTPLayer``/``MTPBlock`` are reused unmodified.

mHC makes the *residual stream itself* four-way: ``embed_tokens`` output is expanded once to
``[B, S, hc_mult, hidden_size]`` before the first layer, each layer's ``hc_pre``/``hc_post``
collapse-computes-reexpands internally (already implemented per-layer in ``decoder_layer.py``),
and the stream is unweighted-mean-collapsed back to ``[B, S, hidden_size]`` once after the last
layer -- so ``self.norm``/``lm_head``/MTP never see the 4-stream shape (design doc F6, confirmed
via a peer review of this exact expand/collapse boundary).
"""

import re
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn as nn
from pydantic import Field, computed_field
from typing_extensions import Self, override

from xtuner.v1.model.base import DEFAULT_FLOAT8_CFG, HFSaveCfg, TorchCompileOption
from xtuner.v1.model.moe.moe import MoE, MoEConfig
from xtuner.v1.module.decoder_layer.mhc import MHCConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .decoder_layer import Glm53DenseDecoderLayer, Glm53MoEDecoderLayer
from .nope_dsa_mla import NoPEDSAMLAConfig


try:
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig as HFGlm5NextTextConfig
except ImportError:
    HFGlm5NextTextConfig = None  # type: ignore[misc, assignment]

from xtuner.v1.module.attention.kda import KDAConfig


# KDA (fla's fused_kda_gate/chunk_kda) and NoPE-DSA (custom SparseMLA/indexer kernels) both
# contain ops torch.compile's dynamo frontend cannot trace, so their call sites must be
# fullgraph=False graph-break boundaries -- confirmed by a real 8-GPU run crashing with
# `torch._dynamo.exc.Unsupported: Skip calling torch.compiler.disable()d function` on
# fused_kda_gate under the default (empty) compile_cfg; fullgraph=False lets dynamo compile
# everything around the one break instead of falling back to eager for the whole method
# (design doc F6's compile row).
#
# hc_pre/hc_post must stay INSIDE a compiled region -- hc_pre's fp32 rms-norm intermediates and
# _hc_post_eager's broadcast-multiply are only affordable once inductor fuses them. They are
# registered on their own rather than riding along inside the enclosing method's boundary,
# because the EP table below drops _pre_moe_forward/_post_moe_forward and would otherwise leave
# 42 of 45 layers running them eagerly under the production topology.
GLM53_MOE_NON_EP_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.mhc.hc_pre": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.mhc._hc_post_eager": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.moe.glm53.decoder_layer.Glm53MoEDecoderLayer._pre_moe_forward": TorchCompileOption(
        fullgraph=False
    ),
    "xtuner.v1.model.moe.glm53.decoder_layer.Glm53MoEDecoderLayer._post_moe_forward": TorchCompileOption(
        fullgraph=False
    ),
    "xtuner.v1.module.attention.kda.KimiDeltaAttention.forward": TorchCompileOption(fullgraph=False),
    "xtuner.v1.model.moe.glm53.nope_dsa_mla.NoPEDSAMultiLatentAttention.forward": TorchCompileOption(fullgraph=False),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._shared_experts_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.model.moe.glm53.decoder_layer.Glm53DenseDecoderLayer._forward": TorchCompileOption(fullgraph=False),
    **DEFAULT_FLOAT8_CFG,
}

# Under EP, drop the whole MoE-layer compile boundary (all2all dispatch doesn't trace safely),
# matching GLM-5.2's convention.
GLM53_MOE_EP_COMPILE_CFG = GLM53_MOE_NON_EP_COMPILE_CFG.copy()
GLM53_MOE_EP_COMPILE_CFG.pop("xtuner.v1.model.moe.glm53.decoder_layer.Glm53MoEDecoderLayer._pre_moe_forward")
GLM53_MOE_EP_COMPILE_CFG.pop("xtuner.v1.model.moe.glm53.decoder_layer.Glm53MoEDecoderLayer._post_moe_forward")


class Glm53TextMoE(MoE):
    # mHC's extra required `mhc_cfg` keyword makes these constructors intentionally incompatible
    # with the base MoE.build_layers()/build_mtp_block() generic calling convention -- both are
    # fully overridden below rather than relied on generically, so this is correct, not a bug.
    #
    # NOTE: those two overrides are line-for-line copies of the base class's, differing only by
    # the `mhc_cfg` they pass. Any change to `MoE.build_layers` / `MoE.build_mtp_block` (a new
    # config field, a changed mesh argument) must be mirrored here, or this model silently keeps
    # building layers the old way.
    dense_decoder_layer_cls = Glm53DenseDecoderLayer  # type: ignore[assignment]
    moe_decoder_layer_cls = Glm53MoEDecoderLayer  # type: ignore[assignment]

    config: "Glm53TextMoEConfig"

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        if self.config.ep_size > 1:
            return GLM53_MOE_EP_COMPILE_CFG
        return GLM53_MOE_NON_EP_COMPILE_CFG

    def _expand_hc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hc_mult = self.config.mhc.hc_mult
        return (
            hidden_states.unsqueeze(-2)
            .expand(*hidden_states.shape[:-1], hc_mult, hidden_states.shape[-1])
            .contiguous()
        )

    @staticmethod
    def _collapse_hc(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.mean(dim=-2)

    @override
    def _decoder_stack(self, *, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self._expand_hc(hidden_states)
        hidden_states = super()._decoder_stack(hidden_states=hidden_states, **kwargs)
        return self._collapse_hc(hidden_states)

    @override
    def _micro_batch_decoder_stack(self, *, hidden_states_list: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        hidden_states_list = [self._expand_hc(h) for h in hidden_states_list]
        hidden_states_list = super()._micro_batch_decoder_stack(hidden_states_list=hidden_states_list, **kwargs)
        return [self._collapse_hc(h) for h in hidden_states_list]

    @override
    def build_layers(self, config: "Glm53TextMoEConfig") -> nn.ModuleDict:  # type: ignore[override]
        from xtuner.v1.model.utils import module_dict_repr

        layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layers_type[layer_idx]
            attention_config = config.linear_attention if layer_type == "linear_attention" else config.attention
            assert attention_config is not None
            common = dict(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mlp_bias=config.mlp_bias,
                hidden_act=config.hidden_act,
                swiglu_limit=config.swiglu_limit,
                rms_norm_eps=config.rms_norm_eps,
                rms_norm_type=config.rms_norm_type,
                attention_config=attention_config,
                layer_type=layer_type,
                rope_scaling_cfg=config.rope_scaling_cfg,
                generate_config=config.generate_config,
                float8_cfg=config.float8_cfg,
                layer_idx=layer_idx,
                mhc_cfg=config.mhc,
            )
            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = self.dense_decoder_layer_cls(**common)  # type: ignore[arg-type]
            else:
                layers[str(layer_idx)] = self.moe_decoder_layer_cls(
                    **common,  # type: ignore[arg-type]
                    moe_intermediate_size=config.moe_intermediate_size,
                    gate_bias=config.gate_bias,
                    moe_bias=config.moe_bias,
                    num_experts_per_tok=config.num_experts_per_tok,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    with_shared_expert_gate=config.with_shared_expert_gate,
                    hidden_factor=config.hidden_factor,
                    router_config=config.router,
                    router_compute_dtype=config.router_compute_dtype,
                    moe_act_fn_cfg=config.moe_act_fn_cfg,
                    dispatcher=config.dispatcher,
                    ep_mesh=self.ep_mesh,
                    expert_tp_mesh=self.expert_tp_mesh,
                    ep_tp_mesh=self.ep_tp_mesh,
                )
                if self.config.freeze_routers:
                    layers[str(layer_idx)].gate.requires_grad_(False)
                    layers[str(layer_idx)].gate.eval()

        layers.__class__.__repr__ = module_dict_repr  # type: ignore[method-assign]
        return layers

    @override
    def build_mtp_block(self, config: "Glm53TextMoEConfig"):  # type: ignore[override]
        from xtuner.v1.module.mtp import MTPBlock, MTPLayer

        mtp_config = config.mtp_config
        assert mtp_config is not None
        assert mtp_config.share_weights and mtp_config.num_layers == 1, (
            "GLM-5.3-Flash checkpoint has exactly one physical MTP layer (layers.45)."
        )
        decoder_layer = self.moe_decoder_layer_cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            mlp_bias=config.mlp_bias,
            gate_bias=config.gate_bias,
            moe_bias=config.moe_bias,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
            rms_norm_eps=config.rms_norm_eps,
            rms_norm_type=config.rms_norm_type,
            num_experts_per_tok=config.num_experts_per_tok,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            with_shared_expert_gate=config.with_shared_expert_gate,
            hidden_factor=config.hidden_factor,
            layer_type="full_attention",
            attention_config=config.attention,
            rope_scaling_cfg=config.rope_scaling_cfg,
            generate_config=config.generate_config,
            router_config=config.router,
            router_compute_dtype=config.router_compute_dtype,
            moe_act_fn_cfg=config.moe_act_fn_cfg,
            float8_cfg=config.float8_cfg,
            layer_idx=config.num_hidden_layers,
            dispatcher=config.dispatcher,
            ep_mesh=self.ep_mesh,
            expert_tp_mesh=self.expert_tp_mesh,
            ep_tp_mesh=self.ep_tp_mesh,
            mhc_cfg=None,  # checkpoint layers.45 has no hc_* params (design doc F6).
        )
        mtp_layer = MTPLayer(
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            rms_norm_type=config.rms_norm_type,
            decoder_layer=decoder_layer,
            float8_cfg=config.float8_cfg,
        )
        return MTPBlock(mtp_config=mtp_config, mtp_layers=[mtp_layer])

    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if key.startswith("mtp_block."):
            match = re.match(r"mtp_block\.layers\.0\.(.+)", key)
            assert match is not None, f"Unexpected GLM-5.3-Flash MTP key: {key}"
            key = f"layers.{self.config.num_hidden_layers}.{match.group(1)}"
            key = key.replace(".decoder_layer.", ".")
            key = re.sub(r"layers\.(\d+)\.final_layernorm\.", r"layers.\1.shared_head.norm.", key)

        if "layers" in key or "embed_tokens" in key:
            key = "model.language_model." + key
        elif key.startswith("norm."):
            return [key.replace("norm.", "model.language_model.norm.")]

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts)", r"layers.\1.mlp.\2", key)

        if "fused_w1w3.weight" in key:
            return [
                key.replace("fused_w1w3.weight", f"{i}.{proj}_proj.weight")
                for i in range(self.config.n_routed_experts)
                for proj in ("gate", "up")
            ]
        if "fused_w2.weight" in key:
            return [
                key.replace("fused_w2.weight", f"{i}.down_proj.weight") for i in range(self.config.n_routed_experts)
            ]
        if "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        return [key]

    def hf_tensor_to_canonical(self, name: str, loaded_tensor: torch.Tensor) -> torch.Tensor:
        if ("fused_w1w3.weight" in name or "fused_w2.weight" in name) and loaded_tensor.ndim == 3:
            loaded_tensor = loaded_tensor.flatten(0, 1)
        return loaded_tensor

    def param_to_safetensor(self, safetensor: torch.Tensor, hf_param_name: str):
        assert isinstance(hf_param_name, str)
        if hf_param_name.endswith("experts.gate_up_proj"):
            safetensor = safetensor.reshape(self.config.n_routed_experts, -1, self.config.hidden_size)
        elif hf_param_name.endswith("experts.down_proj"):
            safetensor = safetensor.reshape(self.config.n_routed_experts, self.config.hidden_size, -1)
        return safetensor


class Glm53TextMoEConfig(MoEConfig):
    model_type: str = "glm5_next_text"
    vocab_size: int = 154880
    max_position_embeddings: int = 1048576
    pad_token_id: int | None = 154820
    eos_token_id: int = 154820
    hf_eos_token_id: int | list[int] = Field(default_factory=lambda: [154820, 154827, 154829])
    num_hidden_layers: int = 45
    first_k_dense_replace: int = 3
    hidden_size: int = 4096
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    # Clamped SwiGLU applies to *every* FFN: dense first-k layers and shared experts take it as
    # `swiglu_limit` (separate gate/up projections), the routed experts take it through
    # `moe_act_fn_cfg` (one fused gate_up projection). Both must be set -- HF clamps in
    # `Glm5NextTextMLP` and `Glm5NextTextExperts._apply_gate` alike.
    swiglu_limit: float = 10.0
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(act_type="clamped_swiglu", clip_limit=10.0)

    # NoPE: no rotary position embedding is applied by either attention type (KDA has none by
    # construction; NoPE-DSA sets qk_rope_head_dim=0). rotary_emb is still built and computed
    # unconditionally by the base MoE stack but never consumed -- see decoder_layer.py.
    attention: NoPEDSAMLAConfig = NoPEDSAMLAConfig(
        num_attention_heads=64,
        head_dim=256,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        v_head_dim=256,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_kpool=4,
    )
    linear_attention: KDAConfig | None = KDAConfig(num_heads=64, head_dim=128, gate_lower_bound=-5.0)

    tie_word_embeddings: bool = False
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 2048
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=1, topk_group=1, scoring_func="sigmoid", norm_topk_prob=True, router_scaling_factor=2.5
    )
    num_nextn_predict_layers: int | None = 1
    mtp_config: MTPConfig | None = MTPConfig(num_layers=1, share_weights=True)

    mhc: MHCConfig = MHCConfig(hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20)

    # `fully_shard` upcasts every trainable parameter to an fp32 master and then casts it to
    # `MixedPrecisionPolicy.param_dtype` for the forward all-gather, so declaring a parameter
    # `dtype=torch.float32` does not keep it in fp32 *compute*. This pattern list is the only
    # lever that does: `BaseModel._fully_shard` hands the matches to
    # `fully_shard(ignored_params=...)`, leaving them replicated and untouched by the policy
    # (and `_get_save_dtype` then writes them back out as fp32). Needed because
    # `hc_split_sinkhorn`'s 20 iterations and KDA's `fused_kda_gate` are bf16-unstable.
    #
    # `hc_*_fn` is deliberately absent: `hc_pre` already casts it to the activation dtype, and
    # ignored parameters are replicated rather than sharded, so pinning a
    # [mix, hc_mult * hidden_size] matrix on 45 x 2 sites would cost real memory for nothing.
    #
    # Gradients of ignored (replicated) parameters are all-reduced by
    # `MoE.scale_and_reduce_grad`, which `TrainEngine.clip_grad_norm` calls before the grad
    # norm -- PyTorch does not sync them, since `to_local()`/`full_tensor()` label the local
    # gradient `Replicate` without communicating.
    hf_save_cfg: HFSaveCfg = HFSaveCfg(
        fp32_keys_pattern=[
            r"model\.language_model\.layers\.\d+\.hc_(attn|ffn)_(base|scale)",
            r"model\.language_model\.layers\.\d+\.self_attn\.A_log",
            r"model\.language_model\.layers\.\d+\.self_attn\.dt_bias",
        ],
    )

    @computed_field
    def num_key_value_heads(self) -> int:
        # NoPEDSAMLAConfig (absorbed MLA) has no separate KV-head count, same as GLM-5.2's
        # DSAMLAConfig; base TransformerConfig.num_key_value_heads reads self.attention.
        # num_key_value_heads, which doesn't exist on this attention config.
        return self.attention.num_attention_heads

    # checkpoint text_config.layer_types: "linear_attention" x34 + "deepseek_sparse_attention"
    # x11, pattern [KDA,KDA,KDA,DSA]*11 + KDA. "deepseek_sparse_attention" maps to XTuner's
    # generic "full_attention" bucket (build_layers dispatches on config.attention for it).
    glm53_layer_types: list[Literal["linear_attention", "deepseek_sparse_attention"]] = Field(
        default_factory=lambda: (["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * 11 + ["linear_attention"]
    )

    @property
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention", "linear_attention"]]:  # type: ignore[override]
        return ["linear_attention" if t == "linear_attention" else "full_attention" for t in self.glm53_layer_types]

    def build(self) -> Glm53TextMoE:
        assert len(self.glm53_layer_types) == self.num_hidden_layers
        return Glm53TextMoE(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        assert HFGlm5NextTextConfig is not None, "transformers must be pinned to 5.17.0 (glm5_next)."
        cfg = HFGlm5NextTextConfig.from_pretrained(hf_path)
        assert isinstance(cfg, HFGlm5NextTextConfig)

        layer_types = cast(list[str], cfg.layer_types)
        return cls(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=getattr(cfg, "pad_token_id", None),
            eos_token_id=cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id,
            hf_eos_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            first_k_dense_replace=cfg.first_k_dense_replace,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            model_type=cfg.model_type,
            hidden_act=cfg.hidden_act,
            swiglu_limit=cfg.swiglu_limit,
            moe_act_fn_cfg=MoEActFnConfig(act_type="clamped_swiglu", clip_limit=cfg.swiglu_limit),
            attention=NoPEDSAMLAConfig(
                num_attention_heads=cfg.num_attention_heads,
                head_dim=cfg.qk_head_dim,
                kv_lora_rank=cfg.kv_lora_rank,
                q_lora_rank=cfg.q_lora_rank,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                v_head_dim=cfg.v_head_dim,
                qkv_bias=cfg.attention_bias,
                o_bias=cfg.attention_bias,
                dropout=cfg.attention_dropout,
                index_topk=cfg.index_topk,
                index_head_dim=cfg.index_head_dim,
                index_n_heads=cfg.index_n_heads,
                index_kpool=cfg.index_kpool,
                index_kpool_always_select_tail=cfg.index_kpool_always_select_tail,
            ),
            linear_attention=KDAConfig(
                num_heads=cfg.linear_attn_config["num_heads"],
                head_dim=cfg.linear_attn_config["head_dim"],
                conv_kernel_size=cfg.linear_attn_config["short_conv_kernel_size"],
                gate_lower_bound=cfg.linear_attn_config["gate_lower_bound"],
                rms_norm_eps=cfg.rms_norm_eps,
            ),
            glm53_layer_types=list(layer_types),  # type: ignore[arg-type]
            tie_word_embeddings=cfg.tie_word_embeddings,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            hidden_factor=1.0,
            moe_intermediate_size=cfg.moe_intermediate_size,
            router=NoAuxRouterConfig(
                n_group=cfg.n_group,
                topk_group=cfg.topk_group,
                scoring_func=cfg.scoring_func,
                norm_topk_prob=cfg.norm_topk_prob,
                router_scaling_factor=cfg.routed_scaling_factor,
            ),
            mhc=MHCConfig(hc_mult=cfg.hc_mult, hc_eps=cfg.hc_eps, hc_sinkhorn_iters=cfg.hc_sinkhorn_iters),
            num_nextn_predict_layers=getattr(cfg, "num_nextn_predict_layers", None),
            mtp_config=MTPConfig(num_layers=cfg.num_nextn_predict_layers, share_weights=True)
            if getattr(cfg, "num_nextn_predict_layers", 0)
            else None,
        )

    @property
    def hf_config(self):
        return None
