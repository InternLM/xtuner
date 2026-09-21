# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash decoder layers: mHC-wrapped dense (KDA) and MoE (NoPE-DSA)
layers.

Both classes wrap their base class's attention/FFN sub-blocks with the
:mod:`xtuner.v1.module.decoder_layer.mhc` ``hc_pre`` / sub_block / ``hc_post`` pattern
instead of the plain pre-norm residual (see doc/xtuner_glm5p3flash_design.md F4). The
``mhc_cfg=None`` case degrades to the base class's ordinary residual math unchanged --
used by the MTP layer (F6), whose checkpoint has no ``hc_*`` parameters.

:class:`Glm53MoEDecoderLayer` only overrides :meth:`_pre_moe_forward` /
:meth:`_post_moe_forward`, the seams the base :class:`MoEDecoderLayer` already exposes for
the attention+gate and combine+residual phases respectively. This keeps the ~400-line
EP/dispatcher/domino-micro-batch pipeline in ``_forward`` / ``_micro_batch_forward``
completely untouched: those methods pass whatever ``_pre_moe_forward`` returns as
``residual`` opaquely through to ``_post_moe_forward``, so ``_MHCResidual`` (a 4-stream
tensor + the per-call ``post``/``comb`` weights) rides along for free without any extra
instance-attribute state or change to the base class.
"""

from functools import partial
from typing import Callable, NamedTuple, cast

import torch
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import AttnOutputs, RouterResults
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.mhc import MHCConfig, hc_post, hc_pre
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayer,
    _prepare_rollout_routed_experts_for_router,
)
from xtuner.v1.utils import ForwardState
from xtuner.v1.utils.dtensor import materialize_full
from xtuner.v1.utils.init_weight import init_params


class _MHCResidual(NamedTuple):
    """Opaque payload threaded as ``residual`` between :meth:`_pre_moe_forward` and
    :meth:`_post_moe_forward` -- see module docstring."""

    streams: torch.Tensor  # [B, S, hc_mult, hidden_size], saved before the FFN-site hc_pre
    post: torch.Tensor
    comb: torch.Tensor


def _register_hc_params(module: torch.nn.Module, *, site: str, mhc_cfg: MHCConfig, hidden_size: int) -> None:
    mix = (2 + mhc_cfg.hc_mult) * mhc_cfg.hc_mult
    hc_dim = mhc_cfg.hc_mult * hidden_size
    # fp32: the 20-iteration Sinkhorn loop is bf16-NaN-prone (design doc 3.5.3). Declaring the
    # dtype here is not what keeps it -- `fully_shard` upcasts every trainable parameter to an
    # fp32 master and then casts to `MixedPrecisionPolicy.param_dtype` for the forward. What
    # keeps `base`/`scale` in fp32 compute is `Glm53TextMoEConfig.hf_save_cfg.fp32_keys_pattern`,
    # which excludes them from FSDP entirely; `fn` is intentionally not pinned (`hc_pre` casts
    # it to the activation dtype anyway).
    module.register_parameter(f"hc_{site}_fn", torch.nn.Parameter(torch.zeros(mix, hc_dim, dtype=torch.float32)))
    module.register_parameter(f"hc_{site}_base", torch.nn.Parameter(torch.zeros(mix, dtype=torch.float32)))
    module.register_parameter(f"hc_{site}_scale", torch.nn.Parameter(torch.ones(3, dtype=torch.float32)))


def _init_hc_params(module: torch.nn.Module) -> None:
    """Initialize the ``hc_*`` parameters ``default_init_weights`` cannot reach
    by name.

    Mirrors the reference ``Glm5NextHyperConnection.init_weights``: a normal-initialized mixing
    projection, a zero bias, and unit per-sub-block scales (so an untrained layer starts from
    an even stream mix). ``register_parameter``'s ``torch.zeros``/``ones`` above are lost when
    the layer is built on the meta device, so this is not a duplicate of them.
    """
    for site in ("attn", "ffn"):
        init_params(getattr(module, f"hc_{site}_fn"), partial(torch.nn.init.normal_, mean=0.0, std=0.02))
        init_params(getattr(module, f"hc_{site}_base"), torch.nn.init.zeros_)
        init_params(getattr(module, f"hc_{site}_scale"), torch.nn.init.ones_)


def _unshard_hc_site(module: torch.nn.Module, site: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize one ``hc_{site}_*`` triple as plain tensors for
    :func:`hc_pre`.

    Two reasons this happens here rather than inside ``hc_pre``: under expert parallelism the
    parameters are still ``Replicate`` DTensors after FSDP unshards them (see
    :func:`~xtuner.v1.utils.dtensor.materialize_full`), and doing it outside the compile
    boundary avoids one graph break per parameter.

    ``register_parameter`` attaches ``hc_{site}_fn`` etc. dynamically, so ``self.hc_attn_fn``
    resolves through ``nn.Module.__getattr__`` (return type ``Tensor | Module``) rather than
    as a declared attribute -- cast once here instead of at every call site.
    """
    return tuple(  # type: ignore[return-value]
        materialize_full(cast(torch.Tensor, getattr(module, f"hc_{site}_{suffix}")), name=f"hc_{site}_{suffix}")
        for suffix in ("fn", "scale", "base")
    )


class Glm53DenseDecoderLayer(DenseDecoderLayer):
    """Dense decoder layer (KDA + limited-SwiGLU dense MLP), mHC-wrapped at
    both sites.

    Covers GLM-5.3-Flash's first ``first_k_dense_replace`` layers.
    """

    def __init__(self, *, mhc_cfg: MHCConfig | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mhc_cfg = mhc_cfg
        self.use_mhc = mhc_cfg is not None
        if self.use_mhc:
            assert mhc_cfg is not None
            _register_hc_params(self, site="attn", mhc_cfg=mhc_cfg, hidden_size=self.hidden_size)
            _register_hc_params(self, site="ffn", mhc_cfg=mhc_cfg, hidden_size=self.hidden_size)

    def init_weights(self) -> None:
        if self.use_mhc:
            _init_hc_params(self)

    @override
    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        if not self.use_mhc:
            return super()._forward(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        assert self.mhc_cfg is not None

        # ---- attention site: hidden_states is [B, S, hc_mult, hidden_size]
        fn, scale, base = _unshard_hc_site(self, "attn")
        residual = hidden_states
        x, post, comb = hc_pre(
            hidden_states, fn, scale, base, self.mhc_cfg.hc_mult, self.mhc_cfg.hc_sinkhorn_iters, self.mhc_cfg.hc_eps
        )
        attn_outputs: AttnOutputs = self.self_attn(
            hidden_states=self.input_layernorm(x), position_embeddings=position_embeddings, seq_ctx=seq_ctx
        )
        hidden_states = hc_post(attn_outputs["projected_output"], residual, post, comb)

        # ---- ffn site
        fn, scale, base = _unshard_hc_site(self, "ffn")
        residual = hidden_states
        x, post, comb = hc_pre(
            hidden_states, fn, scale, base, self.mhc_cfg.hc_mult, self.mhc_cfg.hc_sinkhorn_iters, self.mhc_cfg.hc_eps
        )
        ffn_out = self.mlp(self.post_attention_layernorm(x))
        hidden_states = hc_post(ffn_out, residual, post, comb)

        return hidden_states


class Glm53MoEDecoderLayer(MoEDecoderLayer):
    """MoE decoder layer (NoPE-DSA / KDA attention + routed MoE), mHC-wrapped
    at both sites.

    ``mhc_cfg=None`` builds the plain (non-mHC) layer the MTP block reuses this same class
    for (design doc F6): checkpoint layer 45 has no ``hc_*`` parameters.
    """

    def __init__(self, *, mhc_cfg: MHCConfig | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mhc_cfg = mhc_cfg
        self.use_mhc = mhc_cfg is not None
        if self.use_mhc:
            assert mhc_cfg is not None
            _register_hc_params(self, site="attn", mhc_cfg=mhc_cfg, hidden_size=self.hidden_size)
            _register_hc_params(self, site="ffn", mhc_cfg=mhc_cfg, hidden_size=self.hidden_size)

    def init_weights(self) -> None:
        if self.use_mhc:
            _init_hc_params(self)

    @override
    def _pre_moe_forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: ForwardState,
        past_key_values: list[list[torch.Tensor]] | None = None,
        attention_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor | _MHCResidual, torch.Tensor, RouterResults, AttnOutputs]:
        # `residual`'s declared type in the base class is `Tensor`; widening it to
        # `Tensor | _MHCResidual` is intentional (see module docstring) -- the base class's
        # `_forward` / `_micro_batch_forward` only ever pass it through opaquely to
        # `_post_moe_forward`, never inspecting it, so this is a safe covariant extension of
        # the informal "opaque payload" contract, not a real Liskov violation.
        if not self.use_mhc:
            return super()._pre_moe_forward(
                hidden_states,
                seq_ctx,
                position_embeddings,
                state,
                past_key_values=past_key_values,
                attention_kwargs=attention_kwargs,
            )
        assert self.mhc_cfg is not None
        assert state == ForwardState.TRAINING, "mHC-wrapped GLM-5.3-Flash decoder layers only support SFT training"

        # ---- attention site: hidden_states is [B, S, hc_mult, hidden_size]
        fn, scale, base = _unshard_hc_site(self, "attn")
        attn_residual = hidden_states
        x, post, comb = hc_pre(
            hidden_states, fn, scale, base, self.mhc_cfg.hc_mult, self.mhc_cfg.hc_sinkhorn_iters, self.mhc_cfg.hc_eps
        )
        attention_forward = cast(Callable[..., AttnOutputs], self.self_attn)
        attn_outputs = attention_forward(
            hidden_states=self.input_layernorm(x),
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            **(attention_kwargs or {}),
        )
        hidden_states = hc_post(attn_outputs["projected_output"], attn_residual, post, comb)

        # ---- ffn site
        fn, scale, base = _unshard_hc_site(self, "ffn")
        ffn_residual = hidden_states
        x, post, comb = hc_pre(
            hidden_states, fn, scale, base, self.mhc_cfg.hc_mult, self.mhc_cfg.hc_sinkhorn_iters, self.mhc_cfg.hc_eps
        )
        hidden_states = self.post_attention_layernorm(x)

        if seq_ctx.rollout_routed_experts is not None and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]
            rollout_routed_experts = _prepare_rollout_routed_experts_for_router(
                rollout_routed_experts,
                hidden_states,
                offload_rollout_routed_experts=seq_ctx.offload_rollout_routed_experts,
            )
        else:
            rollout_routed_experts = None
        router_results: RouterResults = self.gate(hidden_states, rollout_routed_experts)

        return _MHCResidual(ffn_residual, post, comb), hidden_states, router_results, attn_outputs

    @override
    def _post_moe_forward(
        self,
        combined_hidden_states: torch.Tensor,
        residual: torch.Tensor | _MHCResidual,
        shared_experts_out: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_mhc:
            assert isinstance(residual, torch.Tensor)
            return super()._post_moe_forward(combined_hidden_states, residual, shared_experts_out)
        assert isinstance(residual, _MHCResidual)

        if self.n_shared_experts > 0:
            assert shared_experts_out is not None
            combined_hidden_states = combined_hidden_states + shared_experts_out
        combined_hidden_states = combined_hidden_states * self.hidden_factor
        return hc_post(combined_hidden_states, residual.streams, residual.post, residual.comb)
