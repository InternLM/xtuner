import math
import re
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import Field
from typing_extensions import Self

from transformers.models.auto import AutoConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.moe import (
    MoE,
    MoEConfig,
)
from xtuner.v1.module import RouterResults
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import HiddenStates, MoEActFnConfig, MoEDecoderLayer
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.ops import get_apply_rotary_emb
from xtuner.v1.utils import get_logger


logger = get_logger()


class Step3p5RotaryEmbedding(nn.Module):
    """Per-profile rotary embedding for Step3.5.

    Step3.5 uses a different RoPE configuration for ``full_attention`` and ``sliding_attention``
    layers (different ``rope_theta``, ``partial_rotary_factor`` and — only on full-attention layers
    — llama3 frequency smoothing). The model-level rotary in :class:`MoE` cannot express this, so each
    Step3.5 decoder layer owns its own instance and recomputes ``(cos, sin)`` from
    ``seq_ctx.position_ids``. Inverse frequencies are computed faithfully against HuggingFace
    (``ROPE_INIT_FUNCTIONS`` for llama3 is replicated inline so the result is independent of the
    installed transformers version).

    Args:
        head_dim (int): Per-head dimension.
        rope_theta (float): RoPE base wavelength.
        partial_rotary_factor (float): Fraction of ``head_dim`` that is rotated.
        max_position_embeddings (int): Maximum sequence length (kept for parity with HF caching).
        llama3_cfg (dict | None): llama3 smoothing parameters (``factor``, ``low_freq_factor``,
            ``high_freq_factor``, ``original_max_position_embeddings``) or ``None`` for default RoPE.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        partial_rotary_factor: float,
        max_position_embeddings: int,
        llama3_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        # Stored as attributes (positional-or-keyword signature) so the test scaffolding's
        # `materialize_submodule` can re-instantiate and recompute `inv_freq` after `to_empty`.
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings
        self.llama3_cfg = llama3_cfg
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        if llama3_cfg is not None:
            inv_freq = self._apply_llama3_smoothing(inv_freq, **llama3_cfg)
        self.attention_scaling = 1.0
        self.max_seq_len_cached = max_position_embeddings
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float().to(x.device)).transpose(
                1, 2
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def _apply_llama3_smoothing(
        inv_freq: torch.Tensor,
        *,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
    ) -> torch.Tensor:
        # Mirrors transformers `_compute_llama3_parameters` smoothing applied on top of the base inv_freq.
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama


def _set_partial_rotary(self_attn, rotary_emb: Step3p5RotaryEmbedding) -> None:
    # The per-layer RoPE profile (full=partial 0.5, sliding=1.0) is owned by the Step3.5 decoder
    # layer, so the matching partial-rotary apply is set directly on the attention here instead of
    # threading a rope config through `build`. This keeps the per-layer RoPE fully contained and
    # avoids the deprecated `RopeScalingConfig` (whose only consumer in `MultiHeadAttention` is this
    # `apply_rotary_emb` selection). Only `MultiHeadAttention` has `apply_rotary_emb`.
    if hasattr(self_attn, "apply_rotary_emb"):
        self_attn.apply_rotary_emb = get_apply_rotary_emb(
            None, enable_partial_rotary=rotary_emb.partial_rotary_factor != 1.0
        )


class Step3p5DenseDecoderLayer(DenseDecoderLayer):
    """Dense decoder layer that recomputes RoPE per layer (Step3.5
    first_k_dense layers)."""

    def __init__(self, *, rotary_emb: Step3p5RotaryEmbedding, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rotary_emb = rotary_emb
        _set_partial_rotary(self.self_attn, rotary_emb)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        # Ignore the model-level position embeddings: this layer's RoPE profile differs per layer type.
        position_embeddings = self.rotary_emb(hidden_states, seq_ctx.position_ids)
        return super().forward(hidden_states, position_embeddings, seq_ctx)


class Step3p5MoEDecoderLayer(MoEDecoderLayer):
    """MoE decoder layer that recomputes RoPE per layer and clamps the shared
    expert SwiGLU."""

    def __init__(
        self,
        *,
        rotary_emb: Step3p5RotaryEmbedding,
        shared_swiglu_limit: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rotary_emb = rotary_emb
        _set_partial_rotary(self.self_attn, rotary_emb)
        if shared_swiglu_limit is not None:
            assert self.shared_experts is not None
            self.shared_experts.swiglu_limit = shared_swiglu_limit

    def forward(
        self,
        *hidden_states: torch.Tensor,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[HiddenStates, RouterResults] | tuple[torch.Tensor, ...]:
        if len(hidden_states) == 1:
            assert isinstance(seq_ctx, SequenceContext)
            position_embeddings = self.rotary_emb(hidden_states[0], seq_ctx.position_ids)
        else:
            assert isinstance(seq_ctx, list)
            position_embeddings = [self.rotary_emb(h, sc.position_ids) for h, sc in zip(hidden_states, seq_ctx)]
        return super().forward(*hidden_states, seq_ctx=seq_ctx, position_embeddings=position_embeddings)


class Step3p5MoE(MoE):
    def build_rotary_embedding(self, config: "Step3p5MoEConfig"):  # type: ignore[override]
        # Model-level rotary is unused (each decoder layer owns its own), but `MoE.forward` still
        # computes it; return the full-attention profile so the call is valid. CPU init keeps it
        # correct even when the model is built on the meta device (see BaseModel.build_rotary_embedding).
        with torch.device("cpu"):
            return config.build_layer_rotary("full_attention")

    def build_layers(self, config: "Step3p5MoEConfig") -> nn.ModuleDict:  # type: ignore[override]
        from xtuner.v1.model.utils.misc import module_dict_repr

        layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layers_type[layer_idx]
            attention_config = config.attention if layer_type == "full_attention" else config.sliding_attention
            # CPU init so the rotary buffers are real even under a meta-device model build. The
            # per-layer partial-rotary apply is set from this rotary inside the decoder layer (so no
            # rope config is threaded through `build` — see `_set_partial_rotary`).
            with torch.device("cpu"):
                rotary_emb = config.build_layer_rotary(layer_type)

            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = Step3p5DenseDecoderLayer(
                    rotary_emb=rotary_emb,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    attention_config=attention_config,
                    layer_type=layer_type,
                    rope_scaling_cfg=None,
                    generate_config=config.generate_config,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                )
            else:
                layers[str(layer_idx)] = Step3p5MoEDecoderLayer(
                    rotary_emb=rotary_emb,
                    shared_swiglu_limit=config.shared_swiglu_limits.get(layer_idx),
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    moe_intermediate_size=config.moe_intermediate_size,
                    mlp_bias=config.mlp_bias,
                    gate_bias=config.gate_bias,
                    moe_bias=config.moe_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    num_experts_per_tok=config.num_experts_per_tok,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    with_shared_expert_gate=config.with_shared_expert_gate,
                    hidden_factor=config.hidden_factor,
                    layer_type=layer_type,
                    attention_config=attention_config,
                    rope_scaling_cfg=None,
                    generate_config=config.generate_config,
                    router_config=config.router,
                    router_compute_dtype=config.router_compute_dtype,
                    moe_act_fn_cfg=config.layer_moe_act_fn_cfg(layer_idx),
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    dispatcher=config.dispatcher,
                    ep_mesh=self.ep_mesh,
                )
                if config.freeze_routers:
                    layers[str(layer_idx)].gate.requires_grad_(False)
                    layers[str(layer_idx)].gate.eval()

        layers.__class__.__repr__ = module_dict_repr  # type: ignore[method-assign]
        return layers

    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if key.startswith("norm."):
            return ["model." + key]
        if key == "lm_head.weight":
            return ["lm_head.weight"]

        # Routed experts: the fused XTuner grouped-linear maps to per-expert HF tensors (Qwen3-MoE
        # style). The checkpoint stores experts *split* per expert under `moe.experts.{i}.*`; the
        # interleaved key order `[gate_0, up_0, gate_1, up_1, ...]` matches XTuner's expert-major
        # fused layout, so the default `safetensors_to_params` (concat along dim 0) and the default
        # save split both shard correctly across FSDP/EP without any per-model override.
        n = self.config.n_routed_experts
        if "experts.fused_w1w3.weight" in key:
            out: list[str] = []
            for i in range(n):
                out.append(key.replace("experts.fused_w1w3.weight", f"moe.experts.{i}.gate_proj.weight"))
                out.append(key.replace("experts.fused_w1w3.weight", f"moe.experts.{i}.up_proj.weight"))
            return out
        if "experts.fused_w2.weight" in key:
            return [key.replace("experts.fused_w2.weight", f"moe.experts.{i}.down_proj.weight") for i in range(n)]

        # Router linear + per-expert bias.
        if "gate.router.e_score_correction_bias" in key:
            return [key.replace("gate.router.e_score_correction_bias", "moe.router_bias")]
        if re.search(r"layers\.\d+\.gate\.weight$", key):
            return [key.replace("gate.weight", "moe.gate.weight")]

        # Shared expert: XTuner `shared_experts.*` -> HF `share_expert.*`.
        if "shared_experts." in key:
            return [key.replace("shared_experts.", "share_expert.")]

        return [key]


class Step3p5MoEConfig(MoEConfig):
    model_type: str | None = "step3p5"
    bos_token_id: int | None = None
    rms_norm_type: Literal["default", "zero_centered"] = "zero_centered"

    # --- Step3.5-specific extra fields ---
    # Second attention profile: sliding-window layers use a different head count than full layers.
    sliding_attention: MHAConfig = Field(
        default_factory=lambda: MHAConfig(
            num_attention_heads=96,
            num_key_value_heads=8,
            head_dim=128,
            qk_norm=True,
            head_gate=True,
            rms_norm_type="zero_centered",
            rms_norm_eps=1e-5,
            sliding_window=512,
        )
    )
    # Per-profile RoPE.
    full_rope_theta: float = 5_000_000.0
    sliding_rope_theta: float = 10_000.0
    full_partial_rotary_factor: float = 0.5
    sliding_partial_rotary_factor: float = 1.0
    rope_factor: float = 2.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 32.0
    rope_original_max_position_embeddings: int = 131072
    # Sparse per-layer SwiGLU clip limits: {layer_idx: limit}. `routed` clamps the routed experts'
    # activation; `shared` clamps the shared expert. Layers absent from the dict are unclamped.
    routed_swiglu_limits: dict[int, float] = {}
    shared_swiglu_limits: dict[int, float] = {}

    def build(self) -> Step3p5MoE:
        return Step3p5MoE(self)

    # `layers_type` follows Step3.5's fixed pattern: every 4th layer is full attention, the rest sliding.
    @property
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention", "linear_attention"]]:  # type: ignore[override]
        return ["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)]

    def build_layer_rotary(self, layer_type: str) -> Step3p5RotaryEmbedding:
        if layer_type == "full_attention":
            return Step3p5RotaryEmbedding(
                head_dim=self.attention.head_dim,
                rope_theta=self.full_rope_theta,
                partial_rotary_factor=self.full_partial_rotary_factor,
                max_position_embeddings=self.max_position_embeddings,
                llama3_cfg={
                    "factor": self.rope_factor,
                    "low_freq_factor": self.rope_low_freq_factor,
                    "high_freq_factor": self.rope_high_freq_factor,
                    "original_max_position_embeddings": self.rope_original_max_position_embeddings,
                },
            )
        return Step3p5RotaryEmbedding(
            head_dim=self.sliding_attention.head_dim,
            rope_theta=self.sliding_rope_theta,
            partial_rotary_factor=self.sliding_partial_rotary_factor,
            max_position_embeddings=self.max_position_embeddings,
            llama3_cfg=None,
        )

    def layer_moe_act_fn_cfg(self, layer_idx: int) -> MoEActFnConfig:
        limit = self.routed_swiglu_limits.get(layer_idx)
        if limit is not None:
            return MoEActFnConfig(act_type="swiglu_clip", clip_limit=limit)
        return self.moe_act_fn_cfg

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        assert hf_config.model_type == "step3p5", f"Expected model_type 'step3p5', got '{hf_config.model_type}'"

        head_dim = hf_config.head_dim
        full_heads = hf_config.num_attention_heads
        kv_heads = hf_config.num_attention_groups
        sliding_setting = hf_config.attention_other_setting
        rope_scaling = hf_config.rope_scaling

        num_layers = hf_config.num_hidden_layers
        swiglu_limits = list(getattr(hf_config, "swiglu_limits", []))[:num_layers]
        swiglu_limits_shared = list(getattr(hf_config, "swiglu_limits_shared", []))[:num_layers]
        routed_swiglu_limits = {i: float(v) for i, v in enumerate(swiglu_limits) if v}
        shared_swiglu_limits = {i: float(v) for i, v in enumerate(swiglu_limits_shared) if v}

        attention = MHAConfig(
            num_attention_heads=full_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            qk_norm=hf_config.use_qk_norm,
            head_gate=hf_config.use_head_wise_attn_gate,
            rms_norm_type="zero_centered",
            rms_norm_eps=hf_config.rms_norm_eps,
            sliding_window=-1,
        )
        sliding_attention = MHAConfig(
            num_attention_heads=sliding_setting["num_attention_heads"],
            num_key_value_heads=sliding_setting["num_attention_groups"],
            head_dim=sliding_setting["head_dim"],
            qk_norm=hf_config.use_qk_norm,
            head_gate=hf_config.use_head_wise_attn_gate,
            rms_norm_type="zero_centered",
            rms_norm_eps=hf_config.rms_norm_eps,
            sliding_window=hf_config.sliding_window,
        )

        # First MoE layer index -> number of leading dense layers.
        moe_layers = [int(i) for i in str(hf_config.moe_layers_enum).split(",")]
        first_k_dense_replace = min(moe_layers)

        # HF stores eos as a list of acceptable ids; TransformerConfig wants a single int.
        eos = hf_config.eos_token_id
        eos_token_id = int(eos[0]) if isinstance(eos, (list, tuple)) else int(eos)

        return cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            pad_token_id=getattr(hf_config, "pad_token_id", None),
            bos_token_id=getattr(hf_config, "bos_token_id", None),
            eos_token_id=eos_token_id,
            num_hidden_layers=num_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            hidden_act="silu",
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            attention=attention,
            sliding_attention=sliding_attention,
            n_routed_experts=hf_config.moe_num_experts,
            n_shared_experts=1,
            num_experts_per_tok=hf_config.moe_top_k,
            first_k_dense_replace=first_k_dense_replace,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            with_shared_expert_gate=False,
            router=NoAuxRouterConfig(
                scoring_func="sigmoid",
                n_group=1,
                topk_group=1,
                norm_topk_prob=hf_config.norm_expert_weight,
                router_scaling_factor=hf_config.moe_router_scaling_factor,
            ),
            router_compute_dtype="float32",
            full_rope_theta=hf_config.rope_theta[0],
            sliding_rope_theta=hf_config.rope_theta[1],
            full_partial_rotary_factor=hf_config.partial_rotary_factors[0],
            sliding_partial_rotary_factor=hf_config.partial_rotary_factors[1],
            rope_factor=rope_scaling["factor"],
            rope_low_freq_factor=rope_scaling["low_freq_factor"],
            rope_high_freq_factor=rope_scaling["high_freq_factor"],
            rope_original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            routed_swiglu_limits=routed_swiglu_limits,
            shared_swiglu_limits=shared_swiglu_limits,
        )

    @property
    def hf_config(self):
        # Step3.5 ships as a `trust_remote_code` model with no built-in transformers config class, so
        # `save_hf` falls back to copying config.json / tokenizer / *.py from the source dir.
        return None


class Step3p5FlashConfig(Step3p5MoEConfig):
    vocab_size: int = 128896
    max_position_embeddings: int = 262144
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int = 128007
    num_hidden_layers: int = 45
    hidden_size: int = 4096
    intermediate_size: int = 11264
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention: MHAConfig = Field(
        default_factory=lambda: MHAConfig(
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            qk_norm=True,
            head_gate=True,
            rms_norm_type="zero_centered",
            rms_norm_eps=1e-5,
            sliding_window=-1,
        )
    )
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    moe_intermediate_size: int = 1280
    with_shared_expert_gate: bool = False
    router_compute_dtype: Literal["float32", "native"] = "float32"
    router: NoAuxRouterConfig = Field(
        default_factory=lambda: NoAuxRouterConfig(
            scoring_func="sigmoid",
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            router_scaling_factor=3.0,
        )
    )
    routed_swiglu_limits: dict[int, float] = Field(default_factory=lambda: {43: 7.0, 44: 7.0})
    shared_swiglu_limits: dict[int, float] = Field(default_factory=lambda: {44: 16.0})
