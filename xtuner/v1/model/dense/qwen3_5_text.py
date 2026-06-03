import re
from typing import Literal

from pydantic import Field, computed_field

from xtuner.v1.model.base import HFSaveCfg, TransformerConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.rope import RopeParametersConfig

from .qwen3vl_text import Qwen3VLTextDense


class Qwen3_5_VLTextDense(Qwen3VLTextDense):
    def to_hf_key_list(self, key: str) -> list[str]:
        # Emit the standalone language-model layout (``model.<...>``). The VLM nesting
        # under ``model.language_model.`` is applied by the config's ``hf_key_mapping``
        # so this tower stays unaware of how it is composed into the VLM.
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if "layers" in key:
            # HF stores the GatedDeltaNet under ``linear_attn`` while XTuner keeps the
            # generic ``self_attn`` attribute name regardless of attention type, so the
            # rename is driven by the per-layer ``layers_type``.
            layer_idx = int(re.findall(r"layers\.(\d+)\.", key)[0])
            if self.config.layers_type[layer_idx] == "linear_attention":
                key = key.replace("self_attn", "linear_attn")

        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        else:
            return [key]


class Qwen3_5_VLTextDenseConfig(TransformerConfig):
    rms_norm_type: Literal["default", "zero_centered"] = "zero_centered"
    # The dense text tower emits the standalone ``model.<...>`` layout; this remaps it
    # to the VLM's ``model.language_model.<...>`` namespace on both load and save.
    hf_key_mapping: dict[str, str] | None = {r"^model\.": "model.language_model."}
    # Qwen3.5 keeps the GatedDeltaNet gated-RMSNorm weight and the per-head decay
    # parameter ``A_log`` in fp32; the rest of the model runs in bf16.
    hf_save_cfg: HFSaveCfg = HFSaveCfg(
        fp32_keys_pattern=[
            r"model\.language_model\.layers\.\d+\.linear_attn\.norm\.weight",
            r"model\.language_model\.layers\.\d+\.linear_attn\.A_log",
        ],
    )

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "linear_attention"]]:
        # ``full_attention_interval`` == 4: every 4th layer (idx 3, 7, ...) is full
        # attention, the rest are linear (GatedDeltaNet).
        return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(self.num_hidden_layers)]

    def build(self) -> Qwen3_5_VLTextDense:
        return Qwen3_5_VLTextDense(self)


class Qwen3_5_VLTextDense4BConfig(Qwen3_5_VLTextDenseConfig):
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    pad_token_id: int | None = None
    eos_token_id: int = 248044
    num_hidden_layers: int = 32
    hidden_size: int = 2560
    intermediate_size: int = 9216
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    attention: MHAConfig = MHAConfig(
        with_gate=True,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        qk_norm=True,
        rms_norm_eps=1e-6,
        rms_norm_type="zero_centered",
    )
    linear_attention: GatedDeltaNetConfig = GatedDeltaNetConfig(
        num_value_heads=32,
        num_key_heads=16,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(
            rope_theta=10000000.0,
            rope_type="qwen3_vl",
            mrope_section=[11, 11, 10],
            partial_rotary_factor=0.25,
        )
    )
