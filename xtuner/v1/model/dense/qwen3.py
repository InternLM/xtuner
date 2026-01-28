import re
from pathlib import Path

import torch
from typing_extensions import Self

from transformers import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config as HFQwen3DenseConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.rope.rope import RopeScalingConfig

from .dense import Dense


class Qwen3Dense(Dense):
    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        else:
            return [key]


class Qwen3DenseConfig(TransformerConfig):
    use_sliding_window: bool = False
    pad_token_id: int | None = None
    bos_token_id: int

    def build(self) -> Qwen3Dense:
        return Qwen3Dense(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path | None = None, hf_config: PretrainedConfig | None = None) -> Self:
        if hf_path is not None:
            from transformers import AutoConfig
            from transformers.models.qwen3 import Qwen3Config as HFConfig
            hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
            assert isinstance(hf_config, HFConfig)
        assert hf_config is not None and isinstance(hf_config, PretrainedConfig)

        config = cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            pad_token_id=getattr(hf_config, "pad_token_id"),
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            num_hidden_layers=hf_config.num_hidden_layers,
            max_window_layers=hf_config.max_window_layers if hasattr(hf_config, "max_window_layers") else hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            hidden_act=hf_config.hidden_act,
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=hf_config.head_dim,
                sliding_window=hf_config.sliding_window if hasattr(hf_config, "sliding_window") else 1024,
                qk_norm=True,
            ),
            model_type=hf_config.model_type,
            use_sliding_window=hf_config.use_sliding_window if hasattr(hf_config, "use_sliding_window") else False,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope_scaling_cfg = RopeScalingConfig(type="qwen3_vl", 
                                                mrope_section=hf_config.rope_scaling['mrope_section']) 
                                                if hasattr(hf_config, "rope_scaling") else None,
        )

        return config

    @property
    def hf_config(self) -> HFQwen3DenseConfig:
        """Check if the configuration can be saved in HuggingFace format."""
        return HFQwen3DenseConfig(
            architectures=["Qwen3ForCausalLM"],
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            max_window_layers=self.max_window_layers,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_key_value_heads,
            head_dim=self.attention.head_dim,
            sliding_window=self.attention.sliding_window,
            use_sliding_window=self.use_sliding_window,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_scaling={
                'rope_type': 'default',
                'mrope_section': self.rope_scaling_cfg.mrope_section,
                "mrope_interleaved": True
            } if self.rope_scaling_cfg is not None else None,
            dtype=torch.bfloat16,
        )


# TODO: Unify the config name style
class Qwen3Dense8BConfig(Qwen3DenseConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    num_hidden_layers: int = 36
    max_window_layers: int = 36
    hidden_size: int = 4096
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"

    attention: MHAConfig = MHAConfig(
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, qk_norm=True, sliding_window=1024
    )
    tie_word_embeddings: bool = False


class Qwen3Dense4BConfig(Qwen3DenseConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 262144
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    num_hidden_layers: int = 36
    max_window_layers: int = 36
    hidden_size: int = 2560
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    hidden_act: str = "silu"

    attention: MHAConfig = MHAConfig(
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, qk_norm=True, sliding_window=1024
    )
    tie_word_embeddings: bool = True


class Qwen3Dense0P6BConfig(Qwen3DenseConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    num_hidden_layers: int = 28
    max_window_layers: int = 28
    hidden_size: int = 1024
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"

    attention: MHAConfig = MHAConfig(
        num_attention_heads=16, num_key_value_heads=8, head_dim=128, qk_norm=True, sliding_window=None
    )
    tie_word_embeddings: bool = False  # 官方的是 False
