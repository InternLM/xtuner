import re
from pathlib import Path

import torch
from typing_extensions import Self

from transformers.models.qwen2 import Qwen2Config as HFQwen2DenseConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module.attention import MHAConfig

from .dense import Dense


class Qwen2Dense(Dense):
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


class Qwen2DenseConfig(TransformerConfig):
    use_sliding_window: bool = False
    bos_token_id: int

    def build(self) -> Qwen2Dense:
        return Qwen2Dense(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        from transformers import AutoConfig
        from transformers.models.qwen2 import Qwen2Config as HFConfig

        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)

        assert isinstance(hf_config, HFConfig)

        config = cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            pad_token_id=getattr(hf_config, "pad_token_id"),
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            num_hidden_layers=hf_config.num_hidden_layers,
            max_window_layers=hf_config.max_window_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            hidden_act=hf_config.hidden_act,
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
                sliding_window=hf_config.sliding_window,
                qk_norm=False,
                qkv_bias=True,
            ),
            use_sliding_window=hf_config.use_sliding_window,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )
        return config

    @property
    def hf_config(self) -> HFQwen2DenseConfig:
        """Check if the configuration can be saved in HuggingFace format."""
        return HFQwen2DenseConfig(
            architectures=["Qwen2ForCausalLM"],
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
            dtype=torch.bfloat16,
        )


# TODO: Unify the config name style
class Qwen2Dense7BConfig(Qwen2DenseConfig):
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    bos_token_id: int = 151643
    pad_token_id: int | None = None
    eos_token_id: int = 151643  # eos_id
    num_hidden_layers: int = 28
    hidden_size: int = 3584
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-06
    rope_theta: float = 10000
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        qk_norm=False,
        qkv_bias=True,
    )
    # sliding_window= 4096
    tie_word_embeddings: bool = False
