import re

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

    def build(self) -> Qwen2Dense:
        return Qwen2Dense(self)


# TODO: Unify the config name style
class Qwen2Dense7BConfig(Qwen2DenseConfig):
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    pad_token_id: int = 151645  # eos_id
    eos_token_id: int = 151645
    bos_token_id: int = 151643
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
