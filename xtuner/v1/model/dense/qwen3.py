import re

from xtuner.v1.config import TransformerConfig
from xtuner.v1.module.attention import MHAConfig

from .dense import Dense


class Qwen3Dense(Dense):
    def to_hf_key_list(self, key: str) -> list[str]:
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

    def build(self) -> Qwen3Dense:
        return Qwen3Dense(self)


# TODO: Unify the config name style
class Qwen3Dense8BConfig(Qwen3DenseConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    pad_token_id: int = 151645  # eos_id
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
