import json
import re
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from typing_extensions import Self

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module import AttnOutputs, GatedDeltaNetConfig, MHAConfig, MLAConfig, RMSNorm
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.mtp import MTPBlock
from xtuner.v1.module.mtp.config import MTPConfig

from .qwen2 import Qwen2Dense, Qwen2DenseConfig


class MiMoMTPDenseLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool,
        hidden_act: str,
        rms_norm_eps: float,
        rms_norm_type: Literal["default", "zero_centered"],
        attention_config: GatedDeltaNetConfig | MLAConfig | MHAConfig,
        rope_scaling_cfg=None,
        generate_config=None,
        float8_cfg=None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.token_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.hidden_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.input_proj = build_linear(hidden_size * 2, hidden_size, bias=False, float8_cfg=float8_cfg)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.final_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.self_attn = attention_config.build(
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            rope_scaling_cfg=rope_scaling_cfg,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )
        self.mlp = DenseMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
            hidden_act=hidden_act,
            float8_cfg=float8_cfg,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        future_embeddings = self.token_layernorm(future_embeddings)
        hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([hidden_states, future_embeddings], dim=-1))

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs: AttnOutputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
        hidden_states = residual + attn_outputs["projected_output"]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)

        empty = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states, empty, empty


class MiMoDense(Qwen2Dense):
    config: "MiMoDenseConfig"

    def to_hf_key_list(self, key: str) -> list[str]:
        if key.startswith("mtp_block.layers."):
            key = re.sub(r"^mtp_block\.layers\.(\d+)\.", r"model.mtp_layers.\1.", key)
            return [key]
        return super().to_hf_key_list(key)

    def build_mtp_block(self, config: TransformerConfig) -> MTPBlock:
        mtp_config = getattr(config, "mtp_config", None)
        assert mtp_config is not None, "mtp_config must be provided"

        mtp_layers = []
        last_layer_idx = config.num_hidden_layers - 1
        layers_type_list = config.layers_type
        attention_config: GatedDeltaNetConfig | MLAConfig | MHAConfig
        if layers_type_list[last_layer_idx] in ["full_attention", "sliding_attention"]:
            attention_config = config.attention
        elif layers_type_list[last_layer_idx] == "linear_attention":
            assert config.linear_attention is not None, (
                "linear_attention config must be provided for linear_attention layer"
            )
            attention_config = config.linear_attention
        else:
            raise ValueError(f"Unsupported layer type {layers_type_list[last_layer_idx]}")

        for i in range(mtp_config.num_layers):
            mtp_layers.append(
                MiMoMTPDenseLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    attention_config=attention_config,
                    rope_scaling_cfg=config.rope_scaling_cfg,
                    generate_config=config.generate_config,
                    float8_cfg=config.float8_cfg,
                    layer_type=layers_type_list[last_layer_idx],
                    layer_idx=config.num_hidden_layers + i,
                )
            )
        return MTPBlock(mtp_layers=mtp_layers)


class MiMoDenseConfig(Qwen2DenseConfig):
    mtp_config: MTPConfig | None = None
    model_type: str | None = "mimo"

    def build(self) -> MiMoDense:
        return MiMoDense(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        hf_path = Path(hf_path)
        hf_config = json.loads((hf_path / "config.json").read_text())
        num_mtp_layers = int(hf_config.get("num_nextn_predict_layers", 0))

        return cls(
            vocab_size=hf_config["vocab_size"],
            max_position_embeddings=hf_config["max_position_embeddings"],
            pad_token_id=hf_config.get("pad_token_id"),
            bos_token_id=hf_config["bos_token_id"],
            eos_token_id=hf_config["eos_token_id"],
            num_hidden_layers=hf_config["num_hidden_layers"],
            max_window_layers=hf_config.get("max_window_layers"),
            hidden_size=hf_config["hidden_size"],
            intermediate_size=hf_config["intermediate_size"],
            rms_norm_eps=hf_config["rms_norm_eps"],
            rope_theta=hf_config["rope_theta"],
            hidden_act=hf_config["hidden_act"],
            attention=MHAConfig(
                num_attention_heads=hf_config["num_attention_heads"],
                num_key_value_heads=hf_config["num_key_value_heads"],
                head_dim=hf_config.get("head_dim", 128),
                sliding_window=hf_config.get("sliding_window"),
                qk_norm=False,
                qkv_bias=hf_config.get("attention_bias", True),
                o_bias=False,
            ),
            use_sliding_window=hf_config.get("use_sliding_window", False),
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            mtp_config=MTPConfig(num_layers=num_mtp_layers) if num_mtp_layers > 0 else None,
        )

    @property
    def hf_config(self):
        # Keep save_hf() on the "copy original HF files" path so remote-code files such
        # as configuration_mimo.py / modeling_mimo.py are preserved in exported checkpoints.
        return None


class MiMoDense7BConfig(MiMoDenseConfig):
    vocab_size: int = 151680
    max_position_embeddings: int = 32768
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None
    num_hidden_layers: int = 36
    max_window_layers: int = 36
    hidden_size: int = 4096
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-5
    rope_theta: float = 640000.0
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        qk_norm=False,
        qkv_bias=True,
        sliding_window=32768,
    )
    tie_word_embeddings: bool = False
    mtp_config: MTPConfig | None = MTPConfig(num_layers=1)
