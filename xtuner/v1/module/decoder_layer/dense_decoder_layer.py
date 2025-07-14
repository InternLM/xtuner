import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from xtuner.v1.config.base_model import TransformerConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import (
    RMSNorm,
    build_attnention,
)
from xtuner.v1.utils import ForwardState

from ..linear.linear import _Linear


class DenseMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = _Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = _Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DenseDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = build_attnention(config, layer_idx)
        self.mlp = DenseMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            state=ForwardState.TRAINING,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn.prefilling(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            state=ForwardState.DECODING,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_kv_cache(
            max_batch_size=max_batch_size,
            max_length=max_length,
            block_size=block_size,
        )
