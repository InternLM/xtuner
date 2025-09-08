from typing import Literal, cast

import torch
import torch.nn as nn
from pydantic import BaseModel

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class RopeScalingConfig(BaseModel):
    type: Literal["default", "linear", "dynamic", "yarn", "longrope", "llama3"] = "default"
    max_position_embeddings: int | None = None
    original_max_position_embeddings: int | None = None

    # For inference
    factor: float = 1.0
    beta_fast: float | None = None
    beta_slow: float | None = None
    short_factor: list[float] | None = None
    long_factor: list[float] | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        from xtuner.v1.model.base import TransformerConfig

        config = cast(TransformerConfig, config)
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling_cfg", None)
        if rope_scaling is None:
            self.rope_type = "default"
        else:
            self.rope_type = rope_scaling["type"]

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq: torch.Tensor
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids: torch.Tensor, device: torch.device):
        """Dynamic RoPE layers should recompute `inv_freq` in the following
        situations:

        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = cast(int, seq_len.item())

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
