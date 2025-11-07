from typing import Literal, Protocol, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from typing_extensions import overload

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class RopeScalingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["default", "linear", "dynamic", "yarn", "longrope", "llama3", "qwen3_vl"] = "default"

    max_position_embeddings: int | None = None  # TODO: 无用参数考虑删除
    original_max_position_embeddings: int | None = None  # TODO: 无用参数考虑删除

    # For Qwen3VL
    mrope_section: list[int] | None = None  # e.g. [24, 20, 20]

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


class RotaryEmbeddingProtocol(Protocol):
    """Protocol for attention modules."""

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the rope module."""
        ...

    def __call__(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        from xtuner.v1.model.base import TransformerConfig

        config = cast(TransformerConfig, config)
        super().__init__()

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq: torch.Tensor
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids: torch.LongTensor, device: torch.device):
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
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # [B, H/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, S]
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(
                1, 2
            )  # [B, S, H/2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, H]
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # [B, S, H]

    @overload  # type: ignore
    def __call__(  # type: ignore
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    __call__ = nn.Module.__call__


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        from xtuner.v1.model.base import TransformerConfig

        config = cast(TransformerConfig, config)
        super().__init__()

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq: torch.Tensor
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling_cfg.mrope_section
        assert self.mrope_section is not None

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # type: ignore
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @overload  # type: ignore
    def __call__(  # type: ignore
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    __call__ = nn.Module.__call__


def get_rope_embedding(config, device=None) -> RotaryEmbeddingProtocol:
    from xtuner.v1.model import TransformerConfig

    config = cast(TransformerConfig, config)
    rope_scaling_cfg = config.rope_scaling_cfg
    if rope_scaling_cfg is not None and rope_scaling_cfg.type == "qwen3_vl":
        return Qwen3VLTextRotaryEmbedding(config, device=device)
    else:
        return RotaryEmbedding(config, device=device)
