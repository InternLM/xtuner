from functools import lru_cache
from typing import Callable, Literal, Optional, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self, deprecated, overload

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.device import get_device


DEVICE = get_device()

logger = get_logger()


@deprecated(
    "RopeScalingConfig is deprecated and will be removed in a future version. "
    "Use RopeParametersConfig from xtuner.v1.model.base instead.",
    category=FutureWarning,
)
class RopeScalingConfig(BaseModel):
    """Deprecated: Use RopeParametersConfig from xtuner.v1.model.base instead.

    This class is kept for backward compatibility. New code should use RopeParametersConfig
    which provides a unified interface for all RoPE configurations.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["default", "linear", "dynamic", "yarn", "longrope", "llama3", "qwen3_vl"] = "default"

    max_position_embeddings: int | None = None
    original_max_position_embeddings: int | None = None

    # For Qwen3VL
    mrope_section: list[int] | None = None  # e.g. [24, 20, 20]
    partial_rotary_factor: float = 1.0

    factor: float | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    short_factor: list[float] | None = None
    long_factor: list[float] | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = False

    # For FoPE
    fope_init_factor: float | None = None
    fope_sep_head: bool | None = None
    num_inv_freq: int | None = None

    @property
    def use_fope(self) -> bool:
        return self.fope_init_factor is not None or self.fope_sep_head is not None or self.num_inv_freq is not None


class RopeParametersConfig(BaseModel):
    """Unified RoPE (Rotary Position Embedding) parameters configuration.

    This class consolidates all rope-related parameters and serves as the primary
    configuration for RoPE. It is compatible with both transformers 4.57.0 and 5.2.0+.

    For backward compatibility:
    - Old configs with rope_theta + rope_scaling_cfg will be automatically converted
    - The computed fields rope_theta and rope_scaling provide backward compatible access
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Core rope parameters (transformers 5.2.0+ style)
    rope_theta: float = 10000.0
    rope_type: Literal["default", "linear", "dynamic", "yarn", "longrope", "llama3", "qwen3_vl"] = "default"

    # Position embeddings
    # chenchiyu: remove max_position_embeddings since no one use it, no value fill into it,
    # And hf origin rope_scaling doesn't include it either.
    original_max_position_embeddings: int | None = None

    # Scaling parameters for yarn/llama3/longrope
    factor: float | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    short_factor: list[float] | None = None
    long_factor: list[float] | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = False

    # For Qwen3VL
    mrope_section: list[int] | None = None  # e.g. [24, 20, 20]
    partial_rotary_factor: float = 1.0

    # For FoPE
    fope_init_factor: float | None = None
    fope_sep_head: bool | None = None
    num_inv_freq: int | None = None

    @property
    def use_fope(self) -> bool:
        """Check if FoPE is enabled."""
        return self.fope_init_factor is not None or self.fope_sep_head is not None or self.num_inv_freq is not None

    @classmethod
    @lru_cache(maxsize=None)
    def _get_rope_scaling_to_parameters_mapping(cls) -> dict[str, str]:
        """Dynamically build mapping from RopeScalingConfig field names to
        RopeParametersConfig field names.

        Based on RopeParametersConfig.model_fields, excluding rope_theta. rope_type maps to 'type' for backward
        compatibility with RopeScalingConfig.

        Result is cached since model_fields is static after class definition.
        """
        mapping: dict[str, str] = {}
        for field_name in cls.model_fields.keys():
            if field_name == "rope_theta":
                continue
            elif field_name == "rope_type":
                mapping["type"] = "rope_type"
            else:
                mapping[field_name] = field_name
        return mapping

    @classmethod
    @lru_cache(maxsize=None)
    def get_rope_scaling_field_names(cls) -> tuple[str, ...]:
        """Return field names in RopeParametersConfig that correspond to
        rope_scaling fields.

        Excludes rope_theta and rope_type themselves.
        """
        return tuple(v for v in cls._get_rope_scaling_to_parameters_mapping().values() if v != "rope_type")

    @classmethod
    def from_legacy_cfg(
        cls,
        rope_theta: float | None = None,
        rope_scaling_cfg=None,  # type: ignore  # RopeScalingConfig type
    ) -> "RopeParametersConfig":
        """Create RopeParametersConfig from legacy rope_theta and
        rope_scaling_cfg.

        This method provides backward compatibility for old configs that use separate rope_theta and rope_scaling_cfg
        fields.
        """
        kwargs: dict = {}

        # Set rope_theta only if explicitly provided (otherwise use RopeParametersConfig default)
        if rope_theta is not None:
            kwargs["rope_theta"] = rope_theta

        # Copy fields from rope_scaling_cfg if provided
        if rope_scaling_cfg is not None:
            for src_field, dst_field in cls._get_rope_scaling_to_parameters_mapping().items():
                if hasattr(rope_scaling_cfg, src_field):
                    value = getattr(rope_scaling_cfg, src_field)
                    if value is not None:
                        kwargs[dst_field] = value

        return cls(**kwargs)

    def to_rope_scaling_dict(self) -> dict | None:
        """Convert to rope_scaling dict format for HF compatibility."""
        if self.rope_type == "default" and not self.use_fope:
            return None

        result: dict = {"type": self.rope_type}

        # Add all non-None scaling parameters dynamically
        for field_name in self.get_rope_scaling_field_names():
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        return result

    def to_rope_parameters_dict(self) -> dict | None:
        """Convert to rope_parameters dict format for HF compatibility."""
        if not self.use_fope:
            return self.model_dump(exclude={"fope_init_factor", "fope_sep_head", "num_inv_freq"})
        else:
            return self.model_dump()

    @classmethod
    def from_hf_config(cls, hf_config, default_value_dict=None) -> Optional["RopeParametersConfig"]:
        """Create RopeParametersConfig from HF config with version
        compatibility."""
        # TODO: remove default_value_dict, it's used for DeepseekV3Config in some cases for now.
        kwargs: dict = default_value_dict or {}
        default_rope_theta = kwargs["rope_theta"] if "rope_theta" in kwargs else 10000.0

        hf_rope_parameters = getattr(hf_config, "rope_parameters", None)
        hf_rope_scaling = getattr(hf_config, "rope_scaling", None)
        hf_rope_theta = getattr(hf_config, "rope_theta", None)
        if isinstance(hf_rope_parameters, dict):
            # Try rope_parameters dict (transformers 5.2.0)
            # In 5.2.0, all rope params are consolidated into rope_parameters
            rope_theta = hf_rope_parameters.get("rope_theta", default_rope_theta)
            if "rope_type" in hf_rope_parameters:
                kwargs["rope_type"] = hf_rope_parameters["rope_type"]

            # Copy other parameters dynamically from scaling field names
            for field_name in cls.get_rope_scaling_field_names():
                if field_name in hf_rope_parameters:
                    kwargs[field_name] = hf_rope_parameters[field_name]

            kwargs["rope_theta"] = rope_theta

            return cls(**kwargs)

        elif hf_rope_theta is not None or hf_rope_scaling is not None:
            # Try rope_scaling dict (transformers 4.57.0)
            # In 4.57.0, scaling params are in rope_scaling dict and rope_theta is a separate attribute of hf_config

            if isinstance(hf_rope_scaling, dict):
                # Note: rope_theta should NOT be in rope_scaling dict in standard transformers format.
                # It is obtained from hf_config.rope_theta (4.57.0) or rope_parameters (5.2.0) above.
                # Copy scaling parameters from rope_scaling
                for key in list(cls._get_rope_scaling_to_parameters_mapping().keys()) + ["max_position_embeddings"]:
                    if key in hf_rope_scaling:
                        assert key != "max_position_embeddings", (
                            "hf_config.rope_scaling should not include max_position_embeddings. "
                            "This value should be obtained from hf_config.max_position_embeddings."
                        )
                        kwargs[cls._get_rope_scaling_to_parameters_mapping()[key]] = hf_rope_scaling[key]

            kwargs["rope_theta"] = hf_rope_theta or default_rope_theta

            return cls(**kwargs)

        else:
            # no rope related parameters found, return None
            return None


class RotaryEmbeddingProtocol(Protocol):
    """Protocol for attention modules."""

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the rope module."""
        ...

    def __call__(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def to(self, device: torch.device) -> Self: ...


def compute_default_rope_parameters(
    config,
    device: Optional["torch.device"] = None,
) -> tuple["torch.Tensor", float]:
    """Compute default RoPE parameters with compatibility for both old and new
    config formats.

    Supports:
    - New format: config.rope_parameters_cfg (RopeParametersConfig from base.py)
    - Old format: config.rope_theta + config.rope_scaling_cfg
    """
    from xtuner.v1.model.base import TransformerConfig

    config = cast(TransformerConfig, config)

    # Try new format first: rope_parameters_cfg
    if config.rope_parameters_cfg is not None:
        rope_parameters_cfg = config.rope_parameters_cfg
        base = getattr(rope_parameters_cfg, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(rope_parameters_cfg, "partial_rotary_factor", 1.0)
    else:
        # Fall back to old format
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config.rope_scaling_cfg, "partial_rotary_factor", 1.0)

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    inv_freq = inv_freq.to(device=device)
    return inv_freq, attention_factor


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

        # Get rope_type from rope_parameters_cfg (new format) or rope_scaling_cfg (old format)
        if config.rope_parameters_cfg is not None:
            self.rope_type = config.rope_parameters_cfg.rope_type
        elif config.rope_scaling_cfg is not None:
            self.rope_type = config.rope_scaling_cfg.type

        assert self.rope_type in ["default", "linear", "yarn", "llama3"], (
            f"Unsupported rope_type: {self.rope_type}. Supported types are: 'default', 'linear', 'yarn', 'llama3'."
        )

        # The implementation of RoPE has been refactored in Transformers V5, and
        # the following approach is used for compatibility.
        self.rope_init_fn: Callable = compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq: torch.Tensor
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq.to(DEVICE), persistent=False)
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
            # TODO: remove to(x.device) because from_hf has already moved the rotary_emb module to the correct device
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


def _compute_fope_parameters(
    num_inv_freq: int | None, inv_freq: torch.Tensor, max_position_embeddings: int
) -> torch.Tensor:
    if inv_freq.device.type == "meta":
        return inv_freq

    logger.debug(f"At inv_freq.device.type: {inv_freq.device.type}, _compute_fope_parameters ")
    assert (inv_freq[:-1] > inv_freq[1:]).all(), "Expected inv_freq to be in decreasing order"

    inv_freq_idx_selected = torch.ones_like(inv_freq, dtype=torch.bool)
    if num_inv_freq is not None:
        num_inv_freq = num_inv_freq
        inv_freq_idx_selected[num_inv_freq:] = False
    else:
        inv_freq_idx_selected = inv_freq > (2.0 * torch.pi / max_position_embeddings)
        # num_inv_freq = inv_freq_idx_selected.sum().item()

    inv_freq = inv_freq[inv_freq_idx_selected]

    return inv_freq


class FourierEmbedding(RotaryEmbedding):
    def __init__(self, config, device=None):
        from xtuner.v1.model.base import TransformerConfig

        config = cast(TransformerConfig, config)
        super().__init__(config, device)

        rope_scaling_cfg = config.rope_scaling_cfg
        assert rope_scaling_cfg is not None
        self.num_inv_freq = rope_scaling_cfg.num_inv_freq
        self.fope_sep_head = rope_scaling_cfg.fope_sep_head
        self.fope_init_factor = rope_scaling_cfg.fope_init_factor

        # zero out under-trained frequencies
        inv_freq = _compute_fope_parameters(self.num_inv_freq, self.inv_freq, config.max_position_embeddings)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if self.num_inv_freq is not None:
            assert (self.inv_freq > (2.0 * torch.pi / config.max_position_embeddings)).all() or (
                self.inv_freq.shape[-1] == self.num_inv_freq
            ), "FoPE is wrongly initialized."

        self.head_dim = getattr(self.config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.input_dim = self.inv_freq.shape[-1]
        self.output_dim = self.inv_freq.shape[-1]

        if self.fope_sep_head:
            sin_coef = torch.randn(self.config.num_key_value_heads, self.input_dim, self.output_dim).to(
                self.inv_freq.device
            )
            cos_coef = torch.randn(self.config.num_key_value_heads, self.input_dim, self.output_dim).to(
                self.inv_freq.device
            )
        else:
            sin_coef = torch.randn(self.input_dim, self.output_dim).to(self.inv_freq.device)
            cos_coef = torch.randn(self.input_dim, self.output_dim).to(self.inv_freq.device)

        # use same generator to initialize sin_coef and cos_coef, so each rank will get the same sin_coef and cos_coef
        generator = torch.Generator(device=self.inv_freq.device)
        generator.manual_seed(123)
        torch.nn.init.xavier_normal_(sin_coef, gain=self.fope_init_factor, generator=generator)
        torch.nn.init.xavier_normal_(cos_coef, gain=self.fope_init_factor, generator=generator)

        if self.input_dim == self.output_dim:
            sin_coef += torch.eye(self.input_dim, device=sin_coef.device)
            cos_coef += torch.eye(self.input_dim, device=cos_coef.device)
        else:
            sin_coef += self.get_step_eye(sin_coef)
            cos_coef += self.get_step_eye(cos_coef)

        self.register_buffer("sin_coef", sin_coef.to(DEVICE), persistent=True)
        self.register_buffer("cos_coef", cos_coef.to(DEVICE), persistent=True)

    def get_step_eye(self, _param):
        import math

        _step_eye = torch.zeros_like(_param)

        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i * step < self.input_dim:
                _step_eye[..., i * step, i] = 1.0

        return _step_eye

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            raise NotImplementedError

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        batch_size, seq_len, hidden_size = x.shape
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            if self.fope_sep_head:
                pos_cos = freqs.cos().unsqueeze(1).expand(batch_size, self.config.num_key_value_heads, seq_len, -1)
                pos_sin = freqs.sin().unsqueeze(1).expand(batch_size, self.config.num_key_value_heads, seq_len, -1)
            else:
                pos_cos = freqs.cos()
                pos_sin = freqs.sin()

            if self.fope_sep_head:
                sin = torch.einsum("bhtD, hDd -> bhtd", pos_sin, self.sin_coef.float())
                cos = torch.einsum("bhtD, hDd -> bhtd", pos_cos, self.cos_coef.float())
            else:
                sin = torch.einsum("btD, Dd -> btd", pos_sin, self.sin_coef.float())
                cos = torch.einsum("btD, Dd -> btd", pos_cos, self.cos_coef.float())

            sin = F.pad(input=sin, pad=(0, self.head_dim // 2 - sin.size(-1)), mode="constant", value=1)
            cos = F.pad(input=cos, pad=(0, self.head_dim // 2 - cos.size(-1)), mode="constant", value=1)

            sin = torch.cat((sin, sin), dim=-1)
            cos = torch.cat((cos, cos), dim=-1)

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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

        # The implementation of RoPE has been refactored in Transformers V5, and
        # the following approach is used for compatibility.
        self.rope_init_fn: Callable = compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq: torch.Tensor
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        # Get mrope_section from rope_parameters_cfg (new format) or rope_scaling_cfg (old format)
        if config.rope_parameters_cfg is not None:
            self.mrope_section = config.rope_parameters_cfg.mrope_section
        elif config.rope_scaling_cfg is not None:
            self.mrope_section = config.rope_scaling_cfg.mrope_section
        else:
            self.mrope_section = None
        assert self.mrope_section is not None, "mrope_section is required for Qwen3VLTextRotaryEmbedding"

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
            # TODO: remove to(x.device) because from_hf has already moved the rotary_emb module to the correct device
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
    from xtuner.v1.model.compose.qwen3_vl.modeling_vision import Qwen3VLVisionConfig, Qwen3VLVisionRotaryEmbedding

    if isinstance(config, Qwen3VLVisionConfig):
        return Qwen3VLVisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)  # type: ignore[return-value]

    config = cast(TransformerConfig, config)
    rope_scaling_cfg = config.rope_scaling_cfg

    if rope_scaling_cfg is not None and rope_scaling_cfg.type == "qwen3_vl":
        return Qwen3VLTextRotaryEmbedding(config, device=device)
    elif rope_scaling_cfg is not None and rope_scaling_cfg.use_fope:
        logger.info("Using FoPE rotary embedding.")
        return FourierEmbedding(config, device=device)
    else:
        return RotaryEmbedding(config, device=device)
