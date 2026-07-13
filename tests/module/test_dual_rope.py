"""Unit tests for `DualRotaryEmbedding` and the dual-rope fields on
`RopeParametersConfig`.

These tests are CPU-only and avoid any GPU / distributed setup. They pin the
core dual-rope invariant: each branch (`use_compressed=False/True`) must be
bit-identical to a single-rope `RotaryEmbedding` configured with the matching
theta. This protects the existing yarn / default rope numerics from drifting
when the dual-rope path is added.
"""

import math
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module.attention.dsa import DSAConfig
from xtuner.v1.module.rope.rope import (
    DualRotaryEmbedding,
    RopeParametersConfig,
)


# DeepSeek-V4-Flash yarn parameters; mirrors the local HF config.
_YARN_KWARGS: dict[str, Any] = {
    "rope_type": "yarn",
    "factor": 16.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 65536,
    # V4 truncates the yarn correction range (`floor` / `ceil`); `RopeParametersConfig`
    # defaults to False, so `DeepSeekV4Config` pins it and these tests must too.
    "truncate": True,
}


def _make_config(
    *,
    rope_theta: float,
    compress_rope_theta: float | None = None,
    compress_ratios: list[int] | None = None,
    max_position_embeddings: int = 64,
    yarn: bool = True,
) -> TransformerConfig:
    """Build a minimal `TransformerConfig` for rope-only tests."""
    rope_kwargs: dict[str, Any] = {"rope_theta": rope_theta}
    if yarn:
        rope_kwargs |= _YARN_KWARGS
    if compress_rope_theta is not None:
        rope_kwargs["compress_rope_theta"] = compress_rope_theta
    if compress_ratios is not None:
        rope_kwargs["compress_ratios"] = compress_ratios

    return TransformerConfig(
        vocab_size=128,
        max_position_embeddings=max_position_embeddings,
        eos_token_id=0,
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        # `DualRotaryEmbedding` is DeepSeek-V4-specific and sizes `inv_freq` from
        # `qk_rope_head_dim`, which only the DSA attention config exposes.
        attention=DSAConfig(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            qk_rope_head_dim=16,
            q_lora_rank=16,
            o_lora_rank=16,
            o_groups=2,
            sliding_window=8,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=8,
        ),
        rope_parameters_cfg=RopeParametersConfig(**rope_kwargs),
    )


def _reference_inv_freq(*, base: float, original_seq_len: int, dim: int = 16) -> torch.Tensor:
    """Reimplementation of the V4 reference `precompute_freqs_cis` frequency ramp.

    An independent oracle for `DualRotaryEmbedding`: XTuner routes through transformers'
    `ROPE_INIT_FUNCTIONS`, so comparing against that would be circular. `original_seq_len == 0`
    is the reference's "interpolation off" switch.

    Args:
        base (float): RoPE theta.
        original_seq_len (int): Pretraining context length; ``0`` disables interpolation.
        dim (int): Rope-carrying head dim (``qk_rope_head_dim``). Defaults to ``16``.

    Returns:
        torch.Tensor: `inv_freq` of shape ``[dim // 2]``.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len <= 0:
        return freqs

    def find_correction_dim(num_rotations: float) -> float:
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(find_correction_dim(_YARN_KWARGS["beta_fast"])), 0)
    high = min(math.ceil(find_correction_dim(_YARN_KWARGS["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    return freqs / _YARN_KWARGS["factor"] * (1 - smooth) + freqs * smooth


class TestDualRotaryEmbedding:
    def setup_method(self) -> None:
        self.seq_len = 64
        self.x = torch.zeros(1, self.seq_len, 64, dtype=torch.float32)
        self.position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

    def test_use_compressed_false_is_plain_rope(self) -> None:
        # The dense branch is plain RoPE at `rope_theta`: DeepSeek-V4 builds its
        # sliding-window layers with position interpolation switched off
        # (`original_seq_len=0` in the reference `precompute_freqs_cis`), so the
        # configured `rope_type` (yarn) must not reach this base.
        dual = DualRotaryEmbedding(
            _make_config(rope_theta=10000.0, compress_rope_theta=160000.0), device=torch.device("cpu")
        )

        assert torch.equal(dual.inv_freq_dense.cpu(), _reference_inv_freq(base=10000.0, original_seq_len=0))

    def test_use_compressed_true_is_yarn_rope(self) -> None:
        dual = DualRotaryEmbedding(
            _make_config(rope_theta=10000.0, compress_rope_theta=160000.0), device=torch.device("cpu")
        )

        expected = _reference_inv_freq(base=160000.0, original_seq_len=_YARN_KWARGS["original_max_position_embeddings"])
        torch.testing.assert_close(dual.inv_freq_compressed.cpu(), expected)

    def test_cos_sin_are_not_mscale_rescaled(self) -> None:
        dual = DualRotaryEmbedding(
            _make_config(rope_theta=10000.0, compress_rope_theta=160000.0), device=torch.device("cpu")
        )

        # V4 never rescales cos/sin by yarn's mscale — its reference returns
        # unit-modulus `polar(ones, freqs)` and HF pins `attention_factor=1.0`.
        assert dual.attention_scaling == 1.0
        cos, _ = dual(self.x, self.position_ids, use_compressed=False)
        assert torch.equal(cos[0, 0], torch.ones_like(cos[0, 0]))

    def test_config_roundtrip(self) -> None:
        original = RopeParametersConfig(
            rope_theta=10000.0,
            compress_rope_theta=160000.0,
            compress_ratios=[0, 0, 4, 128, 4, 128, 4, 0],
            **_YARN_KWARGS,
        )
        recovered = RopeParametersConfig(**original.to_rope_parameters_dict())  # type: ignore[arg-type]

        assert recovered.compress_rope_theta == 160000.0
        assert recovered.compress_ratios == [0, 0, 4, 128, 4, 128, 4, 0]
        assert recovered == original

    def test_config_validator_rejects_partial(self) -> None:
        # `compress_ratios` without `compress_rope_theta` is rejected — otherwise the
        # config would silently fall through to single-rope and lose dual-rope behavior.
        with pytest.raises(ValidationError):
            RopeParametersConfig(
                rope_theta=10000.0,
                compress_ratios=[0, 0, 4, 128],
            )
