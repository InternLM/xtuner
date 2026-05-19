"""Unit tests for `DualRotaryEmbedding` and the dual-rope fields on
`RopeParametersConfig`.

These tests are CPU-only and avoid any GPU / distributed setup. They pin the
core dual-rope invariant: each branch (`use_compressed=False/True`) must be
bit-identical to a single-rope `RotaryEmbedding` configured with the matching
theta. This protects the existing yarn / default rope numerics from drifting
when the dual-rope path is added.
"""

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.rope.rope import (
    DualRotaryEmbedding,
    RopeParametersConfig,
    RotaryEmbedding,
)


# DeepSeek-V4-Flash yarn parameters; mirrors the local HF config.
_YARN_KWARGS: dict[str, Any] = {
    "rope_type": "yarn",
    "factor": 16.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 65536,
}


def _make_config(
    *,
    rope_theta: float,
    compress_rope_theta: float | None = None,
    compress_ratios: list[int] | None = None,
    max_position_embeddings: int = 64,
) -> TransformerConfig:
    """Build a minimal `TransformerConfig` for rope-only tests."""
    rope_kwargs: dict[str, Any] = {"rope_theta": rope_theta, **_YARN_KWARGS}
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
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=2, head_dim=32),
        rope_parameters_cfg=RopeParametersConfig(**rope_kwargs),
    )


class TestDualRotaryEmbedding:
    def setup_method(self) -> None:
        self.seq_len = 64
        self.x = torch.zeros(1, self.seq_len, 64, dtype=torch.float32)
        self.position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

    def test_use_compressed_false_matches_single_rope(self) -> None:
        dual_cfg = _make_config(rope_theta=10000.0, compress_rope_theta=160000.0)
        single_cfg = _make_config(rope_theta=10000.0)

        dual = DualRotaryEmbedding(dual_cfg, device=torch.device("cpu"))
        single = RotaryEmbedding(single_cfg, device=torch.device("cpu"))

        dual_cos, dual_sin = dual(self.x, self.position_ids, use_compressed=False)
        single_cos, single_sin = single(self.x, self.position_ids)

        # Bit-identical guards the invariant that the dense branch preserves the
        # existing single-rope numerics, including yarn `attention_scaling`.
        assert torch.equal(dual_cos, single_cos)
        assert torch.equal(dual_sin, single_sin)

    def test_use_compressed_true_matches_high_theta_rope(self) -> None:
        dual_cfg = _make_config(rope_theta=10000.0, compress_rope_theta=160000.0)
        single_cfg = _make_config(rope_theta=160000.0)

        dual = DualRotaryEmbedding(dual_cfg, device=torch.device("cpu"))
        single = RotaryEmbedding(single_cfg, device=torch.device("cpu"))

        dual_cos, dual_sin = dual(self.x, self.position_ids, use_compressed=True)
        single_cos, single_sin = single(self.x, self.position_ids)

        assert torch.equal(dual_cos, single_cos)
        assert torch.equal(dual_sin, single_sin)

    def test_yarn_applied_to_both(self) -> None:
        dual_cfg = _make_config(rope_theta=10000.0, compress_rope_theta=160000.0)
        dual = DualRotaryEmbedding(dual_cfg, device=torch.device("cpu"))

        # yarn's `attention_scaling` is base-independent; we only need to check the
        # single stored value is non-1.0 to confirm yarn was wired in for both bases.
        assert dual.attention_scaling != 1.0

        # Sanity check: inv_freq tensors differ between the two bases.
        assert not torch.equal(dual.inv_freq_dense, dual.inv_freq_compressed)

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
