"""Kimi Delta Attention kernels.

``chunk_kda`` is XTuner's compile-friendly wrap of FLA's chunked KDA kernels, mirroring what
``xtuner/v1/ops/gated_deltanet`` does for GatedDeltaNet. ``XTUNER_HF_IMPL=1`` falls back to FLA's
own entry point, which is the numerical reference but breaks the graph when compiled.
"""

import os


_TRUTHY = {"true", "1", "yes", "on"}


def _hf_impl_enabled() -> bool:
    return os.getenv("XTUNER_HF_IMPL", "").strip().lower() in _TRUTHY


def get_fused_kda_gate_fn():
    if _hf_impl_enabled():
        from fla.ops.kda.gate import fused_kda_gate as _fla_fused_kda_gate

        return _fla_fused_kda_gate
    from .fused_kda_gate import fused_kda_gate as _xtuner_fused_kda_gate

    return _xtuner_fused_kda_gate


def get_causal_conv1d_fn():
    if _hf_impl_enabled():
        from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d

        return _fla_causal_conv1d
    from .causal_conv1d import causal_conv1d as _xtuner_causal_conv1d

    return _xtuner_causal_conv1d


def get_chunk_kda_fn():
    if _hf_impl_enabled():
        from fla.ops.kda import chunk_kda as _fla_chunk_kda

        return _fla_chunk_kda
    from .chunk_kda import chunk_kda as _xtuner_chunk_kda

    return _xtuner_chunk_kda


__all__ = ["get_causal_conv1d_fn", "get_chunk_kda_fn", "get_fused_kda_gate_fn"]
