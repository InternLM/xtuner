"""GatedDeltaNet op-level dispatchers.

`XTUNER_HF_IMPL` controls which implementations XTuner's `GatedDeltaNet` module uses,
mirroring how `xtuner/v1/ops/attn_imp.py::get_attn_impl_fn` and the rms_norm selector
switch between fast / fused paths and HF-exact paths. Under `XTUNER_HF_IMPL=true`:

* `chunk_gated_delta_rule` is the canonical `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  (same callable HF's `Qwen3_5GatedDeltaNet` uses), bypassing XTuner's
  `torch.library.custom_op` wrap.
* `causal_conv1d_fn` is the high-level `causal_conv1d.causal_conv1d_fn` adapted to XTuner's
  channel-last call site: the adapter transposes to channel-first, calls the package wrapper
  with ``seq_idx=None`` (HF's non-packed convention), and transposes back. XTuner's own wrap
  binds the channel-last/seq_idx convention together via an internal transpose, which gives a
  different backward op graph from HF's call pattern even though forward is bitwise.

These switches are only meant for the bitwise-parity tests. Production / training stays on the
XTuner path (compile-friendly custom_op wraps + seq_idx-aware kernel dispatch).
"""

import os


_TRUTHY = {"true", "1", "yes", "on"}


def _hf_impl_enabled() -> bool:
    return os.getenv("XTUNER_HF_IMPL", "").strip().lower() in _TRUTHY


def _hf_causal_conv1d_adapter(x, weight, bias, activation, seq_idx):
    from causal_conv1d import causal_conv1d_fn as _hf_causal_conv1d_fn

    # XTuner's GatedDeltaNet supplies `x` in channel-last (``(batch, seq, dim)``) and always
    # passes a `seq_idx` tensor; HF's call site uses channel-first and `seq_idx=None` for
    # non-packed batches. Adapt: transpose, drop seq_idx, transpose back.
    x_cf = x.transpose(1, 2)
    out = _hf_causal_conv1d_fn(x=x_cf, weight=weight, bias=bias, activation=activation, seq_idx=None)
    return out.transpose(1, 2)


def get_chunk_gated_delta_rule_fn():
    if _hf_impl_enabled():
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _hf_chunk_gated_delta_rule

        return _hf_chunk_gated_delta_rule
    from .chunk_gated_delta_rule import chunk_gated_delta_rule as _xtuner_chunk_gated_delta_rule

    return _xtuner_chunk_gated_delta_rule


def get_causal_conv1d_fn():
    if _hf_impl_enabled():
        return _hf_causal_conv1d_adapter
    from .causal_conv1d import causal_conv1d_fn as _xtuner_causal_conv1d_fn

    return _xtuner_causal_conv1d_fn
