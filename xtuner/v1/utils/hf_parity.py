# Copyright (c) OpenMMLab. All rights reserved.
"""The DeepSeek-V4 HF-bit-parity switch.

DeepSeek-V4 carries several sites where XTuner deliberately computes something
*differently* from HuggingFace's ``modeling_deepseek_v4`` — a bf16 fast path where HF
uses fp32, a gathered sparse attention where HF materialises a dense masked softmax, a
reference-faithful low-precision Indexer score where HF upcasts. Each is a defensible
production choice, and each makes the two implementations disagree in the last few bits.

``XTUNER_V4_HF_PARITY=1`` switches every one of those sites onto HF's exact op sequence so
a whole-model forward reproduces HF **bitwise** (``torch.equal``), which is what the
decoder-layer parity tests assert. It is a *structural* correctness check: it proves the
layer wiring, weight mapping, masking, routing and rope bases are right, and deliberately
takes HF's arithmetic — including where HF is the less accurate of the two — so that no
tolerance has to stand in for "we think this difference is benign".

It is emphatically **not** a production mode. The parity paths trade throughput and, at
the attention site, memory that scales with ``seq_len**2``. Kernel-level numerics are
covered by the op- and module-level tests instead.

This module is the single source of truth for the switch. Sites that honour it:

* ``module.decoder_layer.deepseek_v4.hc_block`` — ``hc_pre`` / ``hc_post``
* ``model.moe.deepseek_v4.DeepSeekV4._hc_head_reduce_compute`` — the final stream collapse
* ``module.attention.dsa.sparse_attn`` — the top-k attention core
* ``module.attention.dsa.indexer`` — the Lightning Indexer's scoring path
"""

import os


__all__ = ["hf_parity_enabled", "set_hf_parity"]

# Read once at import so the flag is a constant for ``torch.compile`` tracing; tests flip
# it through :func:`set_hf_parity`, which is why every site calls the accessor rather than
# capturing the module global by value.
_HF_PARITY = os.getenv("XTUNER_V4_HF_PARITY", "0") == "1"


def hf_parity_enabled() -> bool:
    """Whether DeepSeek-V4 should take its HF-bit-parity paths.

    Returns:
        bool: ``True`` when ``XTUNER_V4_HF_PARITY=1`` or :func:`set_hf_parity` turned it on.
    """
    return _HF_PARITY


def set_hf_parity(enabled: bool) -> bool:
    """Turn the parity paths on or off for the current process.

    Exists because the env var is read at import time and the parity tests need to flip
    the mode after ``xtuner`` is already imported.

    Args:
        enabled (bool): Desired state.

    Returns:
        bool: The previous state, so callers can restore it.
    """
    global _HF_PARITY
    previous = _HF_PARITY
    _HF_PARITY = enabled
    return previous
