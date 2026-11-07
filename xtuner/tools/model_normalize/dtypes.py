"""Optional dtype alignment against a reference structure map.

The map is a JSON object ``{tensor_name: dtype}``; values may also be
``[dtype, shape]`` pairs, as produced by the structure dumps used for release
verification. Only real float dtypes are honoured -- FP8 targets are ignored
because quantized tensors are produced by the converter itself.

Typical use: a training export keeps some weights in float32 while the
released reference model stores them in bfloat16. Passing the reference's
structure map to ``--dtype-map`` casts exactly those tensors, so the product
matches the reference release instead of drifting on storage dtype.
"""

from __future__ import annotations

import json
from pathlib import Path

ALIGNABLE_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def load_dtype_map(path: Path | None) -> dict[str, str]:
    """Load a ``tensor name -> dtype`` map, tolerating ``[dtype, shape]`` values."""
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    return {
        name: str(value[0] if isinstance(value, (list, tuple)) else value)
        for name, value in raw.items()
    }


def alignable_dtype(dtype_map: dict[str, str], name: str) -> str | None:
    """Return the dtype name to cast ``name`` to, or ``None`` to leave it alone."""
    dtype_name = dtype_map.get(name)
    return dtype_name if dtype_name in ALIGNABLE_DTYPES else None


def align_tensor(tensor, dtype_name: str) -> tuple[object, bool]:
    """Cast ``tensor`` to ``dtype_name``.

    Returns:
        tuple[object, bool]: the tensor (possibly cast) and whether a cast happened.
    """
    import torch

    target = getattr(torch, ALIGNABLE_DTYPES[dtype_name])
    if tensor.dtype == target:
        return tensor, False
    return tensor.to(target), True
