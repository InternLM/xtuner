"""HTTP wire codec for routed_experts.

In-cluster transport keeps ``routed_experts`` as a raw ``np.ndarray`` so Ray
RPC can move it via plasma zero-copy.  When the value crosses the HTTP
boundary (``/v1/chat/completions``), FastAPI's ``jsonable_encoder`` cannot
handle ndarray, so we serialize to a JSON-safe dict and reconstruct on the
other side.  No stateful store actor is involved.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


_WIRE_KEYS = ("data", "shape", "dtype")


def to_wire(arr: np.ndarray) -> dict:
    """Encode an ``np.ndarray`` as a JSON-friendly dict for HTTP transport."""
    arr = np.ascontiguousarray(arr)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def from_wire(payload: Any) -> np.ndarray:
    """Decode an HTTP-wire payload back to ``np.ndarray``.

    Accepts the wire dict produced by :func:`to_wire`, or a plain ``np.ndarray``
    (in case the caller already has the in-cluster form — useful for transparent
    fallbacks).
    """
    if isinstance(payload, np.ndarray):
        return payload
    if not is_wire(payload):
        raise TypeError(f"Cannot decode routed_experts wire payload of type {type(payload)}")
    raw = base64.b64decode(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(tuple(payload["shape"]))
    return arr


def is_wire(payload: Any) -> bool:
    """True iff ``payload`` is the wire dict shape produced by
    :func:`to_wire`."""
    return isinstance(payload, dict) and all(k in payload for k in _WIRE_KEYS)
