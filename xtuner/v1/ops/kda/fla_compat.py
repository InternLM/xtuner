"""Compatibility helpers for FLA's KDA state-layout API.

FLA 0.4.2 exposes ``use_exp2`` and ``transpose_state_layout`` on KDA's internal
chunk kernels. FLA 0.5.2 always uses exp2 gates and renames the transposed state
layout to ``state_v_first``. XTuner keeps its public ``transpose_state_layout``
name so model code and checkpoint layouts remain stable.
"""

from __future__ import annotations


try:
    from fla.utils import USE_CUDA_GRAPH
except ImportError:  # FLA 0.5.2 removed the global USE_CUDA_GRAPH switch.
    USE_CUDA_GRAPH = False

import inspect


def fla_state_layout_kwargs(func, transpose_state_layout: bool) -> dict[str, bool]:
    """Translate XTuner's state-layout flag to the installed FLA API."""
    parameters = inspect.signature(func).parameters
    if "state_v_first" in parameters:
        return {"state_v_first": transpose_state_layout}
    if "transpose_state_layout" in parameters:
        kwargs = {"transpose_state_layout": transpose_state_layout}
        if "use_exp2" in parameters:
            kwargs["use_exp2"] = True
        return kwargs
    raise TypeError(
        f"{func.__name__} supports neither 'state_v_first' nor 'transpose_state_layout'; "
        "unsupported flash-linear-attention installation?"
    )
