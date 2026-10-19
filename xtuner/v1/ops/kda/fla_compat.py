"""Compatibility helpers for FLA's KDA state-layout API."""

from __future__ import annotations

import inspect
from collections.abc import Callable


try:
    from fla.utils import USE_CUDA_GRAPH
except ImportError:  # FLA 0.5.2 removed the global USE_CUDA_GRAPH switch.
    USE_CUDA_GRAPH = False


def fla_state_layout_kwargs(
    func: Callable,
    transpose_state_layout: bool,
    hv: int | None = None,
) -> dict[str, bool | int]:
    """Translate XTuner's state-layout flag to the installed FLA KDA API.

    Args:
        func (Callable): FLA Python wrapper or Triton kernel whose signature is inspected.
        transpose_state_layout (bool): Whether XTuner uses the ``[N, H, V, K]`` state layout.
        hv (int, optional): Value-head count required by FLA 0.5.2 Triton kernels.

    Returns:
        dict[str, bool | int]: Keyword arguments using the API contract of ``func``.
    """
    parameters = inspect.signature(func).parameters

    # FLA 0.5.2 Python wrappers renamed the lowercase Python API from
    # ``transpose_state_layout`` to ``state_v_first``.
    if "state_v_first" in parameters:
        return {"state_v_first": transpose_state_layout}

    # FLA 0.5.2 Triton kernels use uppercase constexpr names. They also model GVA
    # with a separate value-head count, so ``HV`` and ``STATE_V_FIRST`` are both
    # required and must not be replaced by the old ``TRANSPOSE_STATE`` name.
    if "STATE_V_FIRST" in parameters:
        if "HV" not in parameters:
            raise TypeError(f"{func.__name__} has 'STATE_V_FIRST' but no required 'HV' parameter")
        if hv is None:
            raise ValueError(f"{func.__name__} requires the value-head count 'hv'")
        return {"HV": hv, "STATE_V_FIRST": transpose_state_layout}

    # FLA 0.4.2 Triton kernels use the old uppercase constexpr name
    # ``TRANSPOSE_STATE``.
    if "TRANSPOSE_STATE" in parameters:
        return {"TRANSPOSE_STATE": transpose_state_layout}

    # FLA 0.4.2 Python wrappers use the old lowercase API name. Some internal
    # functions additionally expose ``use_exp2``; XTuner's KDA gates always use
    # exp2, so enable it when available.
    if "transpose_state_layout" in parameters:
        kwargs = {"transpose_state_layout": transpose_state_layout}
        if "use_exp2" in parameters:
            kwargs["use_exp2"] = True
        return kwargs
    raise TypeError(
        f"{func.__name__} supports none of 'state_v_first', 'STATE_V_FIRST', 'TRANSPOSE_STATE', "
        "or 'transpose_state_layout'; unsupported flash-linear-attention installation?"
    )
