"""Compatibility helpers for flash-linear-attention (FLA) 0.4.2 and 0.5.2.

XTuner vendors FLA-derived kernels under ``xtuner/v1/ops`` that call into installed FLA Python
wrappers and Triton kernels. FLA 0.5.2 renamed the state-layout flag
(``transpose_state_layout`` -> ``state_v_first``) and split the query/value head counts into the
``H``/``HV`` constexprs, so this module centralizes that adaptation for ``xtuner/v1/ops/kda`` and
``xtuner/v1/ops/gated_deltanet``.

Every other keyword XTuner passes is accepted unchanged by both supported versions. Unknown
signatures raise instead of silently dropping the flag, so a future FLA rename fails loudly at the
call site rather than changing numerics.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import lru_cache

import torch


try:
    # Redundant alias marks this as a re-export for ``sequence_parallel_bwd``.
    from fla.utils import USE_CUDA_GRAPH as USE_CUDA_GRAPH
except ImportError:  # FLA 0.5.2 removed the global USE_CUDA_GRAPH switch.
    USE_CUDA_GRAPH = False


def fla_state_layout_kwargs(
    func: Callable,
    transpose_state_layout: bool,
    *,
    q: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
) -> dict[str, bool | int]:
    """Translate XTuner's state-layout flag to the installed FLA API of
    ``func``.

    ``func`` must be the FLA callable itself: passing one of XTuner's own functions whose
    signature happens to contain ``transpose_state_layout`` would silently produce the FLA 0.4.2
    form without adapting anything.

    Args:
        func (Callable): FLA Python wrapper or Triton kernel whose signature is inspected.
        transpose_state_layout (bool): Whether XTuner uses the ``[N, H, V, K]`` state layout.
        q (torch.Tensor | None): Query-side tensor of shape ``[B, T, H, K]``. Only required when
            ``func`` is a Triton kernel that takes the query-head count ``H``.
        v (torch.Tensor | None): Value-side tensor of shape ``[B, T, HV, V]``. Only required when
            ``func`` is a Triton kernel that models grouped value attention with ``HV``.

    Returns:
        dict[str, bool | int]: Keyword arguments using the API contract of ``func``.
    """
    parameters = _signature_parameters(func)

    # FLA 0.5.2 Python wrappers renamed the layout flag to ``state_v_first``.
    if "state_v_first" in parameters:
        return {"state_v_first": transpose_state_layout}

    # FLA 0.5.2 Triton kernels use uppercase constexprs and model grouped value attention with a
    # separate value-head count. ``merge_fwd_bwd_kernel`` is the exception and only takes ``HV``.
    if "STATE_V_FIRST" in parameters:
        kwargs: dict[str, bool | int] = {"HV": _head_count(v, "v"), "STATE_V_FIRST": transpose_state_layout}
        if "H" in parameters:
            kwargs["H"] = _head_count(q, "q")
        return kwargs

    # FLA 0.4.2 Triton kernels use ``TRANSPOSE_STATE`` and the query-head count ``H``.
    if "TRANSPOSE_STATE" in parameters:
        kwargs = {"TRANSPOSE_STATE": transpose_state_layout}
        if "H" in parameters:
            kwargs["H"] = _head_count(q, "q")
        return kwargs

    # FLA 0.4.2 Python wrappers use the lowercase flag. Some internal functions additionally expose
    # ``use_exp2``; XTuner's KDA gates always use exp2, so enable it when the install accepts it.
    if "transpose_state_layout" in parameters:
        kwargs = {"transpose_state_layout": transpose_state_layout}
        if "use_exp2" in parameters:
            kwargs["use_exp2"] = True
        return kwargs

    raise _unsupported_installation_error(func)


def _head_count(tensor: torch.Tensor | None, name: str) -> int:
    if tensor is None:
        raise TypeError(f"fla_state_layout_kwargs needs `{name}=` to pass the head count to this FLA Triton kernel")
    return tensor.shape[-2]


def _unwrap_triton(func: Callable) -> Callable:
    # ``@triton.heuristics`` / ``@triton.autotune`` / ``@triton.jit`` wrappers are not plain
    # callables, so ``inspect.signature`` on them raises TypeError. Unwrap the ``.fn`` chain down to
    # the underlying Python function, which carries the constexpr names used for branching above.
    fn = func
    while hasattr(fn, "fn"):
        fn = fn.fn
    return fn


@lru_cache(maxsize=None)
def _signature_parameters(func: Callable) -> Mapping[str, inspect.Parameter]:
    # The callables are module-level FLA objects, so caching by identity is safe; this keeps the
    # per-layer signature probe (and the ``.fn`` unwrap) off the training hot path.
    return inspect.signature(_unwrap_triton(func)).parameters


def _unsupported_installation_error(func: Callable) -> TypeError:
    try:
        import fla

        version = getattr(fla, "__version__", "") or "<unknown>"
    except ImportError:
        version = "<unknown>"
    return TypeError(
        f"{_unwrap_triton(func).__name__} uses none of the state-layout APIs known to xtuner's "
        f"FLA compat layer; installed flash-linear-attention version: {version}, "
        "supported: 0.4.2, 0.5.2"
    )
