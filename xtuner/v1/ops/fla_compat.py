"""Compatibility helpers for flash-linear-attention (FLA) 0.4.2 and 0.5.2.

XTuner vendors FLA-derived kernels under ``xtuner/v1/ops`` that call into installed
FLA Python wrappers and Triton kernels. FLA 0.5.2 renamed several of those APIs
relative to 0.4.2, so this module centralizes the version-adaptation logic shared
by ``xtuner/v1/ops/kda`` and ``xtuner/v1/ops/gated_deltanet``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any


try:
    # Redundant alias marks this as a re-export for ``sequence_parallel_bwd``.
    from fla.utils import USE_CUDA_GRAPH as USE_CUDA_GRAPH
except ImportError:  # FLA 0.5.2 removed the global USE_CUDA_GRAPH switch.
    USE_CUDA_GRAPH = False

# Lowercase Python-API renames between FLA 0.4.2 and 0.5.2. Keys are the 0.4.2
# kwarg names XTuner passes; values are the 0.5.2 names accepted instead.
_KWARG_ALIASES = {
    "transpose_state_layout": "state_v_first",
    "use_exp2": None,  # dropped entirely in 0.5.2; XTuner's default (exp2) matches
}


def _unwrap_triton(func: Callable) -> Callable:
    # ``@triton.heuristics`` / ``@triton.autotune`` wrappers are not plain
    # callables, so ``inspect.signature`` on them raises TypeError. Unwrap the
    # ``.fn`` chain down to the underlying Python function, which carries the
    # constexpr parameter names used for branching below.
    fn = func
    while hasattr(fn, "fn"):
        fn = fn.fn
    return fn


def _signature_parameters(func: Callable) -> Mapping[str, inspect.Parameter]:
    return inspect.signature(_unwrap_triton(func)).parameters


def _unsupported_installation_error(func: Callable, detail: str | None = None) -> TypeError:
    try:
        import fla

        version = getattr(fla, "__version__", "") or "<unknown>"
    except ImportError:
        version = "<unknown>"
    problem = (
        f"{_unwrap_triton(func).__name__} {detail}"
        if detail
        else (
            f"{_unwrap_triton(func).__name__} uses none of the state-layout/kwarg APIs "
            "known to xtuner's FLA compat layer"
        )
    )
    return TypeError(f"{problem}; installed flash-linear-attention version: {version}, supported: 0.4.2, 0.5.2")


def fla_state_layout_kwargs(func: Callable, transpose_state_layout: bool, *, hv: int, h: int) -> dict[str, bool | int]:
    """Translate XTuner's state-layout flag to the installed FLA KDA API.

    Args:
        func (Callable): FLA Python wrapper or Triton kernel whose signature is inspected.
        transpose_state_layout (bool): Whether XTuner uses the ``[N, H, V, K]`` state layout.
        hv (int): Value-head count, passed as the ``HV`` constexpr on FLA 0.5.2 Triton kernels.
        h (int): Query-head count, passed as the ``H`` constexpr where the kernel still takes it.

    Returns:
        dict[str, bool | int]: Keyword arguments using the API contract of ``func``.
    """
    parameters = _signature_parameters(func)

    # FLA 0.5.2 Python wrappers renamed the lowercase Python API from
    # ``transpose_state_layout`` to ``state_v_first``.
    if "state_v_first" in parameters:
        return {"state_v_first": transpose_state_layout}

    # FLA 0.5.2 Triton kernels use uppercase constexpr names and model GVA with
    # a separate value-head count. Some of them still take the query-head count as
    # ``H`` (``merge_fwd_bwd_kernel`` is the exception and only takes ``HV``).
    if "STATE_V_FIRST" in parameters:
        state_kwargs: dict[str, bool | int] = {"HV": hv, "STATE_V_FIRST": transpose_state_layout}
        if "H" in parameters:
            state_kwargs["H"] = h
        return state_kwargs

    # FLA 0.4.2 Triton kernels use the old uppercase constexpr ``TRANSPOSE_STATE``
    # and take the query-head count as ``H``.
    if "TRANSPOSE_STATE" in parameters:
        kwargs: dict[str, bool | int] = {"TRANSPOSE_STATE": transpose_state_layout}
        if "H" in parameters:
            kwargs["H"] = h
        return kwargs

    # FLA 0.4.2 Python wrappers use the old lowercase API name. Some internal
    # functions additionally expose ``use_exp2``; XTuner's KDA gates always use
    # exp2, so enable it when available.
    if "transpose_state_layout" in parameters:
        kwargs = {"transpose_state_layout": transpose_state_layout}
        if "use_exp2" in parameters:
            kwargs["use_exp2"] = True
        return kwargs

    raise _unsupported_installation_error(func)


def fla_call_kwargs(func: Callable, **kwargs: Any) -> dict[str, Any]:
    """Translate XTuner keyword arguments to the installed FLA API of ``func``.

    For each kwarg XTuner passes, this resolves the name the installed FLA
    version expects:

    * kept as-is when ``func`` accepts it;
    * renamed via ``_KWARG_ALIASES`` when only the 0.5.2 rename is accepted
      (e.g. ``transpose_state_layout`` -> ``state_v_first``);
    * dropped when the kwarg was removed in 0.5.2 and XTuner's value matches
      the 0.5.2 behavior (e.g. ``use_exp2=True`` is the only 0.5.2 mode);
    * rejected otherwise, so silent behavioral drift cannot pass unnoticed.

    Args:
        func (Callable): FLA Python wrapper or Triton kernel whose signature is inspected.
        **kwargs (Any): Keyword arguments in the FLA 0.4.2 calling convention.

    Returns:
        dict[str, Any]: Keyword arguments using the API contract of ``func``.
    """
    parameters = _signature_parameters(func)
    resolved: dict[str, Any] = {}
    for name, value in kwargs.items():
        if name in parameters:
            resolved[name] = value
            continue
        alias = _KWARG_ALIASES.get(name)
        if alias is not None and alias in parameters:
            resolved[alias] = value
        elif name in _KWARG_ALIASES:
            if name == "use_exp2" and value is not True:
                raise TypeError(
                    f"{_unwrap_triton(func).__name__} does not support '{name}'; "
                    "FLA 0.5.2 always uses exp2, pass use_exp2=True"
                )
            # The kwarg no longer exists and XTuner's value matches the new
            # default behavior; omit it.
        else:
            raise _unsupported_installation_error(func, f"does not support kwarg '{name}'")
    return resolved


def fla_call(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Invoke an FLA callable with 0.4.2-convention kwargs adapted to the
    install.

    Positional arguments are forwarded unchanged; only keyword arguments are
    translated.

    Args:
        func (Callable): FLA Python wrapper or Triton kernel to call.
        *args (Any): Positional arguments forwarded to ``func`` as-is.
        **kwargs (Any): Keyword arguments in the FLA 0.4.2 calling convention.

    Returns:
        Any: The return value of ``func``.
    """
    return func(*args, **fla_call_kwargs(func, **kwargs))
