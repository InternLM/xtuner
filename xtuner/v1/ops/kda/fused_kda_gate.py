# modified from
# https://github.com/fla-org/flash-linear-attention/tree/v0.4.2/fla/ops/kda/gate.py
# to support torch.compile
"""Compile-friendly ``fused_kda_gate``.

FLA's ``fused_kda_gate`` carries ``@torch.compiler.disable``, so every KDA layer's forget gate
breaks the enclosing compiled region ("Skip calling `torch.compiler.disable()`d function"). The
kernels underneath -- ``kda_gate_fwd``/``kda_gate_bwd`` -- have no such restriction, so wrapping
them in ``torch.library.custom_op`` lets dynamo trace straight through, the same route
``chunk_kda.py`` takes for the chunked kernel.
"""

import torch
from fla.ops.kda.gate import kda_gate_bwd as _fla_kda_gate_bwd
from fla.ops.kda.gate import kda_gate_fwd as _fla_kda_gate_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


LIBRARY_NAME = "xtuner_kda"


@torch.library.custom_op(
    f"{LIBRARY_NAME}::kda_gate_fwd",
    mutates_args=(),
    schema="(Tensor g, Tensor A_log, Tensor? dt_bias, float? lower_bound) -> Tensor",
)
def kda_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    # `output_dtype` stays at FLA's fp32 default: the exponentiated decay this feeds is
    # precision-sensitive, which is also why `_gate_param` pins A_log/dt_bias to fp32.
    return _fla_kda_gate_fwd(g=g, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound)


@kda_gate_fwd.register_fake
def _kda_gate_fwd_fake(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    return torch.empty_like(g, dtype=torch.float32)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::kda_gate_bwd",
    mutates_args=(),
    schema="(Tensor g, Tensor A_log, Tensor? dt_bias, Tensor dyg, float? lower_bound) -> (Tensor, Tensor, Tensor?)",
)
def kda_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return _fla_kda_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=lower_bound)


@kda_gate_bwd.register_fake
def _kda_gate_bwd_fake(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # `kda_gate_bwd` casts each gradient back to its own tensor: dg is `type_as(g)`, dA is
    # `view_as(A_log).type_as(A_log)`, and dbias is the `[H*K]` column sum cast `.to(dt_bias)`.
    return (
        torch.empty_like(g),
        torch.empty_like(A_log),
        None if dt_bias is None else torch.empty_like(dt_bias),
    )


class KDAGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None,
        lower_bound: float | None,
    ) -> torch.Tensor:
        yg = torch.ops.xtuner_kda.kda_gate_fwd(g, A_log, dt_bias, lower_bound)
        ctx.save_for_backward(g, A_log, dt_bias)
        ctx.lower_bound = lower_bound
        return yg

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dyg: torch.Tensor):
        g, A_log, dt_bias = ctx.saved_tensors
        dg, dA, dbias = torch.ops.xtuner_kda.kda_gate_bwd(g, A_log, dt_bias, dyg, ctx.lower_bound)
        return dg, dA, dbias, None


def fused_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """KDA forget gate, traceable by ``torch.compile``.

    Runs the same ``kda_gate_fwd``/``kda_gate_bwd`` kernels as ``fla.ops.kda.gate``, only behind
    custom ops so a compiled caller does not graph-break.

    Args:
        g (torch.Tensor): Raw gate projection of shape ``[B, T, H, K]``.
        A_log (torch.Tensor): Per-head fp32 decay parameter.
        dt_bias (torch.Tensor | None): Per-head fp32 bias.
        lower_bound (float | None): Safe-gate lower bound, or ``None`` to disable it.

    Returns:
        torch.Tensor: The fp32 gate in log space, shaped like ``g``.
    """
    if kwargs.get("output_dtype") not in (None, torch.float32):
        raise NotImplementedError(
            "xtuner's compile-friendly fused_kda_gate always emits fp32; use fla.ops.kda.gate "
            "directly if a caller needs another dtype."
        )
    return KDAGateFunction.apply(g, A_log, dt_bias, lower_bound)
