# modified from
# https://github.com/fla-org/flash-linear-attention/tree/v0.4.2/fla/modules/conv/causal_conv1d.py
# to support torch.compile
"""Compile-friendly causal short convolution for KDA.

FLA's ``causal_conv1d`` dispatcher derives its chunk table from ``cu_seqlens`` and reaches Triton
launch helpers that dynamo cannot trace, so KDA's three per-layer convolutions (q/k/v) each broke
the enclosing compiled region -- the largest single source of graph breaks left after
``chunk_kda.py``. This wraps the same ``causal_conv1d_fwd``/``causal_conv1d_bwd`` Triton entry
points in ``torch.library.custom_op``, so the numerics are FLA's and the graph stays whole.

Only KDA's call shape is supported: no residual, no ``initial_state``, no returned final state,
and the Triton backend. Anything else should keep using FLA's own dispatcher.
"""

import torch
from fla.modules.conv.triton.ops import causal_conv1d_bwd as _fla_causal_conv1d_bwd
from fla.modules.conv.triton.ops import causal_conv1d_fwd as _fla_causal_conv1d_fwd
from fla.utils import input_guard


LIBRARY_NAME = "xtuner_kda"


@torch.library.custom_op(
    f"{LIBRARY_NAME}::causal_conv1d_fwd",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor? bias, str? activation, Tensor? cu_seqlens) -> Tensor",
)
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    y, _ = _fla_causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    return y


@causal_conv1d_fwd.register_fake
def _causal_conv1d_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::causal_conv1d_bwd",
    mutates_args=(),
    schema="(Tensor x, Tensor dy, Tensor weight, Tensor? bias, str? activation, Tensor? cu_seqlens) "
    "-> (Tensor, Tensor, Tensor?)",
)
def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    dx, dw, db, _, _ = _fla_causal_conv1d_bwd(
        x=x,
        dy=dy,
        dht=None,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=None,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    return dx, dw, db


@causal_conv1d_bwd.register_fake
def _causal_conv1d_bwd_fake(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return (
        torch.empty_like(x),
        torch.empty_like(weight),
        None if bias is None else torch.empty_like(bias),
    )


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        y = torch.ops.xtuner_kda.causal_conv1d_fwd(x, weight, bias, activation, cu_seqlens)
        ctx.save_for_backward(x, weight, bias, cu_seqlens)
        ctx.activation = activation
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor):
        x, weight, bias, cu_seqlens = ctx.saved_tensors
        dx, dw, db = torch.ops.xtuner_kda.causal_conv1d_bwd(x, dy, weight, bias, ctx.activation, cu_seqlens)
        return dx, dw, db, None, None


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Causal depthwise short convolution, traceable by ``torch.compile``.

    Args:
        x (torch.Tensor): Input of shape ``[B, T, D]``.
        weight (torch.Tensor): Depthwise kernel of shape ``[D, W]``.
        bias (torch.Tensor | None): Optional per-channel bias ``[D]``.
        activation (str | None): ``"silu"``/``"swish"`` or ``None``.
        cu_seqlens (torch.Tensor | None): Packed-sequence offsets, so the convolution never reads
            across a document boundary.

    Returns:
        tuple[torch.Tensor, None]: The convolved tensor, and ``None`` for the final state, which
        this entry point does not produce.
    """
    for unsupported in ("residual", "initial_state", "output_final_state", "cp_context"):
        if kwargs.get(unsupported):
            raise NotImplementedError(
                f"xtuner's compile-friendly causal_conv1d does not support {unsupported!r}; use "
                "fla.modules.conv.causal_conv1d directly."
            )
    return CausalConv1dFunction.apply(x, weight, bias, activation, cu_seqlens), None
