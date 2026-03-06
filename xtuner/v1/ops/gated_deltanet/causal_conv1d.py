# Copyright (c) 2024, Tri Dao.

import torch
import torch.nn.functional as F

import causal_conv1d_cuda


LIBRARY_NAME = "DaoAILab"


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_fwd_cpp", mutates_args={"out", "final_states_out"})
def _causal_conv1d_fwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> None:
    if seq_idx is not None:
        # If seq_idx is provided, we must use channel last layout
        x = x.transpose(1, 2)
        out = out.transpose(1, 2)
    causal_conv1d_cuda.causal_conv1d_fwd(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        out,
        final_states_out,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_bwd_cpp", mutates_args={
    "dfinal_states",
    "dx",
    "dweight",
    "dbias",
    "dinitial_states",
})
def _causal_conv1d_bwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor | None,
    dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> None:
    if seq_idx is not None:
        # If seq_idx is provided, we must use channel last layout
        x = x.transpose(1, 2)
        dout = dout.transpose(1, 2)
        dx = dx.transpose(1, 2)
    causal_conv1d_cuda.causal_conv1d_bwd(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        dx,
        dweight,
        dbias,
        dinitial_states,
        silu_activation,
    )

def causal_conv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_fwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        out=out,
        final_states_out=final_states_out,
        silu_activation=silu_activation,
    )
    return out

def causal_conv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor | None,
    return_dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor | None]:
    if seq_idx is None:
        batch_size, dim = x.size()[:2]
    else:
        batch_size, _, dim = x.size()[:3]
    width = weight.size(-1)

    if dx is None:
        dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    dbias = None
    if bias is not None:
        dbias = torch.zeros_like(bias, dtype=torch.float32)
    dinitial_states = None
    if return_dinitial_states:
        dinitial_states = torch.empty(batch_size, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)

    _causal_conv1d_bwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        dout=dout,
        seq_idx=seq_idx,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        dinitial_states=dinitial_states,
        silu_activation=silu_activation,
    )

    dweight = dweight.type_as(weight)
    if dbias is not None:
        dbias = dbias.type_as(bias)
    return dx, dweight, dbias, dinitial_states

class CausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        if seq_idx is not None:
            assert (
                initial_states is None
            ), "initial_states must be None if seq_idx is not None"
            assert (
                not return_final_states
            ), "If seq_idx is not None, we don't return final_states_out"
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        if initial_states is not None and (
            initial_states.stride(2) != 1 and initial_states.stride(1) != 1
        ):
            initial_states = initial_states.contiguous()
        if return_final_states:
            assert (
                x.stride(1) == 1
            ), "Only channel-last layout support returning final_states_out"
            if final_states_out is not None:
                assert (
                    final_states_out.stride(2) == 1 or final_states_out.stride(1) == 1
                )
            else:
                batch, dim, seqlen = x.shape
                width = weight.shape[1]
                final_states_out = torch.empty(
                    batch, width - 1, dim, device=x.device, dtype=x.dtype
                ).transpose(1, 2)
        else:
            final_states_out = None
        ctx.activation = activation in ["silu", "swish"]
        out = causal_conv1d_fwd_function(
            x, weight, bias, seq_idx, initial_states, final_states_out, ctx.activation
        )
        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.return_final_states = return_final_states
        ctx.return_dinitial_states = (
            initial_states is not None and initial_states.requires_grad
        )
        return out if not return_final_states else (out, final_states_out)

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias, dinitial_states = causal_conv1d_bwd_function(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            None,
            ctx.return_dinitial_states,
            ctx.activation,
        )
        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            dinitial_states if initial_states is not None else None,
            None,
            None,
            None,
        )


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    return CausalConv1dFn.apply(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        return_final_states,
        final_states_out,
        activation,
    )