"""FLA's sequence-sharded KDA kernels on XTuner's existing SP process group.

FLA calls its operator context ``FLACPContext``. It is constructed inside custom ops
from global packed offsets and the SP group; it does not define another mesh dimension.
Keeping this object and data-dependent local chunk tables inside the ops lets Dynamo
trace the surrounding attention module without a graph break.
"""

import torch
from fla.modules.conv.cp import CausalConv1dFunctionCP
from fla.modules.conv.triton.ops import causal_conv1d_bwd, causal_conv1d_fwd
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.cp import build_cp_context
from fla.ops.kda.chunk_fwd import chunk_kda_fwd
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, tensor_cache
from torch.distributed.distributed_c10d import _resolve_process_group

from .sequence_parallel_bwd import chunk_kda_bwd


@tensor_cache
def _global_offsets(cu_seqlens: torch.Tensor, version: int) -> torch.Tensor:
    # Include the tensor version so an in-place offset change invalidates the CPU copy.
    return cu_seqlens.cpu()


def _context(cu_seqlens: torch.Tensor, group_name: str, local_length: int, kernel_size: int | None = None):
    group = _resolve_process_group(group_name)
    offsets_cpu = cu_seqlens.cpu() if cu_seqlens.is_inference() else _global_offsets(cu_seqlens, cu_seqlens._version)
    # All ranks have the same global offsets and shard length. Validate before the
    # first collective so malformed contexts cannot leave peers waiting in FLA.
    if int(offsets_cpu[-1]) != local_length * group.size():
        raise ValueError("KDA FLA SP requires global packed offsets covering equal contiguous shards.")
    return build_cp_context(cu_seqlens, group, conv1d_kernel_size=kernel_size, cu_seqlens_cpu=offsets_cpu)


@torch.library.custom_op("xtuner_kda::sequence_conv_fwd", mutates_args=())
def sequence_conv_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = weight.shape[-1]
    context = _context(cu_seqlens, group_name, x.shape[1], width)
    rounded_weight = weight.to(x.dtype)
    rounded_bias = None if bias is None else bias.to(x.dtype)
    # Width one needs no halo. FLA's halo helper slices [-0:] in this case.
    initial_state = None
    if width > 1:
        initial_state = CausalConv1dFunctionCP._prepare_initial_state_for_cp(
            x, rounded_weight, context.cu_seqlens, context, context.group
        )
    y, _ = causal_conv1d_fwd(
        x=x,
        weight=rounded_weight,
        bias=rounded_bias,
        residual=None,
        initial_state=initial_state,
        output_final_state=False,
        activation=activation,
        cu_seqlens=context.cu_seqlens,
        cu_seqlens_cpu=context.cu_seqlens_cpu,
    )
    # Only the first local document can continue from the preceding rank. Save a
    # fixed-size boundary state rather than a tensor with data-dependent local N.
    boundary = x.new_zeros((1, x.shape[-1], width)) if initial_state is None else initial_state[:1].clone()
    return y, boundary


@sequence_conv_fwd.register_fake
def _sequence_conv_fwd_fake(x, weight, bias, activation, cu_seqlens, group_name):
    return torch.empty_like(x), x.new_empty((1, x.shape[-1], weight.shape[-1]))


@torch.library.custom_op(
    "xtuner_kda::sequence_conv_bwd",
    mutates_args=(),
    schema="(Tensor x, Tensor dy, Tensor weight, Tensor? bias, Tensor boundary, str? activation, "
    "Tensor cu_seqlens, str group_name) -> (Tensor, Tensor, Tensor?)",
)
def sequence_conv_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    boundary: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    width = weight.shape[-1]
    context = _context(cu_seqlens, group_name, x.shape[1], width)
    initial_state = None
    if width > 1 and not context.is_first_rank:
        initial_state = boundary.new_zeros((context.num_seqs, x.shape[-1], width))
        initial_state[:1] = boundary
    # Match XTuner's local convolution: low-precision forward values, FP32 master
    # parameter gradients when parameters are kept in FP32 by mixed precision.
    dx, dw, db, _, dh0 = causal_conv1d_bwd(
        x=x,
        dy=dy,
        dht=None,
        weight=weight.to(x.dtype).to(weight.dtype),
        bias=None if bias is None else bias.to(x.dtype).to(bias.dtype),
        residual=None,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=context.cu_seqlens,
        cu_seqlens_cpu=context.cu_seqlens_cpu,
    )
    if width > 1:
        CausalConv1dFunctionCP._correct_dx_for_cp(
            dx, dh0, width, context.group, context.is_first_rank, context.pre_num_conv_tokens
        )
    return dx, dw, db


@sequence_conv_bwd.register_fake
def _sequence_conv_bwd_fake(x, dy, weight, bias, boundary, activation, cu_seqlens, group_name):
    return torch.empty_like(x), torch.empty_like(weight), None if bias is None else torch.empty_like(bias)


class SequenceConvFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, weight, bias, activation, cu_seqlens, group_name):
        y, boundary = sequence_conv_fwd(x, weight, bias, activation, cu_seqlens, group_name)
        ctx.save_for_backward(x, weight, bias, boundary, cu_seqlens)
        ctx.activation, ctx.group_name = activation, group_name
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        x, weight, bias, boundary, cu_seqlens = ctx.saved_tensors
        dx, dw, db = sequence_conv_bwd(x, dy, weight, bias, boundary, ctx.activation, cu_seqlens, ctx.group_name)
        return dx, dw, db, None, None, None


def sequence_causal_conv1d(x, weight, bias, activation, cu_seqlens, group_name):
    return SequenceConvFunction.apply(x, weight, bias, activation, cu_seqlens, group_name)


@torch.library.custom_op("xtuner_kda::sequence_chunk_fwd", mutates_args=())
def sequence_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    group_name: str,
    safe_gate: bool,
    transpose_state_layout: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    context = _context(cu_seqlens, group_name, q.shape[1])
    q_l2, q_rstd = l2norm_fwd(q)
    k_l2, k_rstd = l2norm_fwd(k)
    indices = prepare_chunk_indices(context.cu_seqlens, 64, cu_seqlens_cpu=context.cu_seqlens_cpu)
    result = chunk_kda_fwd(
        q=q_l2,
        k=k_l2,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=context.cu_seqlens,
        cu_seqlens_cpu=context.cu_seqlens_cpu,
        chunk_indices=indices,
        safe_gate=safe_gate,
        cp_context=context,
        transpose_state_layout=transpose_state_layout,
    )
    o, _, g_prefix, Aqk, Akk, *_, initial_state = result
    return o, g_prefix, Aqk, Akk, q_l2, q_rstd, k_l2, k_rstd, initial_state


@sequence_chunk_fwd.register_fake
def _sequence_chunk_fwd_fake(q, k, v, g, beta, cu_seqlens, group_name, safe_gate, transpose_state_layout):
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state_shape = (1, heads, value_dim, key_dim) if transpose_state_layout else (1, heads, key_dim, value_dim)
    return (
        torch.empty_like(v),
        torch.empty_like(g, dtype=torch.float32),
        q.new_empty((batch, length, heads, 64)),
        q.new_empty((batch, length, heads, 64)),
        torch.empty_like(q),
        q.new_empty((batch, length, heads), dtype=torch.float32),
        torch.empty_like(k),
        k.new_empty((batch, length, heads), dtype=torch.float32),
        q.new_empty(state_shape, dtype=torch.float32),
    )


@torch.library.custom_op("xtuner_kda::sequence_chunk_bwd", mutates_args=())
def sequence_chunk_bwd(
    q: torch.Tensor,
    q_rstd: torch.Tensor,
    k: torch.Tensor,
    k_rstd: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    boundary: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    group_name: str,
    safe_gate: bool,
    transpose_state_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context = _context(cu_seqlens, group_name, q.shape[1])
    indices = prepare_chunk_indices(context.cu_seqlens, 64, cu_seqlens_cpu=context.cu_seqlens_cpu)
    dq, dk, dv, db, dg = chunk_kda_bwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=q.shape[-1] ** -0.5,
        initial_state=boundary,
        do=do,
        dht=None,
        cu_seqlens=context.cu_seqlens,
        chunk_indices=indices,
        safe_gate=safe_gate,
        cp_context=context,
        transpose_state_layout=transpose_state_layout,
    )[:5]
    # Keep normalization backward inside the opaque op as well: Triton
    # autotuner state changes cannot be traced through an autograd function.
    return l2norm_bwd(q, q_rstd, dq).to(q), l2norm_bwd(k, k_rstd, dk).to(k), dv, db, dg


@sequence_chunk_bwd.register_fake
def _sequence_chunk_bwd_fake(
    q, q_rstd, k, k_rstd, v, g, beta, Aqk, Akk, boundary, do, cu_seqlens, group_name, safe_gate, transpose_state_layout
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(beta), torch.empty_like(g)


class SequenceChunkFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta, cu_seqlens, group_name, safe_gate, transpose_state_layout):
        o, prefix, Aqk, Akk, q_l2, q_rstd, k_l2, k_rstd, boundary = sequence_chunk_fwd(
            q, k, v, g, beta, cu_seqlens, group_name, safe_gate, transpose_state_layout
        )
        ctx.save_for_backward(q_l2, q_rstd, k_l2, k_rstd, v, prefix, beta, Aqk, Akk, boundary, cu_seqlens)
        ctx.group_name, ctx.safe_gate = group_name, safe_gate
        ctx.transpose_state_layout = transpose_state_layout
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, q_rstd, k, k_rstd, v, prefix, beta, Aqk, Akk, boundary, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg = sequence_chunk_bwd(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            prefix,
            beta,
            Aqk,
            Akk,
            boundary,
            do,
            cu_seqlens,
            ctx.group_name,
            ctx.safe_gate,
            ctx.transpose_state_layout,
        )
        return (
            dq,
            dk,
            dv.to(v),
            dg,
            db.to(beta),
            None,
            None,
            None,
            None,
        )


def sequence_chunk_kda(q, k, v, g, beta, cu_seqlens, group_name, safe_gate=False, transpose_state_layout=False):
    return SequenceChunkFunction.apply(q, k, v, g, beta, cu_seqlens, group_name, safe_gate, transpose_state_layout)
