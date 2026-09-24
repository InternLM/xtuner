# modified from
# https://github.com/fla-org/flash-linear-attention/tree/v0.4.2/fla/ops/kda/chunk.py
# to support torch.compile
"""Compile-friendly ``chunk_kda``.

FLA's public ``fla.ops.kda.chunk_kda`` is decorated ``@torch.compiler.disable``, and its forward
derives the chunk table with ``prepare_chunk_indices``, which calls ``.tolist()`` on
``cu_seqlens`` -- dynamo traces that as ``aten._local_scalar_dense`` and inductor refuses to lower
it. Calling it from a compiled region therefore forces a graph break at every KDA layer, and
GLM-5.3-Flash is KDA-dominated (34 of 45 layers), which measured ~2.4x slower per step than eager.

This module takes the route ``xtuner/v1/ops/gated_deltanet`` already took for GatedDeltaNet: call
FLA's ``chunk_kda_fwd``/``chunk_kda_bwd`` behind ``torch.library.custom_op``. Dynamo treats a
custom op as an opaque node and traces straight through it, so the chunk-table preparation (and
its host sync) stays outside the graph without breaking it.

Only the call shape GLM-5.3-Flash uses is supported -- no ``initial_state``/``output_final_state``
(the gate is computed outside the kernel by ``fused_kda_gate``, so ``use_gate_in_kernel`` is never
set) and no FLA context parallelism. Anything else should keep using FLA's own entry point.
"""

import torch
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.kda.chunk_bwd import chunk_kda_bwd as _fla_chunk_kda_bwd
from fla.ops.kda.chunk_fwd import chunk_kda_fwd as _fla_chunk_kda_fwd
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


LIBRARY_NAME = "xtuner_kda"
# FLA's KDA kernels are specialized on this chunk width; it is not a tunable.
_CHUNK_SIZE = 64


def _chunk_indices(cu_seqlens: torch.Tensor | None) -> torch.Tensor | None:
    # Inside a custom op, so the `.tolist()` host sync this performs is invisible to dynamo.
    # Recomputed in backward rather than carried across the op boundary: its length depends on the
    # packed document lengths, and returning a data-dependent shape would reintroduce exactly the
    # dynamic-shape problem this module exists to avoid.
    return prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE) if cu_seqlens is not None else None


@torch.library.custom_op(
    f"{LIBRARY_NAME}::chunk_kda_fwd",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, Tensor? cu_seqlens, "
    "bool safe_gate, float? lower_bound, bool transpose_state_layout) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    transpose_state_layout: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # L2-norming q/k here rather than through the kernel's own flag mirrors FLA's
    # `ChunkKDAFunction`, which does it outside the kernel and keeps `rstd` for the backward.
    q_l2, q_rstd = l2norm_fwd(q)
    k_l2, k_rstd = l2norm_fwd(k)
    o, _, g_cumsum, Aqk, Akk = _fla_chunk_kda_fwd(
        q=q_l2,
        k=k_l2,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_indices(cu_seqlens),
        chunk_size=_CHUNK_SIZE,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        transpose_state_layout=transpose_state_layout,
    )[:5]
    # `disable_recompute=False` (the default) frees w/u/qg/kg/v_new/h inside the forward and the
    # backward recomputes them, so only these five tensors have to cross the op boundary.
    return o, g_cumsum, Aqk, Akk, q_l2, q_rstd, k_l2, k_rstd


@chunk_kda_fwd.register_fake
def _chunk_kda_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    transpose_state_layout: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    batch, seq_len, num_heads, _ = q.shape
    rstd_shape = (batch, seq_len, num_heads)
    return (
        torch.empty_like(v),
        torch.empty_like(g),
        q.new_empty((batch, seq_len, num_heads, _CHUNK_SIZE)),
        q.new_empty((batch, seq_len, num_heads, _CHUNK_SIZE)),
        torch.empty_like(q),
        q.new_empty(rstd_shape, dtype=torch.float32),
        torch.empty_like(k),
        k.new_empty(rstd_shape, dtype=torch.float32),
    )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::chunk_kda_bwd",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor Aqk, Tensor Akk, Tensor do, "
    "float scale, Tensor? cu_seqlens, bool safe_gate, float? lower_bound, bool transpose_state_layout) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    transpose_state_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, db, dg = _fla_chunk_kda_bwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=None,
        do=do,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_indices(cu_seqlens),
        chunk_size=_CHUNK_SIZE,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        transpose_state_layout=transpose_state_layout,
    )[:5]
    return dq, dk, dv, db, dg


@chunk_kda_bwd.register_fake
def _chunk_kda_bwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    transpose_state_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(beta),
        torch.empty_like(g),
    )


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        cu_seqlens: torch.Tensor | None,
        safe_gate: bool,
        lower_bound: float | None,
        transpose_state_layout: bool,
    ):
        o, g_cumsum, Aqk, Akk, q_l2, q_rstd, k_l2, k_rstd = torch.ops.xtuner_kda.chunk_kda_fwd(
            q, k, v, g, beta, scale, cu_seqlens, safe_gate, lower_bound, transpose_state_layout
        )
        # Only the L2-normed q/k are kept, exactly as FLA's `ChunkKDAFunction` does: it rebinds
        # `q`/`k` to the normed tensors before saving, and its backward feeds those same normed
        # tensors to both the kernel and `l2norm_bwd`.
        ctx.save_for_backward(q_l2, q_rstd, k_l2, k_rstd, v, g_cumsum, beta, Aqk, Akk, cu_seqlens)
        ctx.scale = scale
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.transpose_state_layout = transpose_state_layout
        return o.type_as(q), None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q_l2, q_rstd, k_l2, k_rstd, v, g_cumsum, beta, Aqk, Akk, cu_seqlens = ctx.saved_tensors
        # The kernel consumed the L2-normed q/k, so its gradients are w.r.t. those; `l2norm_bwd`
        # maps them back onto the original projections.
        dq, dk, dv, db, dg = torch.ops.xtuner_kda.chunk_kda_bwd(
            q_l2,
            k_l2,
            v,
            g_cumsum,
            beta,
            Aqk,
            Akk,
            do,
            ctx.scale,
            cu_seqlens,
            ctx.safe_gate,
            ctx.lower_bound,
            ctx.transpose_state_layout,
        )
        dq = l2norm_bwd(q_l2, q_rstd, dq)
        dk = l2norm_bwd(k_l2, k_rstd, dk)
        return dq.to(q_l2), dk.to(k_l2), dv.to(v), dg, db.to(beta), None, None, None, None, None


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    transpose_state_layout: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Chunked Kimi Delta Attention, traceable by ``torch.compile``.

    Numerically identical to ``fla.ops.kda.chunk_kda`` for the supported call shape: it runs the
    same ``chunk_kda_fwd``/``chunk_kda_bwd`` kernels with the same arguments, only behind custom
    ops so a compiled caller does not graph-break.

    Args:
        q (torch.Tensor): Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor): Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor): Values of shape ``[B, T, H, V]``.
        g (torch.Tensor): Forget gate in log space, ``[B, T, H, K]``, already built by
            ``fused_kda_gate``.
        beta (torch.Tensor): Betas of shape ``[B, T, H]``.
        scale (float | None): Attention scale; defaults to ``K ** -0.5``.
        cu_seqlens (torch.Tensor | None): Packed-sequence offsets, as in the varlen attention API.
        safe_gate (bool): Clamp the gate against ``lower_bound`` inside the kernel.
        lower_bound (float | None): The gate lower bound when ``safe_gate`` is set.
        transpose_state_layout (bool): Use the ``[N, H, V, K]`` state layout.
        use_qk_l2norm_in_kernel (bool): Must stay ``True``; GLM-5.3-Flash always L2-norms q/k, and
            the backward depends on the ``rstd`` that path produces.

    Returns:
        tuple[torch.Tensor, None]: Outputs ``[B, T, H, V]``, and ``None`` for the final state,
        which this entry point does not produce.
    """
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError(
            "xtuner's compile-friendly chunk_kda always L2-norms q/k; use fla.ops.kda.chunk_kda "
            "directly if a caller needs it disabled."
        )
    for unsupported in ("initial_state", "output_final_state", "cp_context", "A_log", "dt_bias"):
        if kwargs.get(unsupported):
            raise NotImplementedError(
                f"xtuner's compile-friendly chunk_kda does not support {unsupported!r}; use "
                "fla.ops.kda.chunk_kda directly."
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ChunkKDAFunction.apply(q, k, v, g, beta, scale, cu_seqlens, safe_gate, lower_bound, transpose_state_layout)
