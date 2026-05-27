from __future__ import annotations

# modified from
# https://github.com/fla-org/flash-linear-attention/tree/v0.4.2/fla/ops/gated_delta_rule/chunk.py
# to support torch.compile
import warnings
from typing import TYPE_CHECKING

import torch
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd as origin_chunk_gated_delta_rule_bwd,
)
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd as origin_chunk_gated_delta_rule_fwd,
)
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


@torch.library.custom_op(
    "gated_deltanet::chunk_gated_delta_rule_fwd",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
    "Tensor? initial_state, bool output_final_state, Tensor? cu_seqlens=None, "
    "Tensor? cu_seqlens_cpu=None, Tensor? chunk_indices=None, bool transpose_state_layout=False)"
    " -> (Tensor, Tensor, Tensor, Tensor?, Tensor?, Tensor?)",
)
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if cu_seqlens is not None and chunk_indices is None:
        # 不同 FLA 版本的 prepare_chunk_indices 签名不同；旧版本不接受 cu_seqlens_cpu。
        try:
            chunk_indices = prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
        except TypeError as exc:
            if "cu_seqlens_cpu" not in str(exc):
                raise
            chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    # 不同 FLA 版本的 kernel 签名不同；优先使用新签名，旧环境回退到主干可用的参数集合。
    try:
        result = origin_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=None,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
    except TypeError as exc:
        if not any(key in str(exc) for key in ("cp_context", "chunk_indices", "transpose_state_layout")):
            raise
        result = origin_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    if len(result) == 4:
        g, o, A, final_state = result
    else:
        g, o, A, final_state, initial_state = result
    # initial_state is None in current XTuner implementation,
    # if not None, clone it to ensure the output of this custom operator must not also be an input
    initial_state = initial_state.clone() if initial_state is not None else initial_state
    return g, o, A, final_state, initial_state, chunk_indices


@chunk_gated_delta_rule_fwd.register_fake
def chunk_gated_delta_rule_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    # This fake implementation is only used for shape inference and will not be executed.
    # Returns: (g, o, A, final_state, initial_state, chunk_indices)
    # - g: [B, T, H], float32 cumulative gate
    # - o: [B, T, H, V], same as v
    # - A: [B, T, H, BT] where BT=64 (chunk_size)
    # - final_state: [N, H, K, V] or None
    batch_size, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    chunk_size = 64  # BT

    # g is processed by chunk_local_cumsum but shape remains [B, T, H]
    g_out = torch.empty_like(g, dtype=torch.float32)
    # o has same shape as v: [B, T, H, V]
    o = torch.empty(batch_size, seq_len, num_heads, head_v_dim, device=q.device, dtype=q.dtype)
    # A has shape [B, T, H, chunk_size]
    A = torch.empty(batch_size, seq_len, num_heads, chunk_size, device=q.device, dtype=k.dtype)
    if output_final_state:
        num_states = initial_state.shape[0] if initial_state is not None else batch_size
        if cu_seqlens is not None:
            num_states = cu_seqlens.shape[0] - 1
        final_state_shape = (
            (num_states, num_heads, head_v_dim, head_k_dim)
            if transpose_state_layout
            else (num_states, num_heads, head_k_dim, head_v_dim)
        )
        final_state = torch.empty(final_state_shape, device=q.device, dtype=torch.float32)
    else:
        final_state = None
    initial_state_out = torch.empty_like(initial_state) if initial_state is not None else None
    if chunk_indices is not None:
        chunk_indices_out = torch.empty_like(chunk_indices)
    elif cu_seqlens is not None:
        try:
            num_chunks = torch.library.get_ctx().new_dynamic_size(min=0)
            chunk_indices_out = torch.empty(num_chunks, 2, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        except RuntimeError:
            chunk_indices_out = torch.empty(0, 2, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    else:
        chunk_indices_out = None
    return g_out, o, A, final_state, initial_state_out, chunk_indices_out


@torch.library.custom_op(
    "gated_deltanet::chunk_gated_delta_rule_bwd",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, float scale, "
    "Tensor? initial_state, Tensor do, Tensor? dht, Tensor? cu_seqlens=None, "
    "Tensor? chunk_indices=None, bool transpose_state_layout=False) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)",
)
def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # 不同 FLA 版本的 kernel 签名不同；优先使用新签名，旧环境回退到主干可用的参数集合。
    try:
        dq, dk, dv, db, dg, dh0 = origin_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cp_context=None,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
    except TypeError as exc:
        if not any(key in str(exc) for key in ("cp_context", "chunk_indices", "transpose_state_layout")):
            raise
        dq, dk, dv, db, dg, dh0 = origin_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )
    return dq, dk, dv, db, dg, dh0


@chunk_gated_delta_rule_bwd.register_fake
def chunk_gated_delta_rule_bwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # This fake implementation is only used for shape inference and will not be executed.
    # Returns: (dq, dk, dv, db, dg, dh0)
    # - dq: [B, T, H, K], same as q
    # - dk: [B, T, H, K], same as k
    # - dv: [B, T, H, V], same as v
    # - db: [B, T, H], same as beta
    # - dg: [B, T, H], same as g
    # - dh0: [N, H, K, V], same as initial_state
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    db = torch.empty_like(beta)
    dg = torch.empty_like(g)
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
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
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
    ):
        # XTuner does not use FLA context parallelism. Keep the 0.4.2 apply signature,
        # but pass None into the kernel path so custom ops only carry compile-safe args.
        assert cp_context is None
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, o, A, final_state, initial_state, chunk_indices = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=None,
            transpose_state_layout=transpose_state_layout,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor | None,
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            transpose_state_layout=ctx.transpose_state_layout,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    # XTuner stores packed sequence offsets as int32 in SequenceContext to match the varlen attention path.
    # This wrapper only forwards the tensor to the kernel, so both int32 and int64 packed offsets are valid here.
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
            XTuner stores packed sequence offsets as int32 in `SequenceContext`, so this wrapper also accepts int32.
        cu_seqlens_cpu (torch.Tensor):
            Optional CPU copy of `cu_seqlens` used when preparing chunk indices.
        cp_context (FLACPContext):
            Kept for FLA v0.4.2 signature compatibility. XTuner does not use FLA context parallelism and resets it to
            `None` before launching the compile-friendly custom op.
        transpose_state_layout (bool):
            Whether to use `[N, H, V, K]` state layout instead of `[N, H, K, V]`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Notes:
        Unlike FLA v0.4.2, the internal compile wrapper returns `chunk_indices` from the forward custom op and saves it
        for backward, so the backward custom op does not need to prepare chunk indices again.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if "head_first" in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu
        cp_context = None

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
        transpose_state_layout,
    )
    return o, final_state
