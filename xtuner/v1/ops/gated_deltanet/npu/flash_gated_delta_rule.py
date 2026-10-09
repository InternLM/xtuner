# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import os
from typing import Mapping, Optional

import torch
import torch_npu

from .metadata import get_npu_delta_rule_block_sizes, prepare_npu_metadata
from .triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .triton_core.cumsum import chunk_local_cumsum

# from mindspeed.lite.ops.triton.l2norm import l2norm_bwd, l2norm_fwd
# from mindspeed.lite.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
# from mindspeed.lite.ops.triton.wy_fast import recompute_w_u_fwd
# from mindspeed.lite.ops.triton.solve_tril import solve_tril
# from mindspeed.lite.ops.triton.cumsum import chunk_local_cumsum
# from mindspeed.lite.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from .triton_core.l2norm import l2norm_bwd, l2norm_fwd

# from .triton_core.solve_tril import solve_tril
from .triton_core.solve_tril_fast import solve_tril_npu as solve_tril
from .triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Mapping[str, torch.Tensor]] = None,
    chunk_indices_list: Optional[Mapping[str, list[int]]] = None,
    chunk_size: int = 64,
):
    if chunk_indices is None or chunk_indices_list is None:
        raise ValueError("NPU flash gated delta-rule requires prepared chunk metadata")
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,  # type: ignore[arg-type]
        head_first=False,
    )
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,  # type: ignore[arg-type]
        chunk_indices=chunk_indices[str(chunk_size)],
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,  # type: ignore[arg-type]
        output_dtype=k.dtype,
    )
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()
    w, u = torch_npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        chunk_size=chunk_size,
    )
    # w, u = recompute_w_u_fwd_new(
    #     k=k,
    #     v=v,
    #     beta=beta,
    #     A=A,
    #     g=g,
    #     cu_seqlens=cu_seqlens,
    #     chunk_indices=chunk_indices,
    # )

    # if cu_seqlens is not None:
    #     chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # else:
    #     chunk_indices = None

    h, v_new, final_state = torch_npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state, cu_seqlens, chunk_indices[str(chunk_size)], output_final_state, chunk_size
    )

    # if cu_seqlens is not None:
    #     chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    o = torch_npu.npu_chunk_fwd_o(q, k, v_new, h, g, cu_seqlens, chunk_indices[str(chunk_size)], scale, chunk_size)
    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Mapping[str, torch.Tensor]] = None,
    chunk_indices_list: Optional[Mapping[str, list[int]]] = None,
    chunk_size: int = 64,
):
    if chunk_indices is None or chunk_indices_list is None:
        raise ValueError("NPU flash gated delta-rule requires prepared chunk metadata")
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    w, u = torch_npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        chunk_size=chunk_size,
    )
    # w, u = recompute_w_u_fwd_new(
    #     k=k,
    #     v=v,
    #     beta=beta,
    #     A=A,
    #     g=g,
    #     cu_seqlens=cu_seqlens,
    #     chunk_indices=chunk_indices,
    # )

    # if cu_seqlens is not None:
    #     chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # else:
    #     chunk_indices = None

    do = do.transpose(1, 2).contiguous()

    h, v_new, _ = torch_npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state, cu_seqlens, chunk_indices[str(chunk_size)], False, chunk_size
    )

    # if cu_seqlens is not None:
    #     cu_seqlens1 = cu_seqlens.tolist()
    #     chunk_indices = prepare_chunk_indices1(cu_seqlens1, chunk_size)
    # else:
    #     cu_seqlens1 = None

    dv = torch_npu.npu_chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        scale=scale,
        chunk_size=chunk_size,
    )

    dh, dh0, dv = torch_npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        scale=scale,
        chunk_size=chunk_size,
    )
    dh0 = None

    dq, dk, dw, dg = torch_npu.npu_chunk_bwd_dqkwg(
        q, k, v_new, g, h, do, dh, dv, cu_seqlens_list, chunk_indices_list[str(chunk_size)], scale, chunk_size
    )

    dA = torch_npu.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        chunk_size=chunk_size,
    )

    dk2, dv, db, dg2 = torch_npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        chunk_size=chunk_size,
    )

    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()

    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype} , should be float32")

    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,  # type: ignore[arg-type]
        head_first=False,
    )

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
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[Mapping[str, torch.Tensor]] = None,
        chunk_indices_list: Optional[Mapping[str, list[int]]] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        chunk_indices = ctx.chunk_indices
        cu_seqlens_list = ctx.cu_seqlens_list
        chunk_indices_list = ctx.chunk_indices_list
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
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
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        # if torch.distributed.get_rank()==0:
        #     breakpoint()
        # torch.distributed.barrier()
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, None, None


@torch.compiler.disable
def flash_gated_delta_rule_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Mapping[str, torch.Tensor]] = None,
    chunk_indices_list: Optional[Mapping[str, list[int]]] = None,
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "64")),
    head_first: bool = False,
):
    r"""Run the NPU-native gated delta rule.

    Q/K/V use contiguous head-first layouts. ``g`` and ``beta`` remain
    time-major, and the output is time-major.

    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]`.
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
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

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
    if cu_seqlens is None:
        raise ValueError("NPU flash gated delta-rule requires cu_seqlens")
    if cu_seqlens_list is None or chunk_indices is None or chunk_indices_list is None:
        raise ValueError("NPU flash gated delta-rule requires prepared sequence metadata")
    cu_seqlens = cu_seqlens.to(torch.int64)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("NPU-native q/k/v must have shape [B, H, T, D]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"incompatible NPU-native q/k/v shapes: {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("NPU-native q/k/v must use contiguous head-first storage")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype} , k current type is {k.dtype} ,v current type is {v.dtype} , they should are equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.")
    batch_size, num_heads, seq_len, _ = q.shape
    if g.shape != (batch_size, seq_len, num_heads) or beta.shape != g.shape:
        raise ValueError(
            "g and beta must use [B, T, H] while NPU-native q/k/v use [B, H, T, D]; "
            f"got g={tuple(g.shape)}, beta={tuple(beta.shape)}"
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
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
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state


@torch.compiler.disable
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    cu_seqlens_int64: Optional[torch.Tensor] = None,
    chunk_indices: Optional[Mapping[str, torch.Tensor]] = None,
    chunk_indices_list: Optional[Mapping[str, list[int]]] = None,
):
    """Run NPU gated delta-rule with canonical ``[B, T, H, D]`` Q/K/V.

    Canonical views returned by NPU causal-conv recover their original contiguous head-first storage here. Other
    callers pay a contiguous copy only when their strides cannot represent the native layout directly.
    """
    if cu_seqlens is None:
        raise ValueError("NPU chunk gated delta-rule requires cu_seqlens")
    if cu_seqlens_list is None:
        raise ValueError("NPU chunk gated delta-rule requires cu_seq_lens_q_list")
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    batch_size, seq_len, num_heads, _ = q.shape
    chunk_size = int(os.environ.get("CHUNK_SIZE", "64"))
    block_sizes = get_npu_delta_rule_block_sizes(num_heads, chunk_size)
    required_keys = {str(block_size) for block_size in block_sizes}
    if (
        cu_seqlens_int64 is None
        or chunk_indices is None
        or chunk_indices_list is None
        or not required_keys.issubset(chunk_indices)
        or str(chunk_size) not in chunk_indices_list
    ):
        fallback_metadata = prepare_npu_metadata(
            cu_seqlens=cu_seqlens_list,
            device=q.device,
            total_tokens=batch_size * seq_len,
            block_sizes=block_sizes,
            list_block_sizes={chunk_size},
        )
        cu_seqlens_int64 = fallback_metadata.cu_seqlens_int64
        chunk_indices = fallback_metadata.chunk_indices
        chunk_indices_list = fallback_metadata.chunk_indices_list
    assert cu_seqlens_int64 is not None
    assert chunk_indices is not None
    assert chunk_indices_list is not None

    def to_native(x: torch.Tensor) -> torch.Tensor:
        native = x.transpose(1, 2)
        return native if native.is_contiguous() else native.contiguous()

    return flash_gated_delta_rule_native(
        to_native(q),
        to_native(k),
        to_native(v),
        g=g,
        beta=beta,
        scale=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens_int64,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
