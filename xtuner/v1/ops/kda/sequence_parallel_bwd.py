# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) OpenMMLab. All rights reserved.
"""FLA 0.4.2 KDA backward with corrected sequence-parallel gate addresses.

The merged backward summary must read the current document and head's gate.
This local orchestration keeps that correction scoped to XTuner KDA sequence SP;
the installed FLA module and its other consumers are never patched. All other
FLA backward kernels and collective/merge implementations are reused unchanged.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import expand_h0, merge_fwd_bwd_kernel
from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_wy_dqkg_fused
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp, exp2
from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem("ampere") else [1])
    ],
    key=["H", "K", "V", "BT", "USE_EXP2"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def pre_process_bwd_kernel_merged(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Merged backward kernel that computes both dh (K x V) and dm (K x K) in a single kernel.

    Similar to pre_process_fwd_kernel_merged, this kernel uses a unified grid where:
    - Columns [0, V) are for computing dh (stage 1)
    - Columns [V, V+K) are for computing dm (stage 2)
    """
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles dh (V part) or dm (K part)
    is_dh_part = i_col * BLOCK_SIZE < V

    # Calculate offsets
    q += ((bos * H + i_h) * K).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_k = H * K

    if is_dh_part:
        # ====== Stage 1: Compute dh (K x V) ======
        do += ((bos * H + i_h) * V).to(tl.int64)
        dv += ((bos * H + i_h) * V).to(tl.int64)
        stride_v = H * V
        i_v = i_col

        # Initialize dh accumulators
        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        # Main recurrence for dh (reverse order)
        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                # Note: pre_process_bwd_kernel_stage1 always uses exp for USE_G,
                # regardless of USE_EXP2. This is for consistency with the original design.
                bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
                bg_last_exp = exp(bg_last)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_g_exp = exp(b_g)

            p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))

            # Update dv
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                if USE_EXP2:
                    b_gk_last1 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.0
                    ).to(tl.float32)
                else:
                    b_gk_last1 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.0
                    ).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k2 = 64 + o_k1
                    if USE_EXP2:
                        b_gk_last2 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.0
                        ).to(tl.float32)
                    else:
                        b_gk_last2 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.0
                        ).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k3 = 128 + o_k1
                    if USE_EXP2:
                        b_gk_last3 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.0
                        ).to(tl.float32)
                    else:
                        b_gk_last3 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.0
                        ).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k4 = 192 + o_k1
                    if USE_EXP2:
                        b_gk_last4 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.0
                        ).to(tl.float32)
                    else:
                        b_gk_last4 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.0
                        ).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                # Note: pre_process_bwd_kernel_stage1 always uses exp for USE_G
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, boundary_check=(0, 1))

            # Update dh
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1[:, None])
                else:
                    b_dh1 *= exp(b_gk_last1[:, None])
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2[:, None])
                    else:
                        b_dh2 *= exp(b_gk_last2[:, None])
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3[:, None])
                    else:
                        b_dh3 *= exp(b_gk_last3[:, None])
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4[:, None])
                    else:
                        b_dh4 *= exp(b_gk_last4[:, None])
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        # Store dh results
        p_dh1 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute dm (K x K) ======
        # i_col is for dm part, map to K dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design for backward (FORWARD=False)
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            # Reverse order for backward
            i_t = NT - 1 - _i_t

            # Load k and w with full BK1 rows
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                if USE_EXP2:
                    b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_k = b_k * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + (bos + last_idx) * H * K + i_h * K + row, mask=(row < K), other=0.0).to(
                    tl.float32
                )
                if USE_EXP2:
                    b_gk_last = exp2(b_gk_last)
                else:
                    b_gk_last = exp(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # Compute dm update for backward: m = (diag - w^T @ k) @ m
            # Note: FORWARD=False uses tl.trans(b_w) @ b_k instead of tl.trans(b_k) @ b_w
            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            b_m_i = b_diag - b_kw
            # Keep m chain in fp32 to avoid precision loss from repeated bf16 casting
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        # Store dm result
        p_m = tl.make_block_ptr(dhm + V, (K, K), (V + K, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    rank = dist.get_rank(context.group)

    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    dhm = q.new_zeros(H, K, V + K, dtype=torch.float32)
    if transpose_state_layout:
        dht = q.new_zeros(N, H, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, H, K, V, dtype=torch.float32)

    if not context.is_first_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)
        pre_process_bwd_kernel_merged[grid](
            q=q,
            k=k,
            w=w,
            g=g,
            gk=gk,
            do=do,
            dhm=dhm,
            dv=dv,
            cu_seqlens=cu_seqlens[:2],
            scale=scale,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    if not context.is_last_rank:

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), H)

        merge_fwd_bwd_kernel[grid](
            h=dht[-1],
            ag_hm=ag_dhm,
            pre_or_post_num_ranks=context.post_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            H=H,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            TRANSPOSE_STATE=transpose_state_layout,
        )

    # initial_state is None in the CP mode
    # We only need to compute dht of current rank and pass it to the backward kernel
    return dht, None


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
):
    if disable_recompute is False:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        w, u, qg, kg = recompute_w_u_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if cp_context is not None:
            # Restore the full initial_state tensor from the compressed version.
            # Only the first sequence's state is non-zero as it's the only one that could be cross-rank.
            initial_state = expand_h0(initial_state, context=cp_context)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=transpose_state_layout,
        )
    else:
        w, u, qg, kg, v_new, h = kwargs["w"], kwargs["u"], kwargs["qg"], kwargs["kg"], kwargs["v_new"], kwargs["h"]
        if cp_context is not None:
            # Restore the full initial_state tensor from the compressed version.
            # Only the first sequence's state is non-zero as it's the only one that could be cross-rank.
            initial_state = expand_h0(initial_state, context=cp_context)

    # dAqk = do @ v.T
    # dv = A @ do
    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=qg,
            k=kg,
            w=w,
            do=do,
            dv=dv,
            gk=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            use_exp2=True,
            context=cp_context,
            transpose_state_layout=transpose_state_layout,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )

    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    dA, dbias = None, None
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if use_gate_in_kernel:
        dg, dA, dbias = kda_gate_bwd(g=g_org, A_log=A_log, dt_bias=dt_bias, dyg=dg, lower_bound=lower_bound)

    return dq, dk, dv, db, dg, dh0, dA, dbias
