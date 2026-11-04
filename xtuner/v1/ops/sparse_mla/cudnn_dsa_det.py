# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: MIT
import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90 import (
    _FlashAttentionDSABackwardPostprocessSm90,
    _FlashAttentionDSABackwardPreprocessSm90,
)
from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_major as _get_device_capability,
)
from cudnn.deepseek_sparse_attention.utils.runtime import (
    maybe_contiguous,
    resolve_stream,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

from ._cudnn_dsa_bwd_det_subclass import FlashAttentionDSABackwardSm90Det


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


_DET_TESTED_CUDNN_VERSION = "1.28.0"
_det_runtime_checked = False

# cute.compile results keyed by shape/config, shared by the single-launch and
# multi-launch drivers (module-level so mypy sees plain dicts).
_COMPILE_CACHE_PRE: dict = {}
_COMPILE_CACHE: dict = {}
_COMPILE_CACHE_POST: dict = {}


def _ensure_det_cudnn_runtime() -> None:
    """Fail loudly if the private-API assumptions of the deterministic fork
    break.

    The subclass in ``_cudnn_dsa_bwd_det_subclass.py`` mirrors cudnn 1.28.0's ``scatter_dkv_atomic``; a cudnn upgrade
    that renames or reshapes that method would silently turn the override into dead code and fall back to unordered
    atomics — losing bitwise repeatability while XTUNER_DETERMINISTIC=true claims otherwise. Version/method mismatches
    must raise, never fall back to stock.
    """
    global _det_runtime_checked
    if _det_runtime_checked:
        return
    import cudnn

    version = getattr(cudnn, "__version__", "unknown")
    if version != _DET_TESTED_CUDNN_VERSION:
        # RuntimeError, not assert: `python -O` strips asserts, which would silently
        # void the version pin and lose bitwise repeatability under the det flag.
        raise RuntimeError(
            f"deterministic DSA backward is validated against cudnn {_DET_TESTED_CUDNN_VERSION}, "
            f"found {version}: the scatter_dkv_atomic override mirrors that version's internals. "
            "Re-validate the subclass against the new cudnn (see _cudnn_dsa_bwd_det_subclass.py) "
            "and bump _DET_TESTED_CUDNN_VERSION, or run with XTUNER_DETERMINISTIC=false."
        )
    if not (
        hasattr(FlashAttentionDSABackwardSm90Det, "scatter_dkv_atomic")
        and hasattr(FlashAttentionDSABackwardSm90Det.__mro__[1], "scatter_dkv_atomic")
    ):
        raise RuntimeError("cudnn's FlashAttentionDSABackwardSm90 no longer has scatter_dkv_atomic to override")
    _det_runtime_checked = True


def _check_accum_workspace(n_splits: int, seqlen_k_rounded: int, head_dim_rounded: int, device) -> None:
    """Clear error instead of a raw CUDA OOM when the sliced accumulator cannot
    fit.

    The widened FP32 accumulator costs ``4 * n_splits * seqlen_k_rounded * head_dim_rounded`` bytes — one private slice
    per CTA of a launch (chunk=128 slices at 16K/512-dim is 4 GiB; 512K/576-dim would be 144 GiB).
    """
    workspace_bytes = 4 * n_splits * seqlen_k_rounded * head_dim_rounded
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if workspace_bytes > free_bytes:
        if n_splits == 1:
            hint = "Even a single per-CTA slice does not fit; run with XTUNER_DETERMINISTIC=false."
        else:
            hint = (
                "Reduce `chunk` in flash_attn_bwd_sm90_multilaunch (fewer slices per launch, "
                "proportionally more launches) or run with XTUNER_DETERMINISTIC=false."
            )
        raise RuntimeError(
            f"deterministic DSA backward needs {workspace_bytes / 2**30:.2f} GiB for the per-CTA "
            f"FP32 accumulator ({n_splits} slices x {seqlen_k_rounded} kv x {head_dim_rounded} dim "
            f"x 4 bytes) but only {free_bytes / 2**30:.2f} GiB is free. {hint}"
        )


def flash_attn_bwd_sm90(
    q: torch.Tensor,
    kv: torch.Tensor,  # K=V unified tensor
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    dkv_accum: Optional[torch.Tensor] = None,
    topk_idxs: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    current_stream=None,
    zero_accum: bool = True,
    run_postprocess: bool = True,
    run_preprocess: bool = True,
    only_preprocess: bool = False,
    dpsum: Optional[torch.Tensor] = None,
    lse_log2: Optional[torch.Tensor] = None,
    n_splits_override: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """FlashAttention (DSA) Backward Pass for Hopper (SM90), with K=V.

    Accepts flat (unbatched) tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel.

    Args:
        q: (total_S_q, nheads, headdim) bfloat16
        kv: (total_S_kv, headdim) bfloat16  (K=V, MQA h_kv=1)
        out: (total_S_q, nheads, headdim_v) bfloat16
        dout: (total_S_q, nheads, headdim_v) bfloat16
        lse: (total_S_q, nheads) float32
        softmax_scale: float (default: 1/sqrt(headdim))
        dq: pre-allocated (total_S_q, nheads, headdim), optional
        dkv: pre-allocated (total_S_kv, headdim), optional
        topk_idxs: (total_S_q, topk_max) int32, global indices
        topk_length: (total_S_q,) int32, per-query valid count, optional

    Returns:
        (dq, dkv) — flat layout gradients. Sink support is dropped: this driver
        always runs sink-free (GLM DSA uses attn_sink=-inf, i.e. no sink).
    """
    compute_capability = _get_device_capability()
    assert compute_capability == 9, f"Only SM90, got SM{compute_capability}0"
    _ensure_det_cudnn_runtime()

    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    head_dim_v = 512 if head_dim == 576 else head_dim
    num_head_kv = 1

    # --- wrap flat tensors as batch=1 4D for the CuTe DSL kernel ---
    q4 = q.unsqueeze(0)  # (1, total_S_q, H, D)
    kv4 = kv.unsqueeze(0).unsqueeze(2)  # (1, total_S_kv, 1, D)
    out4 = out.unsqueeze(0)  # (1, total_S_q, H, D_v)
    dout4 = dout.unsqueeze(0)  # (1, total_S_q, H, D_v)
    lse4 = lse.unsqueeze(0)  # (1, total_S_q, H)
    topk4 = topk_idxs.unsqueeze(0) if topk_idxs is not None else None  # (1, total_S_q, TopK)
    tlen4 = topk_length.unsqueeze(0) if topk_length is not None else None  # (1, total_S_q)

    m_block_size = 64
    n_block_size = 64
    KV_stage = 1
    PdS_stage = 1
    SdP_swapAB = False
    dKV_swapAB = False
    dQ_swapAB = False

    q4, kv4, out4, dout4, lse4 = (maybe_contiguous(t) for t in (q4, kv4, out4, dout4, lse4))

    batch_size = 1
    seqlen_q = total_S_q
    seqlen_k = total_S_kv

    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size

    assert q4.dtype in [torch.float16, torch.bfloat16]
    assert q4.dtype == kv4.dtype == out4.dtype == dout4.dtype
    assert lse4.dtype == torch.float32
    assert all(t.is_cuda for t in (q4, kv4, out4, dout4, lse4))
    assert num_head > num_head_kv, "MLA/MQA requires num_head > num_head_kv"
    assert head_dim in [512, 576]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv

    # The subclassed kernel derives each CTA's accumulator slice from blockIdx.x
    # alone, which is unique only when the launch grid is (seqlen_q, 1, 1). The
    # stock scheduler uses grid.y = num_head_kv * qhead_tiles, so more than one
    # query-head tile per CTA (qhead_per_kvhead > tile_m) would make CTAs (x, 0)
    # and (x, 1) silently share slice x and lose bitwise repeatability. Reject
    # such shapes instead of falling back to the nondeterministic stock path.
    if qhead_per_kvhead > m_block_size:
        # RuntimeError, not assert: `python -O` strips asserts — this guard is what
        # keeps the blockIdx.x-only slice offset sound (see the comment above).
        raise RuntimeError(
            f"deterministic DSA backward requires qhead_per_kvhead <= {m_block_size} "
            f"(single query-head tile per CTA), got {qhead_per_kvhead}"
        )

    device = q.device

    # Allocate flat output buffers, then view as 4D for the kernel
    if dq is None:
        dq = torch.empty_like(q)  # (total_S_q, H, D)
    if dkv is None:
        dkv = torch.empty(total_S_kv, head_dim, dtype=kv.dtype, device=device)

    dq4 = dq.unsqueeze(0)  # (1, total_S_q, H, D)
    dkv4 = dkv.unsqueeze(0).unsqueeze(2)  # (1, total_S_kv, 1, D)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if dpsum is None:
        dpsum = torch.empty(
            batch_size,
            seqlen_q_rounded,
            num_head,
            dtype=torch.float32,
            device=device,
        )
    if lse_log2 is None:
        lse_log2 = torch.empty(
            batch_size,
            seqlen_q_rounded,
            num_head,
            dtype=torch.float32,
            device=device,
        )

    # DET fork: per-CTA accum slices via a widened 3D accumulator (stock layout;
    # the subclassed kernel adds `blockIdx.x % n_splits * slice_size` to its scatter
    # target). Slices are summed in fixed order after the main kernel.
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    total_blocks = seqlen_q * num_head_kv * batch_size  # num_head(grid) = n_kv * qhead_tiles = 1
    n_splits = min(sm_count, total_blocks)
    if n_splits_override is not None:
        n_splits = n_splits_override
    if not only_preprocess:
        # Invariant: every CTA of this launch owns a private slice, so the slice
        # offset (blockIdx.x % n_splits) must be unique within the launch. The grid
        # is (seqlen_q, 1, 1) — one CTA per query — so seqlen_q must not exceed
        # n_splits; longer sequences must go through flash_attn_bwd_sm90_multilaunch,
        # which launches at most chunk=n_splits queries at a time.
        if seqlen_q > n_splits:
            # RuntimeError, not assert: `python -O` strips asserts (see above).
            raise RuntimeError(
                f"deterministic DSA backward needs one accumulator slice per CTA: this launch has "
                f"seqlen_q={seqlen_q} CTAs but n_splits={n_splits} slices. "
                "Use flash_attn_bwd_sm90_multilaunch for longer sequences."
            )
    if dkv_accum is None:
        _check_accum_workspace(n_splits, seqlen_k_rounded, head_dim_rounded, device)
        dkv_accum = torch.zeros(
            batch_size,
            num_head_kv,
            n_splits * seqlen_k_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
    elif zero_accum:
        dkv_accum.fill_(0)

    dtype = torch2cute_dtype_map[q4.dtype]
    current_stream = resolve_stream(current_stream)
    arch = 90
    num_threads = 256

    # --- preprocess ---
    compile_key_pre = (
        dtype,
        head_dim_v,
        m_block_size,
        num_threads,
    )
    if compile_key_pre not in _COMPILE_CACHE_PRE:
        o_tensor, do_tensor = (to_cute_tensor(t) for t in (out4, dout4))
        dpsum_tensor, lse_log2_tensor = (to_cute_tensor(t) for t in (dpsum, lse_log2))
        lse_tensor = to_cute_tensor(lse4, assumed_align=4)
        attn_sink_tensor = None
        d_sink_tensor = None
        fa_bwd_pre = _FlashAttentionDSABackwardPreprocessSm90(
            dtype,
            head_dim_v,
            arch,
            m_block_size,
            num_threads=num_threads,
        )
        _COMPILE_CACHE_PRE[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            attn_sink_tensor,
            d_sink_tensor,
            None,
            None,
            None,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if run_preprocess:
        _COMPILE_CACHE_PRE[compile_key_pre](
            out4,
            dout4,
            dpsum,
            lse4,
            lse_log2,
            None,
            None,
            None,
            None,
            None,
            current_stream,
        )
    if only_preprocess:
        return dq, dkv

    # --- main kernel ---
    assert topk4 is not None
    assert topk4.dtype == torch.int32
    have_topk_length = tlen4 is not None
    if tlen4 is not None:
        assert tlen4.dtype == torch.int32
    else:
        tlen4 = torch.empty(1, dtype=torch.int32, device=device)
    max_topk = topk4.shape[-1]

    num_threads = 256
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        # The kernel specializes its query-head tile and masks unused MMA rows.
        # Keep this explicit even though tensor shapes also differ by head count.
        qhead_per_kvhead,
        m_block_size,
        n_block_size,
        num_threads,
        KV_stage,
        PdS_stage,
        SdP_swapAB,
        dKV_swapAB,
        dQ_swapAB,
        have_topk_length,
        max_topk,
        n_splits,
        seqlen_k_rounded * head_dim_rounded,
    )
    if compile_key not in _COMPILE_CACHE:
        q_tensor = to_cute_tensor(q4)
        kv_tensor = to_cute_tensor(kv4)
        do_tensor = to_cute_tensor(dout4)
        dpsum_tensor, lse_log2_tensor = (to_cute_tensor(t) for t in (dpsum, lse_log2))
        dq_tensor = to_cute_tensor(dq4)
        dkv_accum_tensor = to_cute_tensor(dkv_accum)
        topk_idxs_tensor = to_cute_tensor(topk4)
        topk_length_tensor = to_cute_tensor(tlen4)

        fa_bwd_obj = FlashAttentionDSABackwardSm90Det(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            tile_m=m_block_size,
            tile_n=n_block_size,
            KV_stage=KV_stage,
            PdS_stage=PdS_stage,
            SdP_swapAB=SdP_swapAB,
            dKV_swapAB=dKV_swapAB,
            dQ_swapAB=dQ_swapAB,
            num_threads=num_threads,
            have_topk_length=have_topk_length,
            max_topk=max_topk,
        )
        # Baked in at trace time: slice geometry for the subclassed scatter override.
        fa_bwd_obj._det_slice_size = seqlen_k_rounded * head_dim_rounded
        fa_bwd_obj._det_n_splits = n_splits
        _COMPILE_CACHE[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            kv_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_tensor,
            dkv_accum_tensor,
            topk_idxs_tensor,
            topk_length_tensor,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )
    with torch.cuda.nvtx.range("flash_attn_bwd_sm90_kernel"):
        _COMPILE_CACHE[compile_key](
            q4,
            kv4,
            dout4,
            lse_log2,
            dpsum,
            dq4,
            dkv_accum,
            topk4,
            tlen4,
            softmax_scale,
            current_stream,
        )

    # --- DET fork: ordered reduction over per-CTA slices (fixed order, no
    # atomics) — done in fake-col space so the GMMA interleave unpack below
    # runs exactly once, unchanged ---
    if not run_postprocess:
        return dq, dkv
    dkv_accum_reduced = dkv_accum.view(batch_size, num_head_kv, n_splits, -1).sum(dim=2)  # fixed order

    # --- postprocess: fake-col f32 dKVAccum -> real-col bf16 dKV ---
    hdim_chunk = 64 if head_dim == 576 else min(128, head_dim)
    N_hdim_chunks = head_dim // hdim_chunk
    num_threads_post = hdim_chunk

    compile_key_post = (
        dtype,
        hdim_chunk,
        n_block_size,
        head_dim,
        num_threads_post,
        N_hdim_chunks,
    )
    if compile_key_post not in _COMPILE_CACHE_POST:
        dkv_accum_tensor = to_cute_tensor(dkv_accum_reduced)
        dkv_tensor = to_cute_tensor(dkv4)

        postprocess = _FlashAttentionDSABackwardPostprocessSm90(
            dtype,
            hdim_chunk=hdim_chunk,
            tile_n=n_block_size,
            head_dim=head_dim,
            num_threads=num_threads_post,
            N_hdim_chunks=N_hdim_chunks,
        )
        _COMPILE_CACHE_POST[compile_key_post] = cute.compile(
            postprocess,
            dkv_accum_tensor,
            dkv_tensor,
            seqlen_k,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _COMPILE_CACHE_POST[compile_key_post](
        dkv_accum_reduced,
        dkv4,
        seqlen_k,
        current_stream,
    )

    # dq / dkv are already the flat tensors (unsqueeze was a view)
    return dq, dkv


def flash_attn_bwd_sm90_multilaunch(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    chunk: int = 128,
    current_stream=None,
) -> Tuple[torch.Tensor, ...]:
    """DET fork, multi-launch driver: run the (unmodified-scheduler) bwd kernel
    once per `chunk` queries. Every launch has grid=(chunk,1,1) so CTA i owns
    accum slice i; slices persist across launches and launches execute in a
    fixed stream order, so the per-address accumulation order is deterministic.
    A single ordered reduction + postprocess runs at the end.

    Memory contract: the accumulator workspace is `4 * min(chunk, S_q) *
    round_up(S_k, 64) * round_up(D, 32)` bytes (one full-KV FP32 slice per query of
    a launch) plus one reduced accumulator of `4 * round_up(S_k, 64) *
    round_up(D, 32)` bytes. That is 4 GiB at the GLM-5.3 CI shape (S_k=16K,
    D=512) but grows linearly with S_k (144 GiB at S_k=512K, D=576 with
    chunk=128). The driver therefore halves the per-launch query count (and the
    slice count) until the workspace fits 80% of the free memory — longer
    sequences run as more, smaller launches instead of erroring. Only when a
    single slice still does not fit does `_check_accum_workspace` raise a clear
    error instead of a raw CUDA OOM.
    """
    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    device = q.device
    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32
    seqlen_k_rounded = (total_S_kv + 63) // 64 * 64

    # slices are private per CTA of a launch: need one slice per query of a
    # launch (may exceed SM count when a launch spans multiple waves; CTAs of
    # different waves never run concurrently... they may, but slice ownership is
    # by blockIdx which is unique within the launch, so privacy holds regardless
    # of scheduling).
    n_splits = min(chunk, total_S_q)
    # Auto-shrink the launch granularity until the FP32 slice workspace fits a
    # budget of 80% free memory: chunk=128 at 512K/576-dim would need 144 GiB,
    # which op-level users cannot remediate by hand. Halving n_splits also halves
    # the per-launch query count below (launch count grows proportionally), so
    # slice privacy — one slice per CTA of a launch — is preserved by
    # construction. `_check_accum_workspace` remains as the final guard for the
    # n_splits==1 case.
    free_bytes, _ = torch.cuda.mem_get_info(device)
    budget = int(free_bytes * 0.8)
    while n_splits > 1 and 4 * n_splits * seqlen_k_rounded * head_dim_rounded > budget:
        n_splits //= 2
    _check_accum_workspace(n_splits, seqlen_k_rounded, head_dim_rounded, device)
    dkv_accum = torch.zeros(
        1,
        1,
        n_splits * seqlen_k_rounded * head_dim_rounded,
        dtype=torch.float32,
        device=device,
    )
    dq = torch.empty_like(q)
    dkv = torch.empty(total_S_kv, head_dim, dtype=kv.dtype, device=device)

    # preprocess (dPsum / LSElog2) once over the full query range — it is
    # deterministic and independent of the dKV accumulation.
    seqlen_q_rounded = (total_S_q + 63) // 64 * 64
    dpsum_full = torch.empty(1, seqlen_q_rounded, num_head, dtype=torch.float32, device=device)
    lse_log2_full = torch.empty(1, seqlen_q_rounded, num_head, dtype=torch.float32, device=device)
    flash_attn_bwd_sm90(
        q,
        kv,
        out,
        dout,
        lse,
        softmax_scale=softmax_scale,
        dq=dq,
        dkv=dkv,
        dkv_accum=dkv_accum,
        topk_idxs=topk_idxs,
        topk_length=topk_length,
        current_stream=current_stream,
        zero_accum=False,
        run_postprocess=False,
        only_preprocess=True,
        dpsum=dpsum_full,
        lse_log2=lse_log2_full,
        n_splits_override=n_splits,
    )

    n_chunks = (total_S_q + n_splits - 1) // n_splits
    for c in range(n_chunks):
        s, e = c * n_splits, min((c + 1) * n_splits, total_S_q)
        flash_attn_bwd_sm90(
            q[s:e],
            kv,
            out[s:e],
            dout[s:e],
            lse[s:e],
            softmax_scale=softmax_scale,
            dq=dq[s:e],
            dkv=dkv,  # unused until the final postprocess
            dkv_accum=dkv_accum,
            topk_idxs=topk_idxs[s:e],
            topk_length=topk_length[s:e] if topk_length is not None else None,
            current_stream=current_stream,
            zero_accum=False,  # dkv_accum was created zeroed; slices only accumulate
            run_postprocess=False,
            run_preprocess=False,
            dpsum=dpsum_full[:, s:e],
            lse_log2=lse_log2_full[:, s:e],
            n_splits_override=n_splits,
        )

    # --- ordered reduction over slices + postprocess (once, full kv) ---
    current_stream = resolve_stream(current_stream)
    dtype = torch2cute_dtype_map[q.dtype]
    dkv_accum_reduced = dkv_accum.view(1, 1, n_splits, -1).sum(dim=2)  # fixed slice order

    hdim_chunk = 64 if head_dim == 576 else min(128, head_dim)
    N_hdim_chunks = head_dim // hdim_chunk
    num_threads_post = hdim_chunk
    n_block_size = 64
    dkv4 = dkv.unsqueeze(0).unsqueeze(2)

    compile_key_post = (dtype, hdim_chunk, n_block_size, head_dim, num_threads_post, N_hdim_chunks)
    if compile_key_post not in _COMPILE_CACHE_POST:
        dkv_accum_tensor = to_cute_tensor(dkv_accum_reduced)
        dkv_tensor = to_cute_tensor(dkv4)
        postprocess = _FlashAttentionDSABackwardPostprocessSm90(
            dtype,
            hdim_chunk=hdim_chunk,
            tile_n=n_block_size,
            head_dim=head_dim,
            num_threads=num_threads_post,
            N_hdim_chunks=N_hdim_chunks,
        )
        _COMPILE_CACHE_POST[compile_key_post] = cute.compile(
            postprocess,
            dkv_accum_tensor,
            dkv_tensor,
            total_S_kv,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _COMPILE_CACHE_POST[compile_key_post](
        dkv_accum_reduced,
        dkv4,
        total_S_kv,
        current_stream,
    )
    return dq, dkv
