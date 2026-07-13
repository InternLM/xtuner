# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# FlashMLA-backed forward for DSA's per-sample sparse-attention call.
#
# Phase 1 of the cudnn / FlashMLA integration (see [[project-deepseek-v4-design]]):
# the forward pass invokes DeepSeek-AI's ``flash_mla.flash_mla_sparse_fwd`` C++
# kernel, while the backward pass re-runs the native ``sparse_attn`` reference
# under an ``autograd.Function`` so gradients are bit-identical to the
# ``backend="native"`` path. Subsequent phases swap the backward branch to
# cudnn-frontend's ``SparseAttentionBackward`` + score-recompute pair.
#
# Why a recompute-style backward: FlashMLA exposes only the forward kernel
# (returning ``(out, max_logits, lse)``); cudnn's DSA module is also forward-
# free (release notes name only ``SparseAttentionBackward``). Until we wire the
# cudnn backward, autograd needs *some* way to get ``dq / dkv / d_sink``, and
# the simplest correctness-preserving path is re-evaluating the reference
# attention under ``torch.enable_grad`` and pulling gradients via
# ``torch.autograd.grad``. The cost is one extra forward per backward step;
# real perf shows up once Phase 2 replaces that with the cudnn kernel.
# ============================================================================

from __future__ import annotations

from typing import cast

import torch

from .sparse_attn import sparse_attn as _native_sparse_attn


_FLASH_MLA_IMPORT_ERR: ImportError | None
try:
    import flash_mla as _flash_mla  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover - optional dep
    _flash_mla = None  # type: ignore[assignment]
    _FLASH_MLA_IMPORT_ERR = e
else:
    _FLASH_MLA_IMPORT_ERR = None


_CUDNN_DSA_IMPORT_ERR: ImportError | None
try:
    from cudnn.deepseek_sparse_attention import (  # type: ignore[import-not-found]
        sparse_attention_backward_wrapper as _cudnn_sparse_attention_backward,
    )
except ImportError as e:  # pragma: no cover - optional dep
    _cudnn_sparse_attention_backward = None  # type: ignore[assignment]
    _CUDNN_DSA_IMPORT_ERR = e
else:
    _CUDNN_DSA_IMPORT_ERR = None


_CUDNN_PATCHES_APPLIED = False


# FlashMLA's SM90 sparse prefill kernel asserts ``params.topk % (2 * B_TOPK) == 0``
# at ``phase1.cuh:577``, where ``B_TOPK = 64`` is hard-coded in
# ``csrc/sm90/prefill/sparse/config.h``. The effective requirement is therefore
# ``topk % 128 == 0``. V4's per-layer topk is:
#   * compress_ratio==0  → window(=sliding_window=128)            = 128 ✓
#   * compress_ratio==4  → window + index_topk(512)               = 640 ✓ (5 × 128)
#   * compress_ratio==128→ window + index_topk(512)               = 640 ✓
# So *all* released V4 layer configs pass the assertion. The fallback below
# only kicks in for non-standard configs (e.g. small smoke runs that tweak
# sliding_window or index_topk to values that violate the multiple-of-128 rule).
_FLASH_MLA_TOPK_ALIGN = 128


def _flash_mla_topk_ok(topk: int) -> bool:
    return topk > 0 and (topk % _FLASH_MLA_TOPK_ALIGN) == 0


def _ensure_cudnn_patches_applied() -> None:
    """One-shot runtime workaround for cudnn 1.24.0 ↔ cu12 cutlass-dsl 4.5.0 ABI skew.

    ``cudnn/.../utils/sm90/primitives.py:atomic_add_fp32`` calls ``nvvm.atomicrmw``
    without the leading ``res`` positional that cu12 cutlass-dsl 4.5.0 added
    (the cu13 build of cutlass-dsl 4.5.0 — which cudnn pins via the ``[cu13]``
    extra — does not require ``res``). On a CUDA 12.x driver we can't use the
    cu13 cutlass-dsl, so we monkey-patch the call site with the ``res=T.f32()``
    return-type tag that the cu12 binding expects.

    Idempotent (gated by a module-level flag). Lives in xtuner's own code per the
    project rule against editing site-packages files. Drop this whole helper —
    and the ``_ensure_cudnn_patches_applied()`` call in ``_CudnnSparseAttnFn`` —
    once cudnn-frontend ships a cu12 build (the call-site bug also exists when
    paired with cu13 4.5.1+, but the cu13 path is moot for our R570 driver).
    """
    global _CUDNN_PATCHES_APPLIED
    if _CUDNN_PATCHES_APPLIED:
        return

    from cudnn.deepseek_sparse_attention.utils.sm90 import primitives as _cudnn_primitives  # type: ignore[import-not-found]
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import (  # type: ignore[import-not-found]
        dsa_bwd_sm90 as _cudnn_dsa_bwd_sm90,
    )
    from cutlass import Float32  # type: ignore[import-not-found]
    from cutlass._mlir.dialects import nvvm  # type: ignore[import-not-found]
    from cutlass.cutlass_dsl import T, dsl_user_op  # type: ignore[import-not-found]

    @dsl_user_op
    def _patched_atomic_add_fp32(a, gmem_ptr, *, loc=None, ip=None):  # type: ignore[no-untyped-def]
        nvvm.atomicrmw(
            res=T.f32(),
            op=nvvm.AtomicOpKind.FADD,
            ptr=gmem_ptr.llvm_ptr,
            a=Float32(a).ir_value(),
        )

    # Patch both the source (``primitives.atomic_add_fp32``) and the local
    # reference in ``dsa_bwd_sm90`` — the latter holds a separate binding
    # created by ``from .primitives import atomic_add_fp32`` so patching only
    # the source module would not propagate.
    _cudnn_primitives.atomic_add_fp32 = _patched_atomic_add_fp32
    _cudnn_dsa_bwd_sm90.atomic_add_fp32 = _patched_atomic_add_fp32

    _CUDNN_PATCHES_APPLIED = True


def _flash_mla_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure forward over FlashMLA's sparse kernel.

    Args/return shapes use the XTuner convention (``[1, S, H, D]`` / ``[1, T, D]``);
    layout adapters into FlashMLA's plain ``T,H,D`` contract are localised here.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            * ``out``: ``[1, S, H, head_dim]`` in the dtype of ``q``.
            * ``lse``:  ``[S, H]`` fp32, FlashMLA's KV-only log-sum-exp (excludes
              the attention sink) — required as a backward input by both Phase-1
              recompute (it ignores ``lse``) and Phase-2 cudnn bwd (it consumes
              ``lse`` directly).
    """
    if _flash_mla is None:
        raise ImportError(
            "flash_mla is required for backend in {'flash_mla', 'cudnn'}; install from "
            "https://github.com/deepseek-ai/FlashMLA. Original import error: "
            f"{_FLASH_MLA_IMPORT_ERR}"
        )
    # Layout adapters: XTuner ships everything with an explicit batch dim
    # ([1, S, ...]). FlashMLA's contract is plain T,H,D with an explicit
    # h_kv axis on KV / indices, so squeeze the batch dim and add the h_kv=1
    # axis. ``.contiguous()`` is mandatory — FlashMLA's CUDA kernel asserts
    # contiguity on q / kv / indices.
    q_thd = q.squeeze(0).contiguous()  # [S, H, D]
    kv_thd = kv.squeeze(0).unsqueeze(1).contiguous()  # [T, 1, D]
    idx_thd = topk_idxs.squeeze(0).unsqueeze(1).contiguous().int()  # [S, 1, topk]
    sink_f32 = attn_sink.to(torch.float32).contiguous()  # [H]

    out, _max_logits, lse = _flash_mla.flash_mla_sparse_fwd(
        q_thd,
        kv_thd,
        idx_thd,
        float(softmax_scale),
        d_v=q.size(-1),  # V4 head_dim_v == head_dim == d_qk
        attn_sink=sink_f32,
        topk_length=None,
    )
    # Reattach batch dim so caller sees the `[1, S, H, head_dim]` it would
    # get from native sparse_attn.
    return out.unsqueeze(0), lse


def _check_xtuner_shapes(q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor) -> None:
    """Preflight the XTuner ``[1, S, ...]`` packed-varlen contract.

    Kept in a helper so both Phase-1 (`_FlashMLASparseAttnFn`) and Phase-2
    (`_CudnnSparseAttnFn`) raise with the same XTuner-side error message
    instead of a kernel-side generic shape error from FlashMLA / cudnn.
    """
    if q.dim() != 4 or q.size(0) != 1:
        raise ValueError(
            "q must be packed varlen shaped [1, total_tokens, num_heads, head_dim]; "
            f"got {tuple(q.shape)}"
        )
    if kv.dim() != 3 or kv.size(0) != 1:
        raise ValueError(f"kv must be shaped [1, T_total, head_dim]; got {tuple(kv.shape)}")
    if topk_idxs.dim() != 3 or topk_idxs.size(0) != 1 or topk_idxs.size(1) != q.size(1):
        raise ValueError(
            "topk_idxs must be shaped [1, total_tokens, k] matching q's token axis; "
            f"got {tuple(topk_idxs.shape)} vs q {tuple(q.shape)}"
        )


class _FlashMLASparseAttnFn(torch.autograd.Function):
    """Phase 1: FlashMLA forward + native-recompute backward.

    Trades backward speed for correctness — gradients are bit-identical to
    ``backend="native"`` since the backward path literally re-runs the same
    PyTorch reference. Forward gains the FlashMLA kernel speedup.

    For layers whose ``topk`` does not satisfy FlashMLA's alignment requirement
    (typically the V4 ``compress_ratio == 0`` layers with topk=128), this Fn
    transparently routes the *entire* call to native sparse_attn — so a single
    config-level ``backend="flash_mla"`` setting can cover the whole 43-layer
    stack without the user having to know which layers are compress-enabled.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        _check_xtuner_shapes(q, kv, topk_idxs)
        # Fallback for unaligned topk is handled by the public
        # ``flash_mla_sparse_attn`` wrapper before reaching ``.apply``.
        out_1shd, _lse = _flash_mla_forward(q, kv, attn_sink, topk_idxs, softmax_scale)
        # Save the *original* (untransformed) tensors so the autograd backward
        # below feeds the native sparse_attn the exact same args its forward
        # signature expects.
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, cu_seq_lens)
        ctx.softmax_scale = float(softmax_scale)
        return out_1shd

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, dout: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None]:
        q, kv, attn_sink, topk_idxs, cu_seq_lens = ctx.saved_tensors
        softmax_scale: float = ctx.softmax_scale
        needs_q, needs_kv, needs_sink, *_ = ctx.needs_input_grad

        # No-op if no upstream gradient flow is required. Avoids pointlessly
        # recomputing the (expensive) reference forward when none of q/kv/sink
        # are tracked (e.g., frozen-attention probing).
        if not (needs_q or needs_kv or needs_sink):
            return None, None, None, None, None, None

        with torch.enable_grad():
            q_l = q.detach().requires_grad_(needs_q)
            kv_l = kv.detach().requires_grad_(needs_kv)
            sink_l = attn_sink.detach().requires_grad_(needs_sink)
            out_ref = _native_sparse_attn(q_l, kv_l, sink_l, topk_idxs, softmax_scale, cu_seq_lens)
            leaves = [t for t, n in [(q_l, needs_q), (kv_l, needs_kv), (sink_l, needs_sink)] if n]
            grads_tuple = torch.autograd.grad(
                outputs=out_ref,
                inputs=leaves,
                grad_outputs=dout.to(out_ref.dtype),
                allow_unused=False,
            )

        dq = dkv = dsink = None
        g_iter = iter(cast(tuple, grads_tuple))
        if needs_q:
            dq = next(g_iter)
        if needs_kv:
            dkv = next(g_iter)
        if needs_sink:
            dsink = next(g_iter)
        return dq, dkv, dsink, None, None, None


class _CudnnSparseAttnFn(torch.autograd.Function):
    """Phase 2: FlashMLA forward + cudnn-frontend ``SparseAttentionBackward``.

    Forward calls :func:`_flash_mla_forward` (same as Phase 1) but additionally
    saves the FlashMLA KV-only ``lse`` and the produced ``out`` because the
    cudnn backward kernel consumes both. Backward calls
    ``cudnn.deepseek_sparse_attention.sparse_attention_backward_wrapper`` after
    reshaping to its 2-D-KV / 2-D-topk_idxs contract (vs FlashMLA's 3-D one).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        if _cudnn_sparse_attention_backward is None:
            raise ImportError(
                "cudnn-frontend's deepseek_sparse_attention module is required for "
                "backend='cudnn'; install with `pip install nvidia-cudnn-frontend>=1.24.0`. "
                f"Original import error: {_CUDNN_DSA_IMPORT_ERR}"
            )
        _check_xtuner_shapes(q, kv, topk_idxs)
        # Fallback for unaligned topk handled by ``cudnn_sparse_attn`` before
        # reaching ``.apply``.
        # Lazy application — only when cudnn DSA path actually runs, so the
        # cu12/cu13 monkey-patch doesn't fire for ``backend in {native,
        # flash_mla}`` users who don't need cudnn at all.
        _ensure_cudnn_patches_applied()
        out_1shd, lse = _flash_mla_forward(q, kv, attn_sink, topk_idxs, softmax_scale)
        # Save tensors the cudnn backward kernel needs. We deliberately save the
        # *XTuner-layout* (`[1, S, ...]`) tensors here — backward will reshape
        # them to cudnn's 2-D-KV contract at the call site so the layout adapter
        # lives in one place.
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, out_1shd, lse)
        ctx.softmax_scale = float(softmax_scale)
        return out_1shd

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, dout: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None]:
        q, kv, attn_sink, topk_idxs, out, lse = ctx.saved_tensors
        softmax_scale: float = ctx.softmax_scale
        needs_q, needs_kv, needs_sink, *_ = ctx.needs_input_grad

        if not (needs_q or needs_kv or needs_sink):
            return None, None, None, None, None, None

        # cudnn backward contract (see SparseAttentionBackward.api.py):
        #   q          : (S_q, H, D)        bf16/fp16, 3-D (no batch dim)
        #   kv         : (S_kv, D)          bf16/fp16, **2-D** (no h_kv axis)
        #   out / dout : (S_q, H, D_v)      bf16/fp16
        #   lse        : (S_q, H)           fp32
        #   attn_sink  : (H,)               fp32
        #   topk_idxs  : (S_q, topk_max)    int32 **2-D**
        # Returns dq (S_q, H, D), dkv (S_kv, D), d_sink (H,).
        q_shd = q.squeeze(0).contiguous()
        kv_td = kv.squeeze(0).contiguous()
        out_shd = out.squeeze(0).contiguous()
        dout_shd = dout.squeeze(0).to(out.dtype).contiguous()
        idx_2d = topk_idxs.squeeze(0).contiguous().int()
        sink_f32 = attn_sink.to(torch.float32).contiguous()

        result = _cudnn_sparse_attention_backward(  # type: ignore[misc]
            q_shd,
            kv_td,
            out_shd,
            dout_shd,
            lse,
            sink_f32,
            idx_2d,
            softmax_scale=softmax_scale,
        )
        # `sparse_attention_backward_wrapper` returns a TupleDict with keys
        # 'dq', 'dkv', 'd_sink' (see its docstring). Reattach the XTuner batch
        # dim on dq/dkv so the upstream autograd sees the same `[1, ...]`
        # layout it sent in.
        dq = result["dq"].unsqueeze(0) if needs_q else None
        dkv = result["dkv"].unsqueeze(0) if needs_kv else None
        # d_sink is 1-D `[H]` in both libs — match attn_sink's original dtype
        # so downstream optim sees a consistent dtype between backends.
        dsink = result["d_sink"].to(attn_sink.dtype) if needs_sink else None
        return dq, dkv, dsink, None, None, None


def flash_mla_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for :func:`sparse_attn` using FlashMLA's forward kernel.

    Forward calls ``flash_mla.flash_mla_sparse_fwd``; backward re-runs the native
    sparse_attn under ``autograd.Function`` (Phase-1 contract — see module
    docstring).

    Args:
        q (torch.Tensor): Query, ``[1, total_tokens, num_heads, head_dim]``,
            bfloat16 (FlashMLA requirement).
        kv (torch.Tensor): Shared K/V, ``[1, T_total, head_dim]``, bfloat16.
        attn_sink (torch.Tensor): Per-head sink logit, ``[num_heads]``. Cast to
            fp32 internally (FlashMLA's ``attn_sink`` API).
        topk_idxs (torch.Tensor): Top-k indices, ``[1, total_tokens, k]``. Int
            tensor; ``-1`` marks masked slots. Cast to int32 internally
            (FlashMLA's ``indices`` API).
        softmax_scale (float): Softmax scale (typically ``head_dim ** -0.5``).
        cu_seq_lens (torch.Tensor): 1D int32 cumulative per-sample query token
            counts. Unused by FlashMLA itself (one packed sample per call) but
            forwarded into the backward's native re-run.

    Returns:
        torch.Tensor: Attention output, ``[1, total_tokens, num_heads,
        head_dim]``, dtype matching ``q``.
    """
    if not _flash_mla_topk_ok(topk_idxs.size(-1)):
        # Per-layer fallback: V4's ``compress_ratio == 0`` layers have
        # topk == sliding_window (128 in V4), which fails FlashMLA SM90's
        # ``params.topk % (2*B_TOPK) == 0`` assertion. Use native sparse_attn
        # for these layers so a model-wide ``backend="flash_mla"`` toggle is
        # safe across the heterogeneous V4 stack.
        return _native_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, cu_seq_lens)
    return cast(
        torch.Tensor,
        _FlashMLASparseAttnFn.apply(q, kv, attn_sink, topk_idxs, softmax_scale, cu_seq_lens),
    )


def flash_mla_sparse_attn_apply(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Branch-free FlashMLA dispatch.

    Same contract as :func:`flash_mla_sparse_attn` but without the runtime
    topk-alignment check / native fallback — the caller is responsible for
    only invoking this when the layer's topk_max is FlashMLA-compatible (see
    :func:`_flash_mla_topk_ok`). Designed for use under ``torch.compile``
    where a runtime ``if`` on ``topk_idxs.size(-1)`` cannot be constant-folded
    by dynamo and ends up baking *both* branches into the compiled graph
    (including the native branch's ``[1, S, T, D]`` fp32 expand+gather
    materialisation that costs ~32 GiB at V4 production dims).
    """
    return cast(
        torch.Tensor,
        _FlashMLASparseAttnFn.apply(q, kv, attn_sink, topk_idxs, softmax_scale, cu_seq_lens),
    )


def cudnn_sparse_attn_apply(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Branch-free cudnn dispatch.

    Same contract as :func:`cudnn_sparse_attn` but without the runtime
    topk-alignment check / native fallback — see
    :func:`flash_mla_sparse_attn_apply` for the rationale.
    """
    return cast(
        torch.Tensor,
        _CudnnSparseAttnFn.apply(q, kv, attn_sink, topk_idxs, softmax_scale, cu_seq_lens),
    )


def cudnn_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for :func:`sparse_attn` using FlashMLA fwd + cudnn bwd.

    Forward calls ``flash_mla.flash_mla_sparse_fwd``; backward calls
    ``cudnn.deepseek_sparse_attention.sparse_attention_backward_wrapper`` — the
    full Phase-2 contract (see module docstring).

    Args:
        q (torch.Tensor): Query, ``[1, total_tokens, num_heads, head_dim]``,
            bfloat16.
        kv (torch.Tensor): Shared K/V, ``[1, T_total, head_dim]``, bfloat16.
        attn_sink (torch.Tensor): Per-head sink logit, ``[num_heads]``.
        topk_idxs (torch.Tensor): Top-k indices, ``[1, total_tokens, k]``. Int;
            ``-1`` marks masked slots.
        softmax_scale (float): Softmax scale (typically ``head_dim ** -0.5``).
        cu_seq_lens (torch.Tensor): 1D int32 cumulative per-sample query token
            counts. Unused by both kernels (one packed sample per call) but
            preserved in the signature to stay drop-in with :func:`sparse_attn`.

    Returns:
        torch.Tensor: Attention output, ``[1, total_tokens, num_heads,
        head_dim]``, dtype matching ``q``.
    """
    if not _flash_mla_topk_ok(topk_idxs.size(-1)):
        # Same per-layer fallback as ``flash_mla_sparse_attn`` — when FlashMLA
        # can't run this layer the cudnn backward kernel has nothing to consume
        # (it requires FlashMLA's ``lse``). Use native end-to-end for the layer.
        return _native_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, cu_seq_lens)
    return cast(
        torch.Tensor,
        _CudnnSparseAttnFn.apply(q, kv, attn_sink, topk_idxs, softmax_scale, cu_seq_lens),
    )
