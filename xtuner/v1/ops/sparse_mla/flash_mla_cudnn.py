# Copyright (c) OpenMMLab. All rights reserved.
"""FlashMLA forward + cuDNN backward SparseMLA backend for GLM-5.3-Flash's NoPE
DSA.

Design doc section 3.5.2: NoPE's absorbed latent is 512-wide (``qk_rope_head_dim=0``, no
rope tail to concatenate), which FlashMLA supports natively (Automodel's
``_SUPPORTED_ATTENTION_HEAD_DIMS = (512, 576)``) and XTuner's existing cuDNN backward has no
head-dim hardcoding either. Neither of XTuner's existing two-kernel backends works as-is:
``flash_mla`` pairs FlashMLA fwd with a **TileLang** bwd (``tail_dim=0`` triggers
``next_power_of_2(0) == 2`` in that kernel), and ``cudnn_dsa`` pairs a **TileLang** fwd with
cuDNN bwd (same ``tail_dim=0`` problem, on the forward side). This backend pairs the two
halves that don't touch TileLang at all, so no kernel needs the NoPE ``tail_dim=0`` fix.

This module intentionally does not import ``flash_mla.py``'s ``_flash_mla_sparse_forward`` --
that op's ``register_autograd`` is permanently wired to the TileLang backward. Instead it
calls the same underlying ``flash_mla_sparse_fwd`` kernel through its own custom op, paired
with cuDNN's backward custom op from ``cudnn_dsa.py``.
"""

import torch
from torch import Tensor

from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .cudnn_dsa import _cudnn_dsa_sparse_mla_backward_op, ensure_cudnn_dsa_runtime_available
from .flash_mla import _FLASH_MLA_HEAD_ALIGNMENT, ensure_flash_mla_runtime_available
from .protocol import SparseMLAOutputs
from .tilelang import validate_sparse_mla_inputs


logger = get_logger()
_deterministic_backward_warned = False


def flash_mla_cudnn_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    """FlashMLA forward + cuDNN backward, for NoPE (512-wide) absorbed
    latents."""
    num_q_heads = q.shape[1]
    if num_q_heads % _FLASH_MLA_HEAD_ALIGNMENT != 0:
        raise RuntimeError(
            f"FlashMLA sparse prefill requires query heads to be aligned to "
            f"{_FLASH_MLA_HEAD_ALIGNMENT}, but got {num_q_heads} heads."
        )
    validate_sparse_mla_inputs(q, kv, indices, value_dim)
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError("flash_mla_cudnn SparseMLA currently supports kv_group=1 only.")

    indices = indices.to(torch.int32).contiguous()
    scale = float(scaling) if scaling is not None else q.shape[-1] ** -0.5

    raw_output, softmax_lse, _ = _flash_mla_cudnn_forward(q, kv, indices, scale, value_dim or 512)
    return SparseMLAOutputs(raw_output=raw_output, softmax_lse=softmax_lse)


@torch.library.custom_op("sparse_mla::flash_mla_cudnn_forward", mutates_args=(), device_types="cuda")
def _flash_mla_cudnn_forward(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float,
    value_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    from flash_mla import flash_mla_sparse_fwd

    raw_output, _, softmax_lse = flash_mla_sparse_fwd(
        q.contiguous(),
        kv.contiguous(),
        indices.to(torch.int32).contiguous(),
        scaling,
        d_v=value_dim,
    )
    # FlashMLA returns natural-log LSE. cuDNN's backward (below) also expects natural-log
    # LSE since b03bf018 switched _cudnn_dsa_sparse_mla_backward_op off the log2 contract;
    # the third slot keeps the legacy log2 value only for custom-op schema stability.
    return raw_output, softmax_lse, softmax_lse * 1.4426950408889634


@_flash_mla_cudnn_forward.register_fake
def _(q: Tensor, kv: Tensor, indices: Tensor, scaling: float, value_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    out = q.new_empty((*q.shape[:-1], value_dim))
    softmax_lse = q.new_empty(q.shape[:-1], dtype=torch.float32)
    lse_log2 = q.new_empty(q.shape[:-1], dtype=torch.float32)
    return out, softmax_lse, lse_log2


def _setup_flash_mla_cudnn_context(ctx, inputs, output) -> None:
    q, kv, indices, scaling, _ = inputs
    raw_output, softmax_lse, _ = output
    ctx.scaling = scaling
    ctx.save_for_backward(q, kv, indices, raw_output, softmax_lse)


def _deterministic_sparse_mla_backward(
    q: Tensor,
    kv: Tensor,
    raw_output: Tensor,
    grad_output: Tensor,
    indices: Tensor,
    softmax_lse: Tensor,
    scaling: float,
    chunk_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Ordered FP32 reference backward selected by ``XTUNER_DETERMINISTIC``.

    cuDNN's sparse-attention backward accumulates dKV with unordered atomics, so repeated
    runs diverge bitwise even under ``torch.use_deterministic_algorithms`` (dq is stable,
    dkv is not). This fallback recomputes per-query contributions in a fixed chunk order
    and accumulates dKV through ``index_add_``, which PyTorch routes to a deterministic
    implementation in deterministic mode. It is slower than the cuDNN kernel and only runs
    when ``XTUNER_DETERMINISTIC=true``.

    Args:
        q (Tensor): Queries ``[total_q, num_heads, head_dim]`` (bf16).
        kv (Tensor): Absorbed latents ``[total_kv, 1, head_dim]`` (bf16); K and V share the
            tensor, with V occupying the leading ``value_dim`` columns.
        raw_output (Tensor): Saved forward output ``[total_q, num_heads, value_dim]``.
        grad_output (Tensor): Output cotangent, same shape as ``raw_output``.
        indices (Tensor): Top-k key indices ``[total_q, 1, topk]``, -1 padded.
        softmax_lse (Tensor): Natural-log LSE of the scaled scores ``[total_q, num_heads]``.
        scaling (float): Score scale applied before the softmax.
        chunk_size (int): Queries per chunk; bounds the gathered-KV temporaries.

    Returns:
        tuple[Tensor, Tensor]: ``dq`` like ``q`` and ``dkv`` like ``kv``.
    """
    total_q, _, head_dim = q.shape
    value_dim = raw_output.shape[-1]
    kv_flat = kv[:, 0]
    idx = indices[:, 0].to(torch.int64)
    valid = idx != -1
    safe_idx = idx.clamp_min(0)

    dq = torch.empty_like(q)
    dkv_accum = torch.zeros(kv_flat.shape, dtype=torch.float32, device=q.device)
    for start in range(0, total_q, chunk_size):
        end = min(start + chunk_size, total_q)
        sel = safe_idx[start:end]  # [C, topk]
        kv_sel = kv_flat[sel].float()  # [C, topk, head_dim]
        q_c = q[start:end].float()
        go_c = grad_output[start:end].float()
        out_c = raw_output[start:end].float()
        lse_c = softmax_lse[start:end]

        scores = torch.einsum("chd,ckd->chk", q_c, kv_sel) * scaling  # [C, heads, topk]
        scores = scores.masked_fill(~valid[start:end, None], float("-inf"))
        probs = torch.exp(scores - lse_c[..., None])  # invalid slots -> exactly 0
        # rowsum(dprobs * probs) == rowsum(dO * O) because O = probs @ V.
        row_d = (go_c * out_c).sum(-1)
        dprobs = torch.einsum("chd,ckd->chk", go_c, kv_sel[..., :value_dim])
        dscores = probs * (dprobs - row_d[..., None])
        dq[start:end] = (torch.einsum("chk,ckd->chd", dscores, kv_sel) * scaling).to(q.dtype)
        dkv_part = torch.einsum("chk,chd->ckd", dscores, q_c) * scaling
        dkv_part[..., :value_dim] += torch.einsum("chk,chd->ckd", probs, go_c)
        # Invalid slots carry zero contribution (probs == 0), so scattering them into
        # row 0 via the clamped index adds exact zeros.
        dkv_accum.index_add_(0, sel.reshape(-1), dkv_part.reshape(-1, head_dim))
    return dq, dkv_accum.to(kv.dtype).unsqueeze(1)


def _flash_mla_cudnn_backward(ctx, grad_output: Tensor, grad_lse: Tensor, grad_lse_log2: Tensor):
    global _deterministic_backward_warned
    q, kv, indices, raw_output, softmax_lse = ctx.saved_tensors
    if XTUNER_DETERMINISTIC:
        if not _deterministic_backward_warned:
            logger.warning(
                "XTUNER_DETERMINISTIC=true: using the ordered FP32 reference backward for the "
                "flash_mla_cudnn SparseMLA op because cuDNN's sparse backward accumulates dKV "
                "with unordered atomics. Expect slower DSA-layer backward."
            )
            _deterministic_backward_warned = True
        dq, dkv = _deterministic_sparse_mla_backward(
            q, kv, raw_output, grad_output.contiguous(), indices, softmax_lse, ctx.scaling
        )
    else:
        dq, dkv = _cudnn_dsa_sparse_mla_backward_op(
            q,
            kv,
            raw_output,
            grad_output.contiguous(),
            indices,
            softmax_lse,
            ctx.scaling,
        )
    return dq, dkv, None, None, None


_flash_mla_cudnn_forward.register_autograd(
    _flash_mla_cudnn_backward,
    setup_context=_setup_flash_mla_cudnn_context,
)


def ensure_flash_mla_cudnn_runtime_available() -> None:
    ensure_flash_mla_runtime_available()
    ensure_cudnn_dsa_runtime_available()
