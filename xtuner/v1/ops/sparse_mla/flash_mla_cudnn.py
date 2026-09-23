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

from .cudnn_dsa import _cudnn_dsa_sparse_mla_backward_op, ensure_cudnn_dsa_runtime_available
from .flash_mla import _FLASH_MLA_HEAD_ALIGNMENT, ensure_flash_mla_runtime_available
from .protocol import SparseMLAOutputs
from .tilelang import validate_sparse_mla_inputs


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
    # FlashMLA returns natural-log LSE; cuDNN's backward (below) expects log2-space LSE,
    # matching the same contract flash_mla.py uses for its (different) TileLang backward.
    return raw_output, softmax_lse, softmax_lse * 1.4426950408889634


@_flash_mla_cudnn_forward.register_fake
def _(q: Tensor, kv: Tensor, indices: Tensor, scaling: float, value_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    out = q.new_empty((*q.shape[:-1], value_dim))
    softmax_lse = q.new_empty(q.shape[:-1], dtype=torch.float32)
    lse_log2 = q.new_empty(q.shape[:-1], dtype=torch.float32)
    return out, softmax_lse, lse_log2


def _setup_flash_mla_cudnn_context(ctx, inputs, output) -> None:
    q, kv, indices, scaling, _ = inputs
    raw_output, _, lse_log2 = output
    ctx.scaling = scaling
    ctx.save_for_backward(q, kv, indices, raw_output, lse_log2)


def _flash_mla_cudnn_backward(ctx, grad_output: Tensor, grad_lse: Tensor, grad_lse_log2: Tensor):
    q, kv, indices, raw_output, lse_log2 = ctx.saved_tensors
    dq, dkv = _cudnn_dsa_sparse_mla_backward_op(
        q,
        kv,
        raw_output,
        grad_output.contiguous(),
        indices,
        lse_log2,
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
