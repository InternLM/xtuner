# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import Tensor

from .protocol import SparseMLAOutputs
from .tilelang import _tilelang_sparse_mla_backward_op, _validate_tilelang_sparse_mla_inputs


_FLASH_MLA_HEAD_ALIGNMENT = 64


def flash_mla_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    """Run FlashMLA sparse prefill forward."""

    num_q_heads = q.shape[1]
    assert num_q_heads % _FLASH_MLA_HEAD_ALIGNMENT == 0, (
        f"FlashMLA sparse prefill requires query heads to be aligned to {_FLASH_MLA_HEAD_ALIGNMENT}, "
        f"but got {num_q_heads} heads."
    )

    _validate_flash_mla_inputs(q, kv, indices, value_dim)

    indices = indices.to(torch.int32).contiguous()
    scale = float(scaling) if scaling is not None else q.shape[-1] ** -0.5

    raw_output, softmax_lse, _ = _flash_mla_sparse_forward(
        q,
        kv,
        indices,
        scale,
    )

    return SparseMLAOutputs(
        raw_output=raw_output,
        softmax_lse=softmax_lse,
    )


def _validate_flash_mla_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_dim: int | None,
) -> None:
    _validate_tilelang_sparse_mla_inputs(q, kv, indices, value_dim)
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError("FlashMLA SparseMLA currently supports kv_group=1 only.")


@torch.library.custom_op("sparse_mla::flash_mla_sparse_forward", mutates_args=(), device_types="cuda")
def _flash_mla_sparse_forward(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float,
) -> tuple[Tensor, Tensor, Tensor]:
    from flash_mla import flash_mla_sparse_fwd

    raw_output, _, lse_log2 = flash_mla_sparse_fwd(
        q.contiguous(),
        kv.contiguous(),
        indices.to(torch.int32).contiguous(),
        scaling,
        d_v=512,
    )
    # FlashMLA exposes its exp2-based LSE, while XTuner's public contract uses
    # natural-log LSE. Preserve both because the TileLang backward consumes log2.
    return raw_output, lse_log2 * 0.6931471805599453, lse_log2


@_flash_mla_sparse_forward.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float,
) -> tuple[Tensor, Tensor, Tensor]:
    out = q.new_empty((*q.shape[:-1], 512))
    softmax_lse = q.new_empty(q.shape[:-1], dtype=torch.float32)
    lse_log2 = q.new_empty(q.shape[:-1], dtype=torch.float32)
    return out, softmax_lse, lse_log2


def _setup_flash_mla_context(ctx, inputs, output) -> None:
    q, kv, indices, scaling = inputs
    raw_output, _, lse_log2 = output
    ctx.scaling = scaling
    ctx.save_for_backward(q, kv, indices, raw_output, lse_log2)


def _flash_mla_sparse_backward(ctx, grad_output: Tensor, grad_lse: Tensor, grad_lse_log2: Tensor):
    q, kv, indices, raw_output, lse_log2 = ctx.saved_tensors
    dq, dkv = _tilelang_sparse_mla_backward_op(
        q,
        kv,
        raw_output,
        grad_output.contiguous(),
        indices,
        lse_log2,
        ctx.scaling,
    )
    return dq, dkv, None, None


_flash_mla_sparse_forward.register_autograd(
    _flash_mla_sparse_backward,
    setup_context=_setup_flash_mla_context,
)


def ensure_flash_mla_runtime_available() -> None:
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "FlashMLA SparseMLA requires an installed flash_mla package with flash_mla_sparse_fwd support."
        ) from exc

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError(f"FlashMLA SparseMLA requires SM90+, found SM{major}0.")
