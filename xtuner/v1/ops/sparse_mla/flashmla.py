# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import Tensor

from .protocol import SparseMLAOutputs
from .tilelang import _validate_tilelang_sparse_mla_inputs


def flashmla_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    _validate_flashmla_sparse_mla_inputs(q, kv, indices, value_dim)
    indices = indices.to(torch.int32).contiguous()
    topk_length = (indices[:, 0, :] != -1).sum(dim=-1, dtype=torch.int32).contiguous()
    raw_output, softmax_lse = _flashmla_tilelang_sparse_mla_forward(
        q.contiguous(), kv.contiguous(), indices, topk_length, scaling
    )
    return SparseMLAOutputs(raw_output=raw_output, softmax_lse=softmax_lse)


def _validate_flashmla_sparse_mla_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_dim: int | None,
) -> None:
    _validate_tilelang_sparse_mla_inputs(q, kv, indices, value_dim)
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError("FlashMLA SparseMLA currently supports kv_group=1 only.")
    if indices.shape[-1] % 128 != 0:
        raise RuntimeError("FlashMLA SparseMLA requires topk to be divisible by 128.")


@torch.library.custom_op("sparse_mla::flashmla_tilelang_sparse_mla_forward", mutates_args=(), device_types="cuda")
def _flashmla_tilelang_sparse_mla_forward(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    topk_length: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor]:
    from flash_mla import flash_mla_sparse_fwd

    raw_output, _max_logits, softmax_lse = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=float(scaling) if scaling is not None else q.shape[-1] ** -0.5,
        topk_length=topk_length,
    )
    return raw_output, softmax_lse


@_flashmla_tilelang_sparse_mla_forward.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    topk_length: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor]:
    return q.new_empty((*q.shape[:-1], 512)), q.new_empty(q.shape[:-1], dtype=torch.float32)


def _setup_flashmla_tilelang_sparse_mla_context(ctx, inputs, output) -> None:
    q, kv, indices, topk_length, scaling = inputs
    raw_output, softmax_lse = output
    ctx.scaling = scaling
    ctx.save_for_backward(q, kv, indices, raw_output, softmax_lse)


def _flashmla_tilelang_sparse_mla_backward(ctx, grad_output: Tensor, grad_lse: Tensor):
    del grad_lse
    q, kv, indices, raw_output, softmax_lse = ctx.saved_tensors
    from .tilelang import _tilelang_sparse_mla_backward_op

    # FlashMLA returns natural-log LSE, while TileLang backward consumes log2 LSE.
    dq, dkv = _tilelang_sparse_mla_backward_op(
        q,
        kv,
        raw_output,
        grad_output.contiguous(),
        indices,
        (softmax_lse / 0.6931471805599453).contiguous(),
        ctx.scaling,
    )
    return dq, dkv, None, None, None


_flashmla_tilelang_sparse_mla_forward.register_autograd(
    _flashmla_tilelang_sparse_mla_backward,
    setup_context=_setup_flashmla_tilelang_sparse_mla_context,
)


def ensure_flashmla_runtime_available() -> None:
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except Exception as exc:
        raise RuntimeError("FlashMLA SparseMLA requires the FlashMLA forward runtime.") from exc

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError(f"FlashMLA SparseMLA requires SM90+, found SM{major}0.")
