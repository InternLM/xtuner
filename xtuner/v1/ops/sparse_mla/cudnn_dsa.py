# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import Tensor

from .protocol import SparseMLAOutputs
from .tilelang import _validate_tilelang_sparse_mla_inputs


def cudnn_dsa_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    _validate_cudnn_dsa_sparse_mla_inputs(q, kv, indices, value_dim)
    indices = indices.to(torch.int32).contiguous()
    raw_output, softmax_lse, _ = _cudnn_dsa_sparse_mla_forward(q, kv, indices, scaling)
    return SparseMLAOutputs(raw_output=raw_output, softmax_lse=softmax_lse)


def _validate_cudnn_dsa_sparse_mla_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_dim: int | None,
) -> None:
    _validate_tilelang_sparse_mla_inputs(q, kv, indices, value_dim)
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError("cuDNN DSA SparseMLA currently supports kv_group=1 only.")


@torch.library.custom_op("sparse_mla::cudnn_dsa_sparse_mla_forward", mutates_args=(), device_types="cuda")
def _cudnn_dsa_sparse_mla_forward(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.to(torch.int32).contiguous()
    out, lse_log2 = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
    return out, lse_log2 * 0.6931471805599453, lse_log2


@_cudnn_dsa_sparse_mla_forward.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    out = q.new_empty((*q.shape[:-1], 512))
    lse = q.new_empty(q.shape[:-1], dtype=torch.float32)
    return out, lse, lse


def _setup_cudnn_dsa_sparse_mla_context(ctx, inputs, output) -> None:
    q, kv, indices, scaling = inputs
    raw_output, softmax_lse, _ = output
    ctx.scaling = scaling
    ctx.save_for_backward(q, kv, indices, raw_output, softmax_lse)


def _cudnn_dsa_sparse_mla_backward(ctx, grad_output: Tensor, grad_lse: Tensor, grad_lse_log2: Tensor):
    q, kv, indices, raw_output, softmax_lse = ctx.saved_tensors
    dq, dkv = _cudnn_dsa_sparse_mla_backward_op(
        q,
        kv,
        raw_output,
        grad_output.contiguous(),
        indices,
        softmax_lse,
        ctx.scaling,
    )
    return dq, dkv, None, None


_cudnn_dsa_sparse_mla_forward.register_autograd(
    _cudnn_dsa_sparse_mla_backward, setup_context=_setup_cudnn_dsa_sparse_mla_context
)


@torch.library.custom_op("sparse_mla::cudnn_dsa_sparse_mla_backward", mutates_args=(), device_types="cuda")
def _cudnn_dsa_sparse_mla_backward_op(
    q: Tensor,
    kv: Tensor,
    raw_output: Tensor,
    grad_output: Tensor,
    indices: Tensor,
    softmax_lse: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor]:
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import sparse_attention_backward_wrapper

    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError("cuDNN DSA SparseMLA backward currently supports kv_group=1 only.")

    indices_2d = indices[:, 0, :]
    # cuDNN uses a per-query valid length and expects the physical index tensor
    # to be non-negative. GLM pads invalid tail slots with -1 after top-k.
    topk_length = (indices_2d != -1).sum(dim=-1, dtype=torch.int32).contiguous()
    topk_idxs = indices_2d.clamp_min(0).to(torch.int32).contiguous()
    attn_sink = torch.full((q.shape[1],), float("-inf"), dtype=torch.float32, device=q.device)

    outputs = sparse_attention_backward_wrapper(
        q,
        kv[:, 0, :].contiguous(),
        raw_output.contiguous(),
        grad_output,
        softmax_lse.contiguous(),
        attn_sink,
        topk_idxs,
        softmax_scale=scaling,
        topk_length=topk_length,
    )
    return outputs["dq"], outputs["dkv"].unsqueeze(1)


@_cudnn_dsa_sparse_mla_backward_op.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    raw_output: Tensor,
    grad_output: Tensor,
    indices: Tensor,
    softmax_lse: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(kv)


def ensure_cudnn_dsa_runtime_available() -> None:
    try:
        import cudnn  # noqa: F401
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import (  # noqa: F401
            sparse_attention_backward_wrapper,
        )
    except Exception as exc:
        raise RuntimeError(
            "cuDNN DSA SparseMLA requires nvidia-cudnn-frontend with "
            "cudnn.deepseek_sparse_attention.sparse_attention_backward support."
        ) from exc

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError(f"cuDNN DSA SparseMLA requires SM90+, found SM{major}0.")
