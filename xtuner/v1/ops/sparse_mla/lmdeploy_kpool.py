# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy KPool compression and FP8 selection for GLM-5.3.

Compression reuses LMDeploy's forward operator. Its backward differentiates
softmax-weighted pooling and the orthonormal Hadamard transform, using a
straight-through estimator for BF16/FP8 rounding. Integer Top-K IDs remain
nondifferentiable; this backend does not unfreeze the DSA indexer.
"""

from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.comm import gather_for_sequence_parallel

from .kpool import build_pool_index, kpool_output_width, pool_causal_ranges


class KPoolCompression(NamedTuple):
    """Compression outputs, including a differentiable dequantized view.

    The FP8 values and scales are nondifferentiable metadata. Consumers that
    train pooling must use ``dequantized`` rather than quantized storage.
    """

    dequantized: Tensor
    values: Tensor
    scales: Tensor


def lmdeploy_kpool_compress(
    keys: Tensor,
    scores: Tensor,
    ape: Tensor,
    *,
    mode: str = "decode",
    round_scale: bool = True,
) -> KPoolCompression:
    """Run LMDeploy compression with an explicit first-order STE backward.

    Args:
        keys (Tensor): Keys shaped ``[groups, pool_size, head_dim]``.
        scores (Tensor): Compression logits with the same shape as keys.
        ape (Tensor): Position bias shaped ``[pool_size, head_dim]``.
        mode (str): LMDeploy's ``extend`` online or ``decode`` two-pass order.
        round_scale (bool): Round FP8 scales to powers of two.

    Returns:
        KPoolCompression: FP32 dequantized values, FP8 storage and FP32 scales.
    """
    _validate_compression(keys, scores, ape, mode)
    return KPoolCompression(*_LMDeployKPoolCompress.apply(keys, scores, ape, mode, round_scale))


def kpool_compress_backward(
    grad_output: Tensor, keys: Tensor, scores: Tensor, ape: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiate weighted pooling and rotation, treating rounding as STE.

    Args:
        grad_output (Tensor): Gradient of rotated pooled values, ``[groups, head_dim]``.
        keys (Tensor): Original keys, ``[groups, pool_size, head_dim]``.
        scores (Tensor): Original logits with the same shape as keys.
        ape (Tensor): Position bias, ``[pool_size, head_dim]``.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Key, score and position-bias gradients
        in their respective input dtypes.
    """
    # The normalized Walsh-Hadamard matrix is symmetric and orthonormal.
    grad_pooled = _normalized_hadamard(grad_output.float()).unsqueeze(1)
    probabilities = (scores.float() + ape.float()).softmax(dim=1)
    pooled = (probabilities * keys.float()).sum(dim=1, keepdim=True)
    grad_keys = probabilities * grad_pooled
    grad_scores = probabilities * (keys.float() - pooled) * grad_pooled
    grad_ape = grad_scores.sum(dim=0)
    return grad_keys.to(keys.dtype), grad_scores.to(scores.dtype), grad_ape.to(ape.dtype)


@torch.no_grad()
def _kpool_topk_indices(
    q: Tensor,
    k: Tensor,
    gate_scores: Tensor,
    weights: Tensor,
    kpool_ape: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
    index_kpool: int = 4,
    always_select_tail: bool = True,
    alignment: int = 512,
    query_chunk_size: int | None = None,
    balance_sp: bool = False,
    sp_full_pool_work_ratio: float = 0.5,
) -> Tensor:
    """Select packed causal pools using LMDeploy FP8 scoring and Top-K.

    Args:
        q (Tensor): BF16 queries, ``[1, local_tokens, heads, 128]``.
        k (Tensor): Local normalized keys, ``[1, local_tokens, 128]``.
        gate_scores (Tensor): Local compression logits with the key shape.
        weights (Tensor): Unscaled FP32 head projections, ``[1, local_tokens, heads]``.
        kpool_ape (Tensor): Position bias, ``[4, 128]``.
        seq_ctx (SequenceContext): Global document boundaries and local shard metadata.
        index_head_dim (int): Indexer head dimension, currently 128.
        index_topk (int): Token budget, currently 2048 or 8192.
        index_kpool (int): Tokens per pool, currently 4.
        always_select_tail (bool): Append the causal incomplete trailing pool.
        alignment (int): Sparse attention output-width alignment.
        query_chunk_size (int | None): Bound the score matrix's query dimension.
        balance_sp (bool): Query redistribution is not supported by this backend.
        sp_full_pool_work_ratio (float): Unused without query redistribution.

    Returns:
        Tensor: Global token IDs shaped ``[local_tokens, 1, aligned_width]``.
    """
    del sp_full_pool_work_ratio
    _validate_indexer(q, k, gate_scores, weights, index_head_dim, index_topk, index_kpool, query_chunk_size)
    if balance_sp:
        raise ValueError("LMDeploy KPool operators do not support indexer_balance_sp=True")
    output_width = kpool_output_width(index_topk, index_kpool, alignment)
    if q.shape[1] == 0:
        return torch.full((0, 1, output_width), -1, device=q.device, dtype=torch.int32)
    try:
        from lmdeploy.pytorch.backends.cuda.kpool import (
            _get_deep_gemm,
            _mqa_has_local_columns,
            kpool_expand_groups_cuda,
            kpool_select_groups_cuda,
        )
        from lmdeploy.pytorch.kernels.cuda.kpool import rotate_kpool_query
        from lmdeploy.pytorch.nn.kpool import kpool_quantize_fp8
    except ImportError as error:
        raise RuntimeError("GLM-5.3 KPool requires LMDeploy's KPool operators for this geometry") from error

    mesh = seq_ctx.sequence_parallel_mesh
    full_k = gather_for_sequence_parallel(k.squeeze(0), dim=0, sp_mesh=mesh)
    full_scores = gather_for_sequence_parallel(gate_scores.squeeze(0), dim=0, sp_mesh=mesh)
    pool_index = build_pool_index(seq_ctx, full_k.shape[0], index_kpool, full_k.device)
    safe_index = pool_index.clamp_min(0)
    # Partial and padding pools never enter causal selection ranges. Initialize
    # their invalid slots so fused compression never reads arbitrary memory.
    valid = (pool_index >= 0).unsqueeze(-1)
    grouped_k = torch.where(valid, full_k[safe_index], 0)
    grouped_scores = torch.where(valid, full_scores[safe_index], 0)
    # Generated history is closed by LMDeploy's decode two-pass operation.
    compressed = lmdeploy_kpool_compress(grouped_k, grouped_scores, kpool_ape, mode="decode")

    query = rotate_kpool_query(q.squeeze(0))
    query_fp8, query_scale = kpool_quantize_fp8(query, block_size=index_head_dim, round_scale=True)
    head_gate = weights.squeeze(0).float() * q.shape[2] ** -0.5
    query_weight = (head_gate * query_scale.squeeze(-1) * index_head_dim**-0.5).contiguous()
    starts, ends = pool_causal_ranges(seq_ctx, pool_index, query.shape[0], q.device)
    group_lengths = ends - starts
    selected = torch.full((query.shape[0], index_topk // index_kpool), -1, device=q.device, dtype=torch.int32)
    local_columns = _mqa_has_local_columns()
    mqa = _get_deep_gemm().fp8_mqa_logits
    chunk_size = 1024 if query_chunk_size is None else query_chunk_size
    pool_scale = compressed.scales.flatten().contiguous()
    for start in range(0, query.shape[0], chunk_size):
        chunk = slice(start, min(start + chunk_size, query.shape[0]))
        logits = mqa(
            query_fp8[chunk].contiguous(),
            (compressed.values, pool_scale),
            query_weight[chunk].contiguous(),
            starts[chunk].contiguous(),
            ends[chunk].contiguous(),
            clean_logits=False,
            **({"max_seqlen_k": pool_index.shape[0]} if local_columns else {}),
        )
        local_ids = kpool_select_groups_cuda(
            logits,
            group_lengths[chunk],
            group_topk=index_topk // index_kpool,
            row_starts=None if local_columns else starts[chunk],
            max_group_length=min(pool_index.shape[0], logits.shape[1]),
        )
        selected[chunk] = local_ids
        del logits
    document_starts = pool_index[starts.long(), 0].to(torch.int32)
    positions = torch.arange(q.shape[1], device=q.device) + seq_ctx.shard_start
    sequence_lengths = (positions - document_starts + 1).to(torch.int32)
    # Preserve LMDeploy's selected-group order and compact tail placement.
    # Sorting pools or moving tails to a fixed column changes MLA reductions.
    token_ids = kpool_expand_groups_cuda(
        selected,
        group_lengths,
        index_kpool,
        index_topk,
        seq_lens=sequence_lengths if always_select_tail else None,
    )
    token_ids = torch.where(token_ids >= 0, token_ids + document_starts[:, None], -1)
    return F.pad(token_ids, (0, output_width - token_ids.shape[1]), value=-1).unsqueeze(1)


class _LMDeployKPoolCompress(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, keys: Tensor, scores: Tensor, ape: Tensor, mode: str, round_scale: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        if keys.shape[0] == 0:
            values = torch.empty((0, keys.shape[-1]), device=keys.device, dtype=torch.float8_e4m3fn)
            scales = torch.empty((0, 1), device=keys.device, dtype=torch.float32)
        elif keys.is_cuda:
            from lmdeploy.pytorch.kernels.cuda.kpool import compress_kpool

            values, scales = compress_kpool(keys, scores, ape, mode=mode, round_scale=round_scale)
        else:
            from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8

            pooled = kpool_compress(keys, scores, ape, mode=mode)
            values, scales = kpool_quantize_fp8(pooled, block_size=keys.shape[-1], round_scale=round_scale)
        ctx.save_for_backward(keys, scores, ape)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(values, scales)
        return values.float() * scales, values, scales

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_output: Tensor | None, grad_values: Tensor | None, grad_scales: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        del grad_values, grad_scales
        if grad_output is None:
            return None, None, None, None, None
        keys, scores, ape = ctx.saved_tensors
        grad_keys, grad_scores, grad_ape = kpool_compress_backward(grad_output, keys, scores, ape)
        return grad_keys, grad_scores, grad_ape, None, None


def _normalized_hadamard(values: Tensor) -> Tensor:
    width = values.shape[-1]
    output = values
    stride = 1
    while stride < width:
        paired = output.reshape(*output.shape[:-1], width // (2 * stride), 2, stride)
        left, right = paired.unbind(dim=-2)
        output = torch.cat((left + right, left - right), dim=-1).reshape_as(output)
        stride *= 2
    return output * width**-0.5


def _validate_compression(keys: Tensor, scores: Tensor, ape: Tensor, mode: str) -> None:
    if keys.ndim != 3 or scores.shape != keys.shape or ape.shape != keys.shape[1:]:
        raise ValueError("Expected matching [groups, pool, width] keys/scores and [pool, width] APE")
    if keys.device != scores.device or keys.device != ape.device:
        raise ValueError("KPool compression inputs must be on the same device")
    if any(value.dtype not in (torch.bfloat16, torch.float16, torch.float32) for value in (keys, scores, ape)):
        raise TypeError("KPool compression expects BF16, FP16 or FP32 inputs")
    width = keys.shape[-1]
    if keys.shape[1] <= 0 or width <= 0 or width & (width - 1):
        raise ValueError("KPool needs a positive pool size and a power-of-two head dimension")
    if mode not in ("extend", "decode"):
        raise ValueError("KPool mode must be extend or decode")


def _validate_indexer(
    q: Tensor,
    k: Tensor,
    scores: Tensor,
    weights: Tensor,
    head_dim: int,
    topk: int,
    pool_size: int,
    query_chunk_size: int | None,
) -> None:
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != (*q.shape[:2], head_dim):
        raise ValueError("KPool expects q=[1,S,H,D] and k=[1,S,D]")
    if scores.shape != k.shape or weights.shape != q.shape[:-1]:
        raise ValueError("KPool gate/weight shapes must match the keys/queries")
    if head_dim != 128 or q.shape[-1] != head_dim or pool_size != 4 or topk not in (2048, 8192):
        raise ValueError("LMDeploy FP8 KPool requires D=128, pool=4 and topk=2048 or 8192")
    if q.shape[2] not in (32, 64, 128):
        raise ValueError("DeepGEMM KPool requires 32, 64 or 128 query heads")
    if not q.is_cuda or any(value.device != q.device for value in (k, scores, weights)):
        raise ValueError("LMDeploy FP8 KPool requires CUDA tensors on the same device")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
