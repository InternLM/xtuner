# Copyright (c) OpenMMLab. All rights reserved.
"""Explicit LMDeploy FP8/BF16 backends for the frozen GLM-5.3 DSA indexer.

The adapter uses internal LMDeploy APIs verified at revision 3228395546dfb2f6427faf39ec0c0c285f4c378a (FP8) and
1c52dae9 (BF16). Select a backend in the model config; unavailable dependencies/hardware raise an error. No environment
option or automatic fallback changes native KPool.
"""

import importlib
from inspect import signature
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from xtuner.v1.data_proto import SequenceContext

from .kpool import build_pool_index, gather_kpool_features, kpool_output_width, kpool_topk_indices, pool_causal_ranges


def validate_lmdeploy_kpool_config(
    *,
    index_head_dim: int,
    index_n_heads: int,
    index_kpool: int,
    index_topk: int,
    balance_sp: bool,
) -> None:
    if index_head_dim != 128 or index_kpool != 4 or index_topk not in (2048, 8192):
        raise ValueError("LMDeploy KPool requires index_head_dim=128, index_kpool=4 and index_topk=2048 or 8192")
    if index_n_heads not in (32, 64, 128):
        raise ValueError("LMDeploy KPool requires index_n_heads=32, 64 or 128")
    if balance_sp:
        raise ValueError("LMDeploy KPool does not support indexer_balance_sp=True")


def ensure_lmdeploy_kpool_runtime_available(precision: str) -> None:
    """Validate dependency interfaces at build time without probing CUDA.

    CPU/meta model construction is supported; the selected device is checked on first use. BF16 needs serving
    compression and XTuner's TileLang scorer, while FP8 additionally needs DeepGEMM's contiguous MQA operators.
    """
    if precision not in ("fp8", "bf16"):
        raise ValueError(f"Unsupported LMDeploy KPool precision: {precision}")
    try:
        backend = importlib.import_module("lmdeploy.pytorch.backends.cuda.kpool")
        operators = [getattr(backend, "kpool_compress_quantize_cuda")]
        if precision == "bf16":
            if "out_dtype" not in signature(operators[0]).parameters:
                raise AttributeError("kpool_compress_quantize_cuda must accept out_dtype for BF16")
            from .tilelang import ensure_tilelang_runtime_available

            ensure_tilelang_runtime_available()
        else:
            kernels = importlib.import_module("lmdeploy.pytorch.kernels.cuda.kpool")
            reference = importlib.import_module("lmdeploy.pytorch.nn.kpool")
            envs = importlib.import_module("lmdeploy.pytorch.envs")
            # Use the same loader as execution, including its dependency/API
            # checks. This loader does not create CUDA execution metadata.
            deep_gemm = backend._get_deep_gemm()
            operators.extend(
                [
                    getattr(backend, name)
                    for name in (
                        "_get_deep_gemm",
                        "_mqa_has_local_columns",
                        "kpool_expand_groups_cuda",
                        "kpool_select_groups_cuda",
                    )
                ]
            )
            operators.extend([kernels.rotate_kpool_query, reference.kpool_quantize_fp8, deep_gemm.fp8_mqa_logits])
            if not isinstance(envs.dsa_indexer_max_logits_mb, (int, float)) or envs.dsa_indexer_max_logits_mb <= 0:
                raise ValueError("dsa_indexer_max_logits_mb must be a positive budget")
        if not all(callable(operator) for operator in operators):
            raise TypeError("required KPool operators must be callable")
    except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"indexer_backend='lmdeploy_{precision}' requires compatible LMDeploy KPool "
            f"{'and DeepGEMM MQA' if precision == 'fp8' else 'BF16 compression and TileLang'} operators: {error}"
        ) from error


def require_lmdeploy_kpool_device(device: torch.device, precision: str) -> None:
    """Fail on an unsupported execution device; never choose another
    backend."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError(f"indexer_backend='lmdeploy_{precision}' requires an SM90+ CUDA device, got {device}")


@torch.no_grad()
def build_pools_lmdeploy_bf16(
    k: Tensor,
    gate_scores: Tensor,
    kpool_ape: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_kpool: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build unrotated BF16 pools with serving FP32 compression
    intermediates."""
    if k.dtype != torch.bfloat16:
        raise TypeError("lmdeploy_bf16 KPool requires BF16 keys")
    pool_index = build_pool_index(seq_ctx, k.shape[0], index_kpool, k.device)
    valid = pool_index >= 0
    safe = pool_index.clamp_min(0)
    values, _ = _compress_storage(
        torch.where(valid[..., None], k[safe], 0),
        gate_scores[safe].masked_fill(~valid[..., None], -float("inf")),
        kpool_ape,
        "decode",
        True,
        out_dtype=torch.bfloat16,
    )
    values = torch.where(valid.any(dim=1, keepdim=True), values, 0)
    return values, pool_index.to(torch.int32), valid.all(dim=-1)


@torch.no_grad()
def lmdeploy_bf16_kpool_topk_indices(
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
    """Share TileLang's full selection pipeline with an explicit BF16 pool
    builder."""
    _validate_indexer(
        q,
        k,
        gate_scores,
        weights,
        index_head_dim,
        index_topk,
        index_kpool,
        query_chunk_size,
        precision="bf16",
        balance_sp=balance_sp,
    )
    if q.shape[1] == 0:
        return torch.full(
            (0, 1, kpool_output_width(index_topk, index_kpool, alignment)), -1, device=q.device, dtype=torch.int32
        )
    return kpool_topk_indices(
        q,
        k,
        gate_scores,
        weights,
        kpool_ape,
        seq_ctx,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        index_kpool=index_kpool,
        always_select_tail=always_select_tail,
        alignment=alignment,
        query_chunk_size=query_chunk_size,
        balance_sp=balance_sp,
        sp_full_pool_work_ratio=sp_full_pool_work_ratio,
        build_pools_fn=build_pools_lmdeploy_bf16,
    )


@torch.no_grad()
def lmdeploy_fp8_kpool_topk_indices(
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
        query_chunk_size (int | None): Query row cap; the logits budget from
            ``LMDEPLOY_DSA_INDEXER_MAX_LOGITS_MB`` also bounds padded backing
            storage for the active compact or legacy MQA API.
        balance_sp (bool): Query redistribution is not supported by this backend.
        sp_full_pool_work_ratio (float): Unused without query redistribution.

    Returns:
        Tensor: Global token IDs shaped ``[local_tokens, 1, aligned_width]``.
    """
    del sp_full_pool_work_ratio
    _validate_indexer(
        q,
        k,
        gate_scores,
        weights,
        index_head_dim,
        index_topk,
        index_kpool,
        query_chunk_size,
        precision="fp8",
        balance_sp=balance_sp,
    )
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

    full_k, full_scores = gather_kpool_features(k, gate_scores, seq_ctx)
    pool_index = build_pool_index(seq_ctx, full_k.shape[0], index_kpool, full_k.device)
    safe_index = pool_index.clamp_min(0)
    # Partial and padding pools never enter causal selection ranges. Initialize
    # their invalid slots so fused compression never reads arbitrary memory.
    valid = (pool_index >= 0).unsqueeze(-1)
    grouped_k = torch.where(valid, full_k[safe_index], 0)
    grouped_scores = torch.where(valid, full_scores[safe_index], 0)
    # Decode compression is intentional: SequenceContext has no serving prompt
    # or chunk schedule. This does not reproduce prefill's online accumulation.
    # Selection needs only quantized storage, never a differentiable FP32 view.
    pool_values, pool_scales = _compress_storage(grouped_k, grouped_scores, kpool_ape, "decode", True)

    query = q.squeeze(0)
    starts, ends = pool_causal_ranges(seq_ctx, pool_index, query.shape[0], q.device)
    group_lengths = ends - starts
    selected = torch.full((query.shape[0], index_topk // index_kpool), -1, device=q.device, dtype=torch.int32)
    local_columns = _mqa_has_local_columns()
    mqa = _get_deep_gemm().fp8_mqa_logits
    chunk_size = _score_chunk_size(pool_index.shape[0], q.shape[2], query_chunk_size, local_columns=local_columns)
    pool_scale = pool_scales.flatten().contiguous()
    for start in range(0, query.shape[0], chunk_size):
        chunk = slice(start, min(start + chunk_size, query.shape[0]))
        # Rotation temporarily uses FP32. Chunk it too, rather than retaining
        # full-sequence FP32 rotation and FP8 quantization intermediates.
        rotated = rotate_kpool_query(query[chunk].contiguous())
        query_fp8, query_scale = kpool_quantize_fp8(rotated, block_size=index_head_dim, round_scale=True)
        head_gate = weights.squeeze(0)[chunk].float() * q.shape[2] ** -0.5
        query_weight = (head_gate * query_scale.squeeze(-1) * index_head_dim**-0.5).contiguous()
        logits = mqa(
            query_fp8.contiguous(),
            (pool_values, pool_scale),
            query_weight,
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


def _compress_storage(
    keys: Tensor,
    scores: Tensor,
    ape: Tensor,
    mode: str,
    round_scale: bool,
    *,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[Tensor, Tensor]:
    _validate_compression(keys, scores, ape, mode)
    if out_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        raise ValueError("KPool storage must be FP8 or BF16")
    if keys.shape[0] == 0:
        return (
            torch.empty((0, keys.shape[-1]), device=keys.device, dtype=out_dtype),
            torch.empty((0, 1), device=keys.device, dtype=torch.float32),
        )
    if keys.is_cuda:
        # This adapter includes LMDeploy's SM90+ fused/reference dispatch.
        from lmdeploy.pytorch.backends.cuda.kpool import kpool_compress_quantize_cuda

        if out_dtype == torch.bfloat16:
            return kpool_compress_quantize_cuda(
                keys, scores, ape, mode=mode, round_scale=round_scale, out_dtype=out_dtype
            )
        return kpool_compress_quantize_cuda(keys, scores, ape, mode=mode, round_scale=round_scale)
    from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8

    if out_dtype == torch.bfloat16:
        pooled = kpool_compress(keys, scores, ape, mode=mode, rotate=False)
        return pooled.to(out_dtype), torch.ones_like(pooled[:, :1], dtype=torch.float32)
    pooled = kpool_compress(keys, scores, ape, mode=mode)
    return kpool_quantize_fp8(pooled, block_size=keys.shape[-1], round_scale=round_scale)


def _score_chunk_size(pool_count: int, heads: int, query_chunk_size: int | None, *, local_columns: bool) -> int:
    from lmdeploy.pytorch import envs

    # DeepGEMM attention.hpp: compact columns align N to 256; the absolute
    # column API adds a 256-column guard before aligning its row stride to 64.
    # Both APIs pad query rows to block_q, even for the final short chunk.
    # https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/csrc/apis/attention.hpp#L155-L171
    block_q = 128 // heads
    stride = ((pool_count + 255) // 256 * 256) if local_columns else ((pool_count + 256 + 63) // 64 * 64)
    minimum_bytes = block_q * stride * 4
    budget = envs.dsa_indexer_max_logits_mb * (1 << 20)
    budget_rows = budget // minimum_bytes * block_q
    if budget_rows == 0:
        raise ValueError(
            f"DSA score budget {budget} bytes is smaller than the minimum aligned allocation {minimum_bytes} bytes. "
            "Increase LMDEPLOY_DSA_INDEXER_MAX_LOGITS_MB."
        )
    return min(1024 if query_chunk_size is None else query_chunk_size, budget_rows)


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
    *,
    precision: Literal["fp8", "bf16"],
    balance_sp: bool,
) -> None:
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != (*q.shape[:2], head_dim):
        raise ValueError("KPool expects q=[1,S,H,D] and k=[1,S,D]")
    if scores.shape != k.shape or weights.shape != q.shape[:-1]:
        raise ValueError("KPool gate/weight shapes must match the keys/queries")
    validate_lmdeploy_kpool_config(
        index_head_dim=head_dim,
        index_n_heads=q.shape[2],
        index_kpool=pool_size,
        index_topk=topk,
        balance_sp=balance_sp,
    )
    if q.shape[-1] != head_dim:
        raise ValueError("KPool query head dimension must match index_head_dim")
    require_lmdeploy_kpool_device(q.device, precision)
    if any(value.device != q.device for value in (k, scores, weights)):
        raise ValueError("LMDeploy KPool requires tensors on the same CUDA device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise TypeError("LMDeploy KPool requires BF16 queries/keys and FP32 head weights")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
