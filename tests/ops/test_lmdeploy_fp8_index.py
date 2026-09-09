"""GPU tests for XTuner's DeepGEMM-backed FP8 Indexer path.

The public XTuner adapter test uses the contiguous DeepGEMM MQA API used by
the training/prefill path.  Quantization and top-k selector tests exercise
the surrounding contracts without invoking a Triton score fallback.
"""

from __future__ import annotations

import subprocess
import sys
from functools import cache

import pytest
import torch


@cache
def _fp8_index_available() -> bool:
    if not hasattr(torch, "float8_e4m3fn") or not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return False
    if capability[0] < 9:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import triton"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


@cache
def _deepgemm_mqa_available() -> bool:
    """Whether the contiguous DeepGEMM Indexer API is available."""
    if not hasattr(torch, "float8_e4m3fn") or not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_capability()[0] < 9:
            return False
        import deep_gemm
    except Exception:
        return False
    return hasattr(deep_gemm, "fp8_mqa_logits") or hasattr(deep_gemm, "fp8_fp4_mqa_logits")


@cache
def _lmdeploy_topk_available() -> bool:
    if not _fp8_index_available():
        return False
    try:
        import tilelang  # noqa: F401
    except Exception:
        return False
    return True


def _reference_scores(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    dot = torch.einsum("qhd,kd->qhk", q_fp8.float(), k_fp8.float())
    score = torch.relu(dot) * (q_scale * weights).float().unsqueeze(-1)
    score = score.sum(dim=-2) * k_scale.float().unsqueeze(0)
    key_ids = torch.arange(k_fp8.size(0), device=k_fp8.device)
    valid = (key_ids[None, :] >= starts[:, None]) & (key_ids[None, :] < ends[:, None])
    return score.masked_fill(~valid, -torch.inf)


def _ue8m0_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = x.float().abs().amax(dim=-1).clamp_min(1e-6)
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    scale = torch.pow(torch.tensor(2.0, device=x.device), exponent)
    value = (x.float() / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    return value.to(torch.float8_e4m3fn), scale.float()


@pytest.mark.skipif(not _fp8_index_available(), reason="requires SM90 CUDA and Triton")
def test_indexer_fp8_quant_matches_ue8m0_reference():
    from xtuner.v1.ops.sparse_mla.indexer_fp8_quant import indexer_fp8_quant

    device = torch.device("cuda")
    torch.manual_seed(20260907)
    values = torch.randn(259, 128, device=device, dtype=torch.bfloat16)
    values[0].zero_()
    values[1].fill_(448)
    actual, actual_scale = indexer_fp8_quant(values.contiguous())
    expected, expected_scale = _ue8m0_reference(values)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


@pytest.mark.skipif(not _deepgemm_mqa_available(), reason="requires SM90 CUDA and contiguous DeepGEMM MQA")
def test_lmdeploy_adapter_preserves_packed_global_topk_ids():
    from xtuner.v1.ops.sparse_mla.lmdeploy_fp8_index import lmdeploy_fp8_indexer_topk

    device = torch.device("cuda")
    torch.manual_seed(20260908)
    seq_lens = [2048, 2048]
    total = sum(seq_lens)
    heads, head_dim, topk = 32, 128, 2048
    q = (torch.randn(1, total, heads, head_dim, device=device) * 1.5).to(torch.float8_e4m3fn)
    k = (torch.randn(1, total, head_dim, device=device) * 1.5).to(torch.float8_e4m3fn)
    q_scale = torch.rand(1, total, heads, device=device, dtype=torch.float32) + 0.5
    k_scale = torch.rand(1, total, device=device, dtype=torch.float32) + 0.5
    weights = torch.rand(1, total, heads, device=device, dtype=torch.float32) + 0.2
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int32)
    actual = lmdeploy_fp8_indexer_topk(q, q_scale, k, k_scale, weights, cu, cu, 0, head_dim, topk)

    starts = torch.cat(
        [
            torch.full((length,), start, device=device, dtype=torch.int32)
            for start, length in zip([0, seq_lens[0]], seq_lens)
        ]
    )
    ends = torch.cat(
        [
            start + torch.arange(1, length + 1, device=device, dtype=torch.int32)
            for start, length in zip([0, seq_lens[0]], seq_lens)
        ]
    )
    expected_scores = _reference_scores(
        q[0],
        q_scale[0],
        k[0],
        k_scale[0],
        weights[0] * (heads**-0.5) * (head_dim**-0.5),
        starts,
        ends,
    )
    expected = torch.full((total, topk), -1, device=device, dtype=torch.int32)
    for row in range(total):
        count = min(topk, int(ends[row] - starts[row]))
        _, ids = expected_scores[row, starts[row] : ends[row]].topk(count)
        expected[row, :count] = ids.to(torch.int32) + starts[row]
    # DeepGEMM and torch reductions can differ at a near-tie.  Compare set
    # recall rather than relying on an ordering of equal scores.
    actual_set = actual[:, 0].sort(dim=-1).values
    expected_set = expected.sort(dim=-1).values
    valid = expected_set >= 0
    recall = ((actual_set[..., None] == expected_set[..., None, :]).any(dim=-1) & valid).sum() / valid.sum()
    assert float(recall) >= 0.98


@pytest.mark.skipif(not _lmdeploy_topk_available(), reason="requires SM90 CUDA and TileLang")
def test_lmdeploy_sparse_topk_selector_matches_reference_sets():
    from xtuner.v1.ops.sparse_mla.lmdeploy_sparse_index_topk import sparse_index_topk

    device = torch.device("cuda")
    torch.manual_seed(20260909)
    rows, width, topk = 3, 4096, 2048
    scores = torch.randn(rows, width, device=device, dtype=torch.float32)
    seqlens = torch.tensor([4096, 2700, 200], device=device, dtype=torch.int32)
    actual = sparse_index_topk(
        scores,
        torch.ones(rows, device=device, dtype=torch.int32),
        seqlens,
        topk,
    )
    ids = torch.arange(width, device=device)[None, :]
    masked = scores.masked_fill(ids >= seqlens[:, None], -torch.inf)
    expected = torch.full((rows, topk), -1, device=device, dtype=torch.int32)
    for row in range(rows):
        count = min(topk, int(seqlens[row]))
        expected[row, :count] = masked[row].topk(count).indices.to(torch.int32)
    actual_set = actual.sort(dim=-1).values
    expected_set = expected.sort(dim=-1).values
    torch.testing.assert_close(actual_set, expected_set)
