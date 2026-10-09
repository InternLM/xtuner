"""Selector routing contracts for the TileLang DSA indexer wrapper.

The inner ``tilelang_indexer_topk_from_ranges`` accepts an explicit causal
range per query row, so its dispatch can be exercised on CPU with the two
kernel primitives mocked out.  The outer ``tilelang_dsa_topk_indices`` entry
requires CUDA tensors by contract; its forwarding test is GPU-gated.
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import tilelang as tilelang_module
from xtuner.v1.ops.sparse_mla.tilelang import tilelang_indexer_topk_from_ranges


def _ranges_inputs(query_len=8, n_heads=2, head_dim=4, key_len=6):
    q = torch.zeros(query_len, n_heads, head_dim, dtype=torch.bfloat16)
    k = torch.zeros(key_len, head_dim, dtype=torch.bfloat16)
    weights = torch.ones(query_len, n_heads, dtype=torch.float32)
    starts = torch.zeros(query_len, dtype=torch.int32)
    ends = torch.arange(1, query_len + 1, dtype=torch.int32).clamp(max=key_len)
    return q, k, weights, starts, ends


def _row_id_kernel(calls, name):
    def fake(q, k, weights, starts, ends, index_topk):
        calls.append(name)
        row_ids = torch.arange(q.shape[0], dtype=torch.int32)
        return row_ids[:, None, None].expand(-1, 1, min(index_topk, k.shape[0])).contiguous()

    return fake


def _patch_kernels(monkeypatch):
    calls = []
    monkeypatch.setattr(tilelang_module, "_tilelang_dsa_topk_indices_from_ranges", _row_id_kernel(calls, "torch"))
    monkeypatch.setattr(
        tilelang_module,
        "_deep_select_dsa_topk_indices_from_ranges",
        _row_id_kernel(calls, "deep_select"),
    )
    return calls


@pytest.mark.parametrize("selector", ["torch", "deep_select"])
def test_indexer_topk_from_ranges_routes_selector(monkeypatch, selector):
    """The wrapper dispatches on its ``selector`` argument.

    Regression test: the DeepSelect feature commit referenced ``selector``
    without declaring it in this signature, so every call raised NameError
    and GLM-5.3's default KPool indexer backend could not start a training
    step.
    """
    calls = _patch_kernels(monkeypatch)

    q, k, weights, starts, ends = _ranges_inputs()
    result = tilelang_indexer_topk_from_ranges(q, k, weights, starts, ends, 4, selector=selector)

    assert calls == [selector]
    assert result.shape == (q.shape[0], 1, 4)
    assert result.dtype == torch.int32


def test_indexer_topk_from_ranges_defaults_to_torch_selector(monkeypatch):
    """Callers that predate the selector (e.g. the KPool indexer) keep the historical kernel."""
    calls = _patch_kernels(monkeypatch)

    q, k, weights, starts, ends = _ranges_inputs()
    tilelang_indexer_topk_from_ranges(q, k, weights, starts, ends, 4)

    assert calls == ["torch"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tilelang_dsa_topk_indices_forwards_deep_select(monkeypatch):
    """``selector="deep_select"`` must reach the inner wrapper.

    Regression test: the outer entry accepted a selector but dropped it when
    calling ``tilelang_indexer_topk_from_ranges``, so the ``tilelang_deepselect``
    backend silently ran the torch.topk kernel instead of DeepSelect's
    radix-select kernel.
    """
    calls = _patch_kernels(monkeypatch)

    query_len = 8
    q = torch.zeros(1, query_len, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 6, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(1, query_len, 2, device="cuda", dtype=torch.float32)
    seq_ctx = SequenceContext.from_input_ids((torch.arange(query_len, device="cuda").unsqueeze(0),), device="cuda")

    result = tilelang_module.tilelang_dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=64,
        index_topk=4,
        selector="deep_select",
    )

    assert calls == ["deep_select"]
    assert result.shape == (query_len, 1, 4)
    assert result.dtype == torch.int32
