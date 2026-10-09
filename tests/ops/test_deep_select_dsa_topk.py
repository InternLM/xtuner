"""Parity tests for the DeepSelect-backed DSA indexer selector."""

import importlib.util

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import dsa_topk_indices


requires_deep_select = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("deep_select") is None
    or importlib.util.find_spec("tilelang") is None,
    reason="requires CUDA, TileLang and DeepSelect",
)


@pytest.mark.gpu
@requires_deep_select
class TestDeepSelectDSATopK:
    @pytest.mark.parametrize("query_chunk_size", [None, 1000])
    def test_matches_torch_topk_selector(self, query_chunk_size: int | None):
        # The first two samples are shorter than topk (every row keeps its
        # full causal range); the last sample has rows on both sides of topk.
        lengths = (100, 700, 3300)
        index_topk = 512
        query_len = sum(lengths)
        seq_ctx = SequenceContext.from_input_ids(
            tuple(torch.arange(length, device="cuda").unsqueeze(0) for length in lengths),
            device="cuda",
        )
        torch.manual_seed(0)
        q = torch.randn(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
        # K stays aligned to the TileLang clean_logits tile, as in training.
        k = torch.randn(1, 8192, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(1, query_len, 32, device="cuda", dtype=torch.float32)

        kwargs = {"index_head_dim": 128, "index_topk": index_topk, "query_chunk_size": query_chunk_size}
        expected = dsa_topk_indices(q, k, weights, seq_ctx, backend="tilelang", **kwargs)
        actual = dsa_topk_indices(q, k, weights, seq_ctx, backend="tilelang_deepselect", **kwargs)

        assert actual.shape == expected.shape == (query_len, 1, index_topk)
        assert actual.dtype == torch.int32
        assert actual.is_contiguous()
        # Both selectors return the same ID set per row; only the order differs.
        assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
        # cuDNN DSA derives topk_length from the -1 count, so -1 must be a tail.
        valid = actual[:, 0] != -1
        topk_length = valid.sum(dim=-1, keepdim=True)
        assert torch.equal(valid, torch.arange(index_topk, device="cuda") < topk_length)
