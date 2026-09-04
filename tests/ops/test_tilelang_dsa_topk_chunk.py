"""Contracts and CUDA parity tests for TileLang DSA query chunking."""

import subprocess
import sys
from functools import cache
from importlib import import_module

import pytest
import torch

import xtuner.v1.model.moe.glm52.dsa_mla as dsa_mla_module
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm52 import DSAMLAConfig
from xtuner.v1.ops.sparse_mla import dsa_topk_indices
from xtuner.v1.ops.sparse_mla.tilelang import _pad_tilelang_indexer_query_chunk, tilelang_dsa_topk_indices


tilelang_module = import_module("xtuner.v1.ops.sparse_mla.tilelang")


def test_query_tail_padding_uses_an_empty_causal_range():
    q = torch.arange(3 * 2 * 4, dtype=torch.bfloat16).reshape(3, 2, 4)
    weights = torch.ones(3, 2, dtype=torch.float32)
    starts = torch.tensor([5, 5, 5], dtype=torch.int32)
    ends = torch.tensor([6, 7, 8], dtype=torch.int32)

    padded_q, padded_weights, padded_starts, padded_ends, valid_rows = _pad_tilelang_indexer_query_chunk(
        q,
        weights,
        starts,
        ends,
        block_q=4,
    )

    assert valid_rows == 3
    assert padded_q.shape == (4, 2, 4)
    assert padded_weights.shape == (4, 2)
    torch.testing.assert_close(padded_q[:valid_rows], q)
    torch.testing.assert_close(padded_weights[:valid_rows], weights)
    assert padded_starts.tolist() == [5, 5, 5, 8]
    assert padded_ends.tolist() == [6, 7, 8, 8]
    assert padded_starts[-1] == padded_ends[-1]


def test_tilelang_selector_rejects_invalid_chunk_before_device_launch():
    q = torch.empty(1, 1, 2, 4, dtype=torch.bfloat16)
    k = torch.empty(1, 2, 4, dtype=torch.bfloat16)
    weights = torch.empty(1, 1, 2)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu")

    with pytest.raises(ValueError, match="query_chunk_size must be a positive integer"):
        tilelang_dsa_topk_indices(
            q,
            k,
            weights,
            seq_ctx,
            index_head_dim=4,
            index_topk=1,
            query_chunk_size=0,
        )


def test_config_rejects_query_chunk_for_torch_selector():
    config = DSAMLAConfig(
        num_attention_heads=2,
        head_dim=2,
        kv_lora_rank=3,
        q_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        sparse_mla_backend="torch",
        indexer_topk_query_chunk_size=2,
    )

    with pytest.raises(ValueError, match="requires a TileLang selector"):
        config.build(hidden_size=4, layer_idx=0)


def test_config_wires_query_chunk_to_tilelang_indexer(monkeypatch):
    monkeypatch.setattr(dsa_mla_module, "ensure_tilelang_runtime_available", lambda: None)
    config = DSAMLAConfig(
        num_attention_heads=2,
        head_dim=2,
        kv_lora_rank=3,
        q_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        sparse_mla_backend="tilelang",
        indexer_topk_query_chunk_size=3,
    )

    attention = config.build(hidden_size=4, layer_idx=0)

    assert attention.indexer_topk_query_chunk_size == 3
    assert attention.indexer.topk_query_chunk_size == 3


def test_indexer_forwards_query_chunk_to_selector():
    calls = {}

    def fake_selector(q, k, weights, seq_ctx, *, index_head_dim, index_topk, query_chunk_size=None):
        calls["query_chunk_size"] = query_chunk_size
        return torch.zeros((q.shape[1], 1, min(index_topk, k.shape[1])), dtype=torch.int32)

    indexer = dsa_mla_module.DSAIndexer(
        hidden_size=4,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        indexer_backend="tilelang",
        topk_query_chunk_size=3,
    )
    indexer.dsa_topk_indices_func = fake_selector
    hidden_states = torch.randn(1, 5, 4)
    q_resid = torch.randn(1, 5, 4)
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))
    seq_ctx = SequenceContext.from_input_ids((torch.arange(5).view(1, -1),), device="cpu")

    output = indexer(hidden_states, q_resid, position_embeddings, seq_ctx)

    assert calls["query_chunk_size"] == 3
    assert output["dsa_topk_ids"].dtype == torch.int32
    assert output["dsa_topk_ids"].is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_chunk_size", [None, 4, 3])
def test_query_chunk_orchestration_preserves_boundaries(monkeypatch, query_chunk_size):
    """Exercise chunk boundaries without compiling the opaque TileLang kernel."""

    calls = []

    def fake_selector(q, k, weights, starts, ends, index_topk):
        calls.append((q.shape[0], starts.detach().clone(), ends.detach().clone()))
        row_ids = q[:, 0, 0].to(torch.int32)
        return row_ids[:, None, None].expand(-1, 1, min(index_topk, k.shape[0])).contiguous()

    monkeypatch.setattr(tilelang_module, "_tilelang_dsa_topk_indices_from_ranges", fake_selector)
    query_len = 10
    q = torch.empty(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    q[0, :, 0, 0] = torch.arange(query_len, device="cuda")
    k = torch.empty(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(1, query_len, 32, device="cuda", dtype=torch.float32)
    seq_ctx = SequenceContext.from_input_ids(
        (torch.arange(4, device="cuda").unsqueeze(0), torch.arange(6, device="cuda").unsqueeze(0)),
        device="cuda",
    )

    result = tilelang_module.tilelang_dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=8,
        query_chunk_size=query_chunk_size,
    )

    expected_calls = (
        1
        if query_chunk_size is None or query_chunk_size >= query_len
        else (query_len + query_chunk_size - 1) // query_chunk_size
    )
    assert len(calls) == expected_calls
    assert result.shape == (query_len, 1, 8)
    assert result.dtype == torch.int32
    for call_idx, (rows, starts, ends) in enumerate(calls):
        assert rows % 4 == 0
        assert torch.all(starts[:-1] <= starts[1:])
        assert torch.all(ends[:-1] <= ends[1:])
        valid_rows = (
            query_len
            if query_chunk_size is None
            else min(query_chunk_size, query_len - call_idx * query_chunk_size)
        )
        if rows > valid_rows:
            # A padded tail is an empty range at the previous real end.
            assert starts[-1] == ends[-1]
            assert starts[-1] == ends[-2]
        assert torch.all(starts >= 0)
        assert torch.all(ends <= query_len)
    torch.testing.assert_close(result[:, 0, 0], torch.arange(query_len, device="cuda", dtype=torch.int32))


@cache
def _tilelang_available() -> bool:
    if not torch.cuda.is_available():
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import tilelang"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.skipif(not _tilelang_available(), reason="requires CUDA and TileLang")
def test_query_chunked_selector_matches_one_shot_on_packed_tail():
    # Keep K aligned to clean_logits_'s existing 4096-wide tile while making
    # both the packed boundary and the query chunk tail non-aligned.
    lengths = (31, 34)
    query_len = sum(lengths)
    seq_ctx = SequenceContext.from_input_ids(
        tuple(torch.arange(length, device="cuda").unsqueeze(0) for length in lengths),
        device="cuda",
    )

    torch.manual_seed(123)
    q = torch.randn(1, query_len, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8192, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, query_len, 4, device="cuda", dtype=torch.float32)

    one_shot = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=32,
        backend="tilelang",
    )
    chunked = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=32,
        query_chunk_size=17,
        backend="tilelang",
    )

    assert one_shot.dtype == torch.int32
    assert one_shot.is_contiguous()
    assert torch.equal(chunked, one_shot)

    # The same IDs must respect each packed sample's causal support.
    row_starts = torch.cat(
        [
            torch.full((length,), offset, device="cuda", dtype=torch.int32)
            for offset, length in zip((0, lengths[0]), lengths)
        ]
    )
    row_ends = row_starts + torch.cat(
        [torch.arange(length, device="cuda", dtype=torch.int32) + 1 for length in lengths]
    )
    valid = chunked[:, 0] != -1
    assert torch.all((~valid) | (chunked[:, 0] >= row_starts[:, None]))
    assert torch.all((~valid) | (chunked[:, 0] < row_ends[:, None]))
