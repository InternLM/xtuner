"""NVIDIA CuTe DSL fused DSA indexer public behavior tests."""

import importlib.util

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import dsa_topk_indices, get_dsa_topk_indices


def _has_cute_dsl_sm90() -> bool:
    if importlib.util.find_spec("cutlass") is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (9, 0)


pytestmark = pytest.mark.skipif(
    not _has_cute_dsl_sm90(),
    reason="requires nvidia-cutlass-dsl and an SM90 GPU",
)


@pytest.mark.parametrize("compiled", [False, True])
def test_cute_dsl_indexer_matches_torch_for_packed_causal_ranges(compiled: bool):
    # The local shard crosses a packed-sequence boundary and exactly fills two
    # four-query CTAs. This covers both long and heavily padded causal rows.
    torch.manual_seed(31)
    query_len, kv_len, index_topk = 8, 2048, 1024
    q = torch.randn(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, query_len, 32, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1024, kv_len], device="cuda", dtype=torch.int32)
    seq_ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=offsets,
        cu_seq_lens_k=offsets,
        max_length_q=1024,
        max_length_k=1024,
        shard_start=1020,
        shard_size=query_len,
        device="cuda",
    )

    expected = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="torch",
    )
    indexer = get_dsa_topk_indices("cute_dsl")
    if compiled:
        indexer = torch.compile(indexer, fullgraph=True)
    actual = indexer(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
    )

    torch.testing.assert_close(
        actual.sort(dim=-1).values,
        expected.sort(dim=-1).values,
    )
    assert actual.shape == (query_len, 1, index_topk)
    assert actual.dtype == torch.int32


@pytest.mark.parametrize(
    ("kv_len", "index_topk"),
    [
        (32768, 1024),
        (65536, 2048),
    ],
)
def test_cute_dsl_multi_merge_specialization_handles_short_ranges(kv_len: int, index_topk: int):
    # The static K shape selects a multi-merge specialization, while every
    # runtime causal range is shorter than topk and must remain padded.
    torch.manual_seed(33)
    query_len = 5
    q = torch.randn(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, query_len, 32, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 512, kv_len], device="cuda", dtype=torch.int32)
    seq_ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=offsets,
        cu_seq_lens_k=offsets,
        max_length_q=kv_len - 512,
        max_length_k=kv_len - 512,
        shard_start=509,
        shard_size=query_len,
        device="cuda",
    )

    expected = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="torch",
    )
    actual = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="cute_dsl",
    )

    torch.testing.assert_close(
        actual.sort(dim=-1).values,
        expected.sort(dim=-1).values,
    )


@pytest.mark.parametrize(
    ("kv_len", "index_topk"),
    [
        (8192, 1024),
        (10240, 1024),
        (12288, 1024),
        (15360, 1024),
        (8192, 2048),
        (32768, 1024),
        (32768, 2048),
        (65536, 1024),
        (65535, 2048),
        (65536, 2048),
        (65537, 2048),
    ],
)
def test_cute_dsl_indexer_matches_torch_for_production_topk(kv_len: int, index_topk: int):
    torch.manual_seed(37)
    query_len = 5
    q = torch.randn(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, query_len, 32, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    seq_ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=offsets,
        cu_seq_lens_k=offsets,
        max_length_q=kv_len,
        max_length_k=kv_len,
        shard_start=kv_len - query_len,
        shard_size=query_len,
        device="cuda",
    )

    expected = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="torch",
    )
    # Repeated launches expose shared-K reuse races; the 65,535 boundary also
    # exercises the partial-tile fallback around warp candidate collection.
    repeat_count = 200 if kv_len == 32768 or (index_topk == 2048 and kv_len in (65535, 65536, 65537)) else 1
    for _ in range(repeat_count):
        actual = dsa_topk_indices(
            q,
            k,
            weights,
            seq_ctx,
            index_head_dim=128,
            index_topk=index_topk,
            backend="cute_dsl",
        )
        torch.testing.assert_close(
            actual.sort(dim=-1).values,
            expected.sort(dim=-1).values,
        )


def test_cute_dsl_relative_candidate_ids_handle_nonzero_k_start():
    # The static specialization uses uint16 IDs, while every query belongs to
    # the second packed sequence and therefore has a nonzero relative base.
    torch.manual_seed(41)
    query_len, kv_len, index_topk = 5, 65536, 2048
    q = torch.randn(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, query_len, 32, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 32768, kv_len], device="cuda", dtype=torch.int32)
    seq_ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=offsets,
        cu_seq_lens_k=offsets,
        max_length_q=32768,
        max_length_k=32768,
        shard_start=kv_len - query_len,
        shard_size=query_len,
        device="cuda",
    )

    expected = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="torch",
    )
    actual = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="cute_dsl",
    )

    torch.testing.assert_close(
        actual.sort(dim=-1).values,
        expected.sort(dim=-1).values,
    )


@pytest.mark.parametrize("index_topk", [1024, 2048])
def test_cute_dsl_online_compaction_handles_full_candidate_batches(index_topk: int):
    # Scores increase by one per key. Every tile after the first selection
    # therefore passes the online threshold and exercises the candidate-
    # capacity merge guard without an ambiguous tie at the causal boundary.
    query_len, kv_len = 5, 65536
    q = torch.zeros(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    q[:, :, 0, 0] = 32768
    q[:, :, 0, 1] = 128
    q[:, :, 0, 2] = 1
    k = torch.zeros(1, kv_len, 128, device="cuda", dtype=torch.bfloat16)
    key_indices = torch.arange(kv_len, device="cuda", dtype=torch.int32)
    blocks = key_indices // 128
    k[0, :, 0] = (blocks // 256).to(torch.bfloat16)
    k[0, :, 1] = (blocks % 256).to(torch.bfloat16)
    k[0, :, 2] = (key_indices % 128).to(torch.bfloat16)
    weights = torch.zeros(1, query_len, 32, device="cuda", dtype=torch.float32)
    weights[:, :, 0] = 1
    offsets = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    seq_ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=offsets,
        cu_seq_lens_k=offsets,
        max_length_q=kv_len,
        max_length_k=kv_len,
        shard_start=kv_len - query_len,
        shard_size=query_len,
        device="cuda",
    )

    actual = dsa_topk_indices(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=128,
        index_topk=index_topk,
        backend="cute_dsl",
    )
    expected = torch.stack(
        [
            torch.arange(
                end - index_topk,
                end,
                device="cuda",
                dtype=torch.int32,
            )
            for end in range(kv_len - query_len + 1, kv_len + 1)
        ]
    ).unsqueeze(1)

    torch.testing.assert_close(actual.sort(dim=-1).values, expected)
