# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import get_kpool_topk_indices
from xtuner.v1.ops.sparse_mla.cooperative_kpool import cooperative_kpool_topk


@pytest.mark.gpu
@pytest.mark.parametrize(
    "lengths,shard_start,query_len,topk",
    [
        ([3, 7, 131, 1098], 0, 1239, 64),
        ([131, 1098], 128, 1029, 64),
        ([3, 7], 3, 7, 2048),
    ],
)
def test_packed_sharded_selection(lengths, shard_start, query_len, topk):
    torch.manual_seed(5)
    n = sum(lengths)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    ctx = SequenceContext(
        input_ids=torch.zeros(1, query_len, dtype=torch.long, device="cuda"),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        device="cuda",
        shard_start=shard_start,
        shard_size=query_len,
    )
    q = torch.randn(1, query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, n, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.randn_like(k)
    weights = torch.randn(1, query_len, 32, device="cuda")
    ape = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(index_head_dim=128, index_topk=topk, index_kpool=4, alignment=64, query_chunk_size=1024)
    original = get_kpool_topk_indices("tilelang")(q, k, gates, weights, ape, ctx, **kwargs)
    actual = get_kpool_topk_indices("tilelang_cooperative")(q, k, gates, weights, ape, ctx, **kwargs)
    assert actual.shape == original.shape
    torch.testing.assert_close(actual.sort(dim=-1).values, original.sort(dim=-1).values, atol=0, rtol=0)
    boundaries = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
    for row in range(query_len):
        pos = shard_start + row
        begin = max(x for x in boundaries[:-1] if x <= pos)
        valid = actual[row][actual[row] >= 0]
        assert bool(((valid >= begin) & (valid <= pos)).all())


@pytest.mark.gpu
def test_tied_scores_empty_ranges_and_compile():
    q = torch.zeros(1031, 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(258, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.zeros(1031, 32, device="cuda")
    cu = torch.tensor([0, 1031], dtype=torch.int32, device="cuda")
    compiled = torch.compile(cooperative_kpool_topk, fullgraph=True)
    out = compiled(q, k, weights, cu, 0, 4, 16, 1024)
    for row, ids in enumerate(out[:, 0].cpu()):
        valid = ids[ids >= 0]
        assert len(valid) == min(16, (row + 1) // 4)
        assert len(valid.unique()) == len(valid)
        assert bool((valid < (row + 1) // 4).all())
