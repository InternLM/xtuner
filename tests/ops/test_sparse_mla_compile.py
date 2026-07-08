import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import torch_dsa_topk_indices, torch_sparse_mla


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSA compile tests require CUDA")
def test_torch_dsa_topk_indices_supports_fullgraph_compile():
    seq_len = 6
    q = torch.randn(1, seq_len, 2, 4, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, seq_len, 4, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, seq_len, 2, device="cuda", dtype=torch.float32)
    input_ids = torch.arange(seq_len, device="cuda").view(1, -1)
    seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))

    eager = torch_dsa_topk_indices(q, k, weights, seq_ctx, index_head_dim=4, index_topk=4)
    compiled = torch.compile(torch_dsa_topk_indices, fullgraph=True)

    actual = compiled(q, k, weights, seq_ctx, index_head_dim=4, index_topk=4)

    torch.testing.assert_close(actual, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SparseMLA compile tests require CUDA")
def test_torch_sparse_mla_supports_fullgraph_compile_with_padded_indices():
    q = torch.randn(6, 2, 4, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(6, 1, 4, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor(
        [
            [[0, -1, -1, -1]],
            [[1, 0, -1, -1]],
            [[2, 1, 0, -1]],
            [[3, 2, 1, 0]],
            [[4, 3, 2, 1]],
            [[5, 4, 3, 2]],
        ],
        device="cuda",
    )

    eager = torch_sparse_mla(q, kv, indices, scaling=None)
    compiled = torch.compile(torch_sparse_mla, fullgraph=True)

    actual = compiled(q, kv, indices, scaling=None)

    torch.testing.assert_close(actual.raw_output, eager.raw_output)
    torch.testing.assert_close(actual.softmax_lse, eager.softmax_lse)
