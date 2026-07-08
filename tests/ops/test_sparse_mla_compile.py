import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import torch_dsa_topk_indices


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
