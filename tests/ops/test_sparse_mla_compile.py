"""PyTorch DSA 算子的 fullgraph compile 行为测试。

TestSparseMLACompile
    test_topk_indices_matches_eager: DSA top-k 编译结果与 eager 一致。
    test_sparse_mla_with_padded_indices_matches_eager: 含 padding 索引的 SparseMLA 编译结果与 eager 一致。
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import torch_dsa_topk_indices, torch_sparse_mla


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSparseMLACompile:
    def test_topk_indices_matches_eager(self):
        # 验证 DSA top-k public API 可 fullgraph 编译且数值与 eager 一致。
        seq_len = 6
        q = torch.randn(1, seq_len, 2, 4, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, seq_len, 4, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(1, seq_len, 2, device="cuda", dtype=torch.float32)
        input_ids = torch.arange(seq_len, device="cuda").view(1, -1)
        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))

        eager = torch_dsa_topk_indices(q, k, weights, seq_ctx, index_head_dim=4, index_topk=4)
        compiled = torch.compile(torch_dsa_topk_indices, fullgraph=True)

        torch.testing.assert_close(
            compiled(q, k, weights, seq_ctx, index_head_dim=4, index_topk=4),
            eager,
        )

    def test_sparse_mla_with_padded_indices_matches_eager(self):
        # 验证含 -1 padding 的 SparseMLA public API 可 fullgraph 编译且两个输出均与 eager 一致。
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
