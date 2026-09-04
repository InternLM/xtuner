# Copyright (c) OpenMMLab. All rights reserved.
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import dense_dsa_indexer_kl_loss


def test_dense_dsa_indexer_kl_is_finite_and_only_updates_indexer():
    torch.manual_seed(7)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    index_q = torch.randn(1, 4, 2, 4, requires_grad=True)
    index_k = torch.randn(1, 4, 4, requires_grad=True)
    index_weights = torch.randn(1, 4, 2, requires_grad=True)
    teacher_q = torch.randn(1, 4, 3, 6, requires_grad=True)
    teacher_k = torch.randn(1, 4, 3, 6, requires_grad=True)

    loss = dense_dsa_indexer_kl_loss(
        index_q,
        index_k,
        index_weights,
        teacher_q,
        teacher_k,
        seq_ctx,
        softmax_scale=0.25,
        query_block_size=2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (index_q, index_k, index_weights)
    )
    assert teacher_q.grad is None
    assert teacher_k.grad is None
