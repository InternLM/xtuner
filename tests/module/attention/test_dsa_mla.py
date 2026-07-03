import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.attention.dsa_mla import torch_sparse_mla


def _tiny_dsa_attention(indexer_types: list[str] | None = None, layer_idx: int = 0):
    return DSAMLAConfig(
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
        indexer_types=indexer_types,
    ).build(hidden_size=4, layer_idx=layer_idx)


def test_torch_sparse_mla_handles_invalid_indices_and_backward():
    torch.manual_seed(0)
    q = torch.randn(4, 2, 6, requires_grad=True)
    kv = torch.randn(5, 1, 6, requires_grad=True)
    indices = torch.tensor(
        [
            [[0, -1, -1]],
            [[1, 0, -1]],
            [[2, 1, -1]],
            [[4, 3, 1]],
        ],
        dtype=torch.int64,
    )

    out, lse = torch_sparse_mla(q, kv, indices, scaling=0.5, value_dim=4)
    loss = out.square().mean() + lse.mean()
    loss.backward()

    assert out.shape == (4, 2, 4)
    assert lse.shape == (4, 2)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    assert q.grad is not None
    assert kv.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(kv.grad).all()


def test_dsa_attention_topk_respects_packed_causal_boundaries():
    torch.manual_seed(0)
    attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0)
    hidden_states = torch.randn(1, 5, 4)
    seq_ctx = SequenceContext.from_input_ids(
        (torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])),
        device="cpu",
    )
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    attn(hidden_states, position_embeddings, seq_ctx)

    topk = seq_ctx.dsa_topk_indices[0]
    for token_idx, seq_start in [(0, 0), (1, 0), (2, 2), (3, 2), (4, 2)]:
        valid_indices = topk[token_idx, 0][topk[token_idx, 0] != -1]
        assert valid_indices.numel() == token_idx - seq_start + 1
        assert valid_indices.min().item() >= seq_start
        assert valid_indices.max().item() <= token_idx


def test_dsa_attention_forward_backward_on_packed_inputs():
    torch.manual_seed(0)
    attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0)
    hidden_states = torch.randn(1, 5, 4, requires_grad=True)
    seq_ctx = SequenceContext.from_input_ids(
        (torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])),
        device="cpu",
    )
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    attn_outputs = attn(hidden_states, position_embeddings, seq_ctx)
    loss = attn_outputs["projected_output"].square().mean()
    loss.backward()

    assert attn_outputs["projected_output"].shape == (1, 5, 4)
    assert attn_outputs["raw_output"].shape == (1, 5, 6)
    assert torch.isfinite(attn_outputs["projected_output"]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_dsa_attention_shares_topk_within_microbatch_only():
    torch.manual_seed(0)
    full_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0)
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    hidden_states = torch.randn(1, 4, 4)
    full_attn(hidden_states, position_embeddings, seq_ctx)
    source_topk = seq_ctx.dsa_topk_indices[0]

    shared_attn(hidden_states, position_embeddings, seq_ctx)

    assert set(seq_ctx.dsa_topk_indices) == {0}
    assert seq_ctx.dsa_topk_indices[0] is source_topk

    other_seq_ctx = SequenceContext.from_input_ids((torch.tensor([[5, 6, 7, 8]]),), device="cpu")
    full_attn(torch.randn(1, 4, 4), position_embeddings, other_seq_ctx)

    assert other_seq_ctx.dsa_topk_indices is not seq_ctx.dsa_topk_indices
    assert set(other_seq_ctx.dsa_topk_indices) == {0}


def test_dsa_attention_shared_layer_fails_when_source_topk_is_missing():
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    with pytest.raises(AssertionError, match="Cross-pipeline top-k sharing is not supported"):
        shared_attn(torch.randn(1, 4, 4), position_embeddings, seq_ctx)
