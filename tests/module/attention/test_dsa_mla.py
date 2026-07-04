import math
import subprocess
import sys

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention import DSAMLAConfig, dsa_mla
from xtuner.v1.module.attention.dsa_mla import sparse_mla, torch_sparse_mla


BF16_ATOL = 1e-2
BF16_RTOL = 1.6e-2
DKV_ATOL = 1e-1
DKV_RTOL = 1e-1


def _tilelang_sparse_mla_available() -> bool:
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


def _tilelang_sparse_mla_inputs():
    torch.manual_seed(0)
    seq_len = 64
    topk = 64
    q = torch.randn(seq_len, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((seq_len, 1, topk), -1, device="cuda", dtype=torch.int64)
    for token_idx in range(seq_len):
        indices[token_idx, 0, : token_idx + 1] = torch.arange(token_idx + 1, device="cuda")
    return q, kv, indices


def _tiny_dsa_attention(
    indexer_types: list[str] | None = None,
    layer_idx: int = 0,
    sparse_mla_backend: str = "torch",
):
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
        sparse_mla_backend=sparse_mla_backend,
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


def test_sparse_mla_selects_torch_backend_explicitly():
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

    out, lse = sparse_mla(q, kv, indices, scaling=0.5, value_dim=4, backend="torch")
    loss = out.square().mean() + lse.mean()
    loss.backward()

    assert out.shape == (4, 2, 4)
    assert lse.shape == (4, 2)
    assert q.grad is not None
    assert kv.grad is not None


def test_sparse_mla_tilelang_backend_reports_unsupported_device_clearly():
    q = torch.randn(4, 2, 6)
    kv = torch.randn(5, 1, 6)
    indices = torch.zeros(4, 1, 4, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="requires q, kv, and indices to be CUDA tensors"):
        sparse_mla(q, kv, indices, scaling=0.5, value_dim=4, backend="tilelang")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_mla_tilelang_backend_reports_unsupported_topk_clearly():
    q = torch.randn(64, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(64, 1, 63, device="cuda", dtype=torch.int64)

    with pytest.raises(RuntimeError, match="requires topk to be divisible by 64"):
        sparse_mla(q, kv, indices, scaling=0.5, value_dim=512, backend="tilelang")


@pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and importable TileLang runtime")
def test_sparse_mla_tilelang_forward_matches_torch_sparse_mla():
    q, kv, indices = _tilelang_sparse_mla_inputs()
    scaling = 1 / math.sqrt(q.shape[-1])

    ref_out, ref_lse = sparse_mla(q, kv, indices, scaling=scaling, value_dim=512, backend="torch")
    tilelang_out, tilelang_lse = sparse_mla(q, kv, indices, scaling=scaling, value_dim=512, backend="tilelang")

    torch.testing.assert_close(tilelang_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)
    torch.testing.assert_close(tilelang_lse, ref_lse, atol=BF16_ATOL, rtol=BF16_RTOL)


@pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and importable TileLang runtime")
def test_sparse_mla_tilelang_backward_matches_torch_sparse_mla():
    q, kv, indices = _tilelang_sparse_mla_inputs()
    scaling = 1 / math.sqrt(q.shape[-1])
    q_ref = q.detach().clone().requires_grad_()
    kv_ref = kv.detach().clone().requires_grad_()
    q_tilelang = q.detach().clone().requires_grad_()
    kv_tilelang = kv.detach().clone().requires_grad_()

    ref_out, _ = sparse_mla(q_ref, kv_ref, indices, scaling=scaling, value_dim=512, backend="torch")
    tilelang_out, _ = sparse_mla(q_tilelang, kv_tilelang, indices, scaling=scaling, value_dim=512, backend="tilelang")
    grad_output = torch.randn_like(ref_out)

    ref_out.backward(grad_output)
    tilelang_out.backward(grad_output)

    torch.testing.assert_close(q_tilelang.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
    # dKV accumulates duplicate sparse indices with fp32 atomics before casting to bf16;
    # PyTorch fallback uses a deterministic reduction order, so its bf16 rounding differs.
    torch.testing.assert_close(kv_tilelang.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


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


def test_dsa_attention_selects_tilelang_sparse_mla_backend_explicitly():
    torch.manual_seed(0)
    attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0, sparse_mla_backend="tilelang")
    hidden_states = torch.randn(1, 5, 4)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4, 5]]),), device="cpu")
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    with pytest.raises(RuntimeError, match="TileLang"):
        attn(hidden_states, position_embeddings, seq_ctx)


def test_dsa_attention_tilelang_runtime_checked_once_at_build(monkeypatch):
    calls = 0

    def fake_ensure_tilelang_runtime_available():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(dsa_mla, "_ensure_tilelang_runtime_available", fake_ensure_tilelang_runtime_available)
    _tiny_dsa_attention(indexer_types=["full"], layer_idx=0, sparse_mla_backend="tilelang")

    q = torch.randn(4, 2, 6)
    kv = torch.randn(5, 1, 6)
    indices = torch.zeros(4, 1, 4, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="requires q, kv, and indices to be CUDA tensors"):
        sparse_mla(q, kv, indices, scaling=0.5, value_dim=4, backend="tilelang")

    assert calls == 1


@pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and importable TileLang runtime")
def test_dsa_attention_tilelang_long_packed_sequence_respects_boundaries_and_backward():
    torch.manual_seed(0)
    seq_lens = [128, 160, 224]
    seq_len = sum(seq_lens)
    attn = (
        DSAMLAConfig(
            num_attention_heads=16,
            head_dim=64,
            kv_lora_rank=512,
            q_lora_rank=64,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            index_topk=64,
            index_head_dim=128,
            index_n_heads=4,
            indexer_types=["full"],
            sparse_mla_backend="tilelang",
        )
        .build(hidden_size=64, layer_idx=0)
        .cuda()
        .bfloat16()
    )
    input_ids = []
    offset = 0
    for length in seq_lens:
        input_ids.append(torch.arange(offset, offset + length).view(1, -1))
        offset += length
    hidden_states = torch.randn(1, seq_len, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    position_embeddings = (
        torch.ones(1, seq_len, 64, device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, seq_len, 64, device="cuda", dtype=torch.bfloat16),
    )
    seq_ctx = SequenceContext.from_input_ids(input_ids, device="cuda")

    attn_outputs = attn(hidden_states, position_embeddings, seq_ctx)
    loss = attn_outputs["projected_output"].float().square().mean()
    loss.backward()

    assert torch.isfinite(attn_outputs["projected_output"]).all()
    assert torch.isfinite(attn_outputs["raw_output"]).all()
    assert torch.isfinite(attn_outputs["softmax_lse"]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()

    assert seq_ctx.dsa_topk_indices is not None
    topk = seq_ctx.dsa_topk_indices[0]
    for seq_start, seq_end in zip(seq_ctx.cu_seq_lens_q[:-1].tolist(), seq_ctx.cu_seq_lens_q[1:].tolist()):
        for token_idx in range(seq_start, seq_end):
            valid_indices = topk[token_idx, 0][topk[token_idx, 0] != -1]
            assert valid_indices.numel() == min(64, token_idx - seq_start + 1)
            assert valid_indices.min().item() >= seq_start
            assert valid_indices.max().item() <= token_idx


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


def test_dsa_attention_mtp_layer_reuses_last_main_full_indexer():
    torch.manual_seed(0)
    mtp_attn = _tiny_dsa_attention(indexer_types=["full", "full", "full", "shared", "shared"], layer_idx=5)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4, 5]]),), device="cpu")
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))
    source_topk = torch.full((5, 1, 4), -1, dtype=torch.int64)
    for token_idx in range(5):
        valid = torch.arange(token_idx + 1)[:4]
        source_topk[token_idx, 0, : valid.numel()] = valid
    seq_ctx.dsa_topk_indices = {2: source_topk}

    attn_outputs = mtp_attn(torch.randn(1, 5, 4), position_embeddings, seq_ctx)

    assert torch.isfinite(attn_outputs["projected_output"]).all()
    assert set(seq_ctx.dsa_topk_indices) == {2}
    assert seq_ctx.dsa_topk_indices[2] is source_topk


def test_dsa_attention_shared_layer_fails_when_source_topk_is_missing():
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    with pytest.raises(AssertionError, match="Cross-pipeline top-k sharing is not supported"):
        shared_attn(torch.randn(1, 4, 4), position_embeddings, seq_ctx)
