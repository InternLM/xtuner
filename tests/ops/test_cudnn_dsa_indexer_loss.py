"""Correctness tests for the cuDNN DSA sparse indexer training loss."""

import importlib
import subprocess
import sys
from functools import cache

import pytest
import torch

from xtuner.v1.ops.sparse_mla.cudnn_dsa_indexer_loss import (
    _INDEXER_LOSS_DEBUG_CALLS,
    _copy_aligned_grad_loss,
    _mask_invalid_query_rows,
    _maybe_log_indexer_loss_diagnostics,
    _pad_indexer_heads_for_cudnn,
    _standard_kl_loss,
    _xtuner_indexer_backward,
    dsa_indexer_kl_from_distribution,
    dsa_indexer_kl_loss,
    sparse_attention_target,
    sparse_indexer_predict,
)


@cache
def _cudnn_indexer_training_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cudnn.deepseek_sparse_attention.indexer_backward import indexer_backward_wrapper; "
            "from cudnn.deepseek_sparse_attention.score_recompute import "
            "sparse_attn_score_recompute_wrapper, sparse_indexer_score_recompute_wrapper",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _packed_topk_indices(seq_lens: tuple[int, ...], topk: int, device: str) -> torch.Tensor:
    seq_len = sum(seq_lens)
    indices = torch.full((1, seq_len, topk), -1, dtype=torch.int32, device=device)
    row = 0
    for seq_len_i in seq_lens:
        for offset in range(seq_len_i):
            valid = min(offset + 1, topk)
            indices[0, row, :valid] = torch.arange(
                row + 1 - valid,
                row + 1,
                dtype=torch.int32,
                device=device,
            )
            row += 1
    return indices


def _gather_selected_k(k: torch.Tensor, topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert k.shape[0] == 1, "The test oracle only needs the packed batch-size-one layout."
    valid = topk_indices != -1
    selected = k[:, topk_indices.clamp_min(0)[0].long(), :]
    return selected, valid


def _indexer_predict_oracle(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    selected_k, valid = _gather_selected_k(index_k, topk_indices)
    scores = torch.einsum("bshd,bskd->bshk", index_q.float(), selected_k.float()).relu()
    logits = torch.einsum("bshk,bsh->bsk", scores, weights.float())
    logits = logits.masked_fill(~valid, float("-inf"))
    return torch.softmax(logits, dim=-1).masked_fill(~valid, 0.0)


def _attention_target_oracle(
    attn_q: torch.Tensor,
    attn_k: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_k, valid = _gather_selected_k(attn_k, topk_indices)
    scores = torch.einsum("bshd,bskd->bshk", attn_q.float(), selected_k.float()) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).masked_fill(~valid[:, :, None, :], 0.0)
    target = probs.sum(dim=2)
    target = target / target.sum(dim=-1, keepdim=True)
    return target.masked_fill(~valid, 0.0), lse


def _explicit_standard_kl(
    target: torch.Tensor,
    predict: torch.Tensor,
    topk_indices: torch.Tensor,
    row_coefficient: float,
) -> torch.Tensor:
    valid_rows = (topk_indices != -1).any(dim=-1)
    target_xlogx = torch.special.xlogy(target, target).sum(dim=-1)
    cross_entropy = torch.special.xlogy(target, predict).sum(dim=-1)
    return row_coefficient * (target_xlogx - cross_entropy).masked_fill(~valid_rows, 0.0).sum()


class TestStandardKLLoss:
    @staticmethod
    def _indexer_logits(q, k, weights):
        scores = torch.einsum("bshd,btd->bsht", q, k).relu()
        return torch.einsum("bsht,bsh->bst", scores, weights)

    def test_zero_head_padding_preserves_scores_and_original_gradients(self):
        torch.manual_seed(5)
        q_data = torch.randn(1, 3, 32, 8)
        k_data = torch.randn(1, 4, 8)
        w_data = torch.randn(1, 3, 32)

        reference_q = q_data.clone().requires_grad_()
        reference_k = k_data.clone().requires_grad_()
        reference_w = w_data.clone().requires_grad_()
        reference_logits = self._indexer_logits(reference_q, reference_k, reference_w)
        reference_logits.square().sum().backward()

        actual_q = q_data.clone().requires_grad_()
        actual_k = k_data.clone().requires_grad_()
        actual_w = w_data.clone().requires_grad_()
        padded_q, padded_w, original_heads = _pad_indexer_heads_for_cudnn(actual_q, actual_w)
        actual_logits = self._indexer_logits(padded_q, actual_k, padded_w)
        actual_logits.square().sum().backward()

        assert original_heads == 32
        assert padded_q.shape[-2] == 64
        assert padded_w.shape[-1] == 64
        torch.testing.assert_close(actual_logits, reference_logits)
        torch.testing.assert_close(actual_q.grad, reference_q.grad)
        torch.testing.assert_close(actual_k.grad, reference_k.grad)
        torch.testing.assert_close(actual_w.grad, reference_w.grad)

    def test_grad_loss_copy_realigns_contiguous_storage_offset_view(self):
        storage = torch.tensor([0.0, 0.37], dtype=torch.float32)
        misaligned_grad_loss = storage[1:]
        assert misaligned_grad_loss.is_contiguous()
        assert misaligned_grad_loss.data_ptr() % 16 != 0

        aligned_grad_loss = _copy_aligned_grad_loss(misaligned_grad_loss, torch.device("cpu"))

        assert aligned_grad_loss.shape == (1,)
        assert aligned_grad_loss.dtype == torch.float32
        assert aligned_grad_loss.data_ptr() % 16 == 0
        torch.testing.assert_close(aligned_grad_loss, misaligned_grad_loss)

    def test_backward_adapter_satisfies_cudnn_shape_alignment_and_mutation_contract(self, monkeypatch):
        import cudnn.deepseek_sparse_attention.indexer_backward as cudnn_indexer_backward

        seen = {}

        def fake_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            target,
            predict,
            topk_indices,
            **kwargs,
        ):
            tensors = (index_q, weights, index_k, target, predict, topk_indices, kwargs["grad_loss"])
            assert all(tensor.is_contiguous() for tensor in tensors)
            assert all(tensor.data_ptr() % 16 == 0 for tensor in tensors)
            assert index_q.shape[-2] == 64
            assert weights.shape[-1] == 64
            assert topk_indices.shape[-1] == 128
            assert topk_indices.dtype == torch.int32
            seen["loss_coeff"] = kwargs["loss_coeff"]
            target.zero_()
            predict.zero_()
            return {
                "d_index_q": torch.ones_like(index_q),
                "d_index_k": torch.ones_like(index_k),
                "d_weights": torch.ones_like(weights),
            }

        monkeypatch.setattr(cudnn_indexer_backward, "indexer_backward_wrapper", fake_indexer_backward_wrapper)

        index_q = torch.randn(1, 2, 32, 8)
        index_k = torch.randn(1, 4, 8)
        index_weights = torch.randn(1, 2, 32)
        target = torch.rand(1, 2, 128)
        predict = torch.rand(1, 2, 128)
        target_before = target.clone()
        predict_before = predict.clone()
        safe_topk_indices = torch.zeros(1, 2, 128, dtype=torch.int32)
        grad_storage = torch.tensor([0.0, 0.25], dtype=torch.float32)

        d_q, d_k, d_w = _xtuner_indexer_backward(
            index_q,
            index_k,
            index_weights,
            target,
            predict,
            safe_topk_indices,
            row_coefficient=0.125,
            grad_loss=grad_storage[1],
        )

        assert seen["loss_coeff"] == 0.25
        assert d_q.shape == index_q.shape
        assert d_k.shape == index_k.shape
        assert d_w.shape == index_weights.shape
        torch.testing.assert_close(target, target_before)
        torch.testing.assert_close(predict, predict_before)

    def test_matches_explicit_standard_kl_with_zero_probability_slots(self):
        target = torch.tensor([[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
        predict = torch.tensor([[[0.6, 0.4, 0.0], [0.8, 0.2, 0.0]]], dtype=torch.float32)
        topk_indices = torch.tensor([[[0, 1, -1], [0, -1, -1]]], dtype=torch.int32)

        actual = _standard_kl_loss(target, predict, topk_indices, row_coefficient=0.25)
        expected = _explicit_standard_kl(target, predict, topk_indices, row_coefficient=0.25)

        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual)

    def test_identical_distributions_have_zero_loss(self):
        distribution = torch.tensor([[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
        topk_indices = torch.tensor([[[0, 1, -1], [0, -1, -1]]], dtype=torch.int32)

        loss = _standard_kl_loss(distribution, distribution, topk_indices, row_coefficient=0.5)

        torch.testing.assert_close(loss, torch.tensor(0.0), atol=0.0, rtol=0.0)

    def test_positive_target_with_zero_predict_is_rejected(self):
        target = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        predict = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        topk_indices = torch.tensor([[[0, 1]]], dtype=torch.int32)

        with pytest.raises((AssertionError, RuntimeError), match="zero probability"):
            _standard_kl_loss(target, predict, topk_indices, row_coefficient=1.0)

    def test_query_mask_excludes_causal_padding_chunks(self):
        target = torch.tensor(
            [[[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]],
            dtype=torch.float32,
        )
        predict = torch.tensor(
            [[[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]],
            dtype=torch.float32,
        )
        # Padding chunks are causal sequences, so their top-k rows can contain
        # valid-looking indices even though they must not contribute to loss.
        topk_indices = torch.tensor([[[0, 1], [2, 3], [4, 5]]], dtype=torch.int32)
        valid_query_mask = torch.tensor([[True, False, False]])

        masked_target, masked_predict, masked_topk = _mask_invalid_query_rows(
            target,
            predict,
            topk_indices,
            valid_query_mask,
        )
        actual = _standard_kl_loss(masked_target, masked_predict, masked_topk, row_coefficient=1.0)
        expected = _explicit_standard_kl(
            target[:, :1],
            predict[:, :1],
            topk_indices[:, :1],
            row_coefficient=1.0,
        )

        torch.testing.assert_close(actual, expected)
        assert torch.count_nonzero(masked_target[:, 1:]) == 0
        assert torch.count_nonzero(masked_predict[:, 1:]) == 0
        assert torch.all(masked_topk[:, 1:] == -1)

    def test_query_mask_shape_is_validated(self):
        target = torch.ones(1, 2, 1)
        predict = torch.ones_like(target)
        topk_indices = torch.zeros(1, 2, 1, dtype=torch.int32)

        with pytest.raises(ValueError, match="valid_query_mask must have shape"):
            _mask_invalid_query_rows(target, predict, topk_indices, torch.ones(2, dtype=torch.bool))

    def test_distribution_diagnostics_are_named_and_interval_gated(self, monkeypatch):
        target = torch.tensor([[[0.8, 0.2], [0.0, 0.0]]], dtype=torch.float32)
        predict = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]], dtype=torch.float32)
        topk_indices = torch.tensor([[[0, 1], [-1, -1]]], dtype=torch.int32)
        messages = []
        loss_module = importlib.import_module("xtuner.v1.ops.sparse_mla.cudnn_dsa_indexer_loss")
        monkeypatch.setattr(loss_module.log_rank0, "info", messages.append)
        _INDEXER_LOSS_DEBUG_CALLS.clear()

        for _ in range(3):
            _maybe_log_indexer_loss_diagnostics(
                target,
                predict,
                topk_indices,
                debug_name="layer2",
                debug_interval=3,
            )

        assert len(messages) == 2
        assert "name=layer2 call=1 valid_rows=1" in messages[0]
        assert "call=3" in messages[1]
        assert "kl_mean=" in messages[0]
        assert "target_entropy=" in messages[0]
        assert "top1_match=" in messages[0]


@pytest.mark.skipif(not _cudnn_indexer_training_available(), reason="requires CUDA SM90+ and cuDNN DSA")
class TestCudnnDSAIndexerLoss:
    def test_score_recompute_matches_pytorch_oracles_for_packed_padding(self):
        torch.manual_seed(7)
        device = "cuda"
        seq_len = 128
        topk_indices = _packed_topk_indices((64, 64), topk=128, device=device)
        topk_length = (topk_indices != -1).sum(dim=-1, dtype=torch.int32)

        index_q = torch.randn(1, seq_len, 32, 128, dtype=torch.bfloat16, device=device)
        index_k = torch.randn(1, seq_len, 128, dtype=torch.bfloat16, device=device)
        projected_weights = torch.randn(1, seq_len, 32, dtype=torch.bfloat16, device=device)
        index_weights = projected_weights * ((32 * 128) ** -0.5)

        attn_q = torch.randn(1, seq_len, 64, 576, dtype=torch.bfloat16, device=device)
        attn_k = torch.randn(1, seq_len, 576, dtype=torch.bfloat16, device=device)
        softmax_scale = 576**-0.5
        expected_target, attn_lse = _attention_target_oracle(attn_q, attn_k, topk_indices, softmax_scale)
        expected_predict = _indexer_predict_oracle(index_q, index_k, index_weights, topk_indices)

        actual_target = sparse_attention_target(
            attn_q,
            attn_k,
            attn_lse,
            topk_indices,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
        )
        actual_predict = sparse_indexer_predict(
            index_q,
            index_k,
            index_weights,
            topk_indices,
            topk_length=topk_length,
        )

        torch.testing.assert_close(actual_target, expected_target, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_predict, expected_predict, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_target.sum(dim=-1), torch.ones_like(actual_target[..., 0]))
        torch.testing.assert_close(actual_predict.sum(dim=-1), torch.ones_like(actual_predict[..., 0]))
        assert torch.count_nonzero(actual_target.masked_select(topk_indices == -1)) == 0
        assert torch.count_nonzero(actual_predict.masked_select(topk_indices == -1)) == 0

    def test_distribution_loss_and_gradients_match_pytorch_with_upstream_scale(self):
        torch.manual_seed(11)
        device = "cuda"
        seq_len = 128
        topk_indices = _packed_topk_indices((128,), topk=128, device=device)
        topk_length = (topk_indices != -1).sum(dim=-1, dtype=torch.int32)
        row_coefficient = 0.017
        upstream_scale = 0.37

        q_data = torch.randn(1, seq_len, 32, 128, dtype=torch.bfloat16, device=device)
        k_data = torch.randn(1, seq_len, 128, dtype=torch.bfloat16, device=device)
        w_data = torch.randn(1, seq_len, 32, dtype=torch.bfloat16, device=device) * ((32 * 128) ** -0.5)

        reference_q = q_data.detach().clone().requires_grad_()
        reference_k = k_data.detach().clone().requires_grad_()
        reference_w = w_data.detach().clone().requires_grad_()
        reference_predict = _indexer_predict_oracle(reference_q, reference_k, reference_w, topk_indices)
        target_logits = torch.randn_like(reference_predict)
        target_logits = target_logits.masked_fill(topk_indices == -1, float("-inf"))
        target = torch.softmax(target_logits, dim=-1)
        expected_loss = _explicit_standard_kl(
            target,
            reference_predict,
            topk_indices,
            row_coefficient=row_coefficient,
        )
        (expected_loss * upstream_scale).backward()

        actual_q = q_data.detach().clone().requires_grad_()
        actual_k = k_data.detach().clone().requires_grad_()
        actual_w = w_data.detach().clone().requires_grad_()
        predict = sparse_indexer_predict(
            actual_q,
            actual_k,
            actual_w,
            topk_indices,
            topk_length=topk_length,
        )
        target_before = target.clone()
        predict_before = predict.clone()
        actual_loss = dsa_indexer_kl_from_distribution(
            actual_q,
            actual_k,
            actual_w,
            target,
            predict,
            topk_indices,
            row_coefficient=row_coefficient,
        )
        # Exercise the exact failure mode seen in distributed training: a
        # contiguous scalar view can still have a 4-byte-offset data pointer.
        upstream_storage = torch.tensor([0.0, upstream_scale], dtype=torch.float32, device=device)
        misaligned_upstream_scale = upstream_storage[1]
        assert misaligned_upstream_scale.is_contiguous()
        assert misaligned_upstream_scale.data_ptr() % 16 != 0
        actual_loss.backward(gradient=misaligned_upstream_scale)

        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(target, target_before)
        torch.testing.assert_close(predict, predict_before)
        torch.testing.assert_close(actual_q.grad, reference_q.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(actual_k.grad, reference_k.grad, atol=8e-2, rtol=8e-2)
        torch.testing.assert_close(actual_w.grad, reference_w.grad, atol=5e-2, rtol=5e-2)

    def test_padding_query_mask_zeros_cudnn_loss_and_gradients(self):
        torch.manual_seed(13)
        device = "cuda"
        seq_len = 128
        valid_rows = 64
        topk_indices = _packed_topk_indices((seq_len,), topk=128, device=device)
        topk_length = (topk_indices != -1).sum(dim=-1, dtype=torch.int32)
        valid_query_mask = torch.arange(seq_len, device=device).unsqueeze(0) < valid_rows
        row_coefficient = 1.0 / valid_rows

        q_data = torch.randn(1, seq_len, 32, 128, dtype=torch.bfloat16, device=device)
        k_data = torch.randn(1, seq_len, 128, dtype=torch.bfloat16, device=device)
        w_data = torch.randn(1, seq_len, 32, dtype=torch.bfloat16, device=device) * ((32 * 128) ** -0.5)

        reference_q = q_data.detach().clone().requires_grad_()
        reference_k = k_data.detach().clone().requires_grad_()
        reference_w = w_data.detach().clone().requires_grad_()
        reference_predict = _indexer_predict_oracle(reference_q, reference_k, reference_w, topk_indices)
        target_logits = torch.randn_like(reference_predict).masked_fill(topk_indices == -1, float("-inf"))
        target = torch.softmax(target_logits, dim=-1)
        masked_target, masked_predict, masked_topk = _mask_invalid_query_rows(
            target,
            reference_predict,
            topk_indices,
            valid_query_mask,
        )
        expected_loss = _explicit_standard_kl(
            masked_target,
            masked_predict,
            masked_topk,
            row_coefficient=row_coefficient,
        )
        expected_loss.backward()

        actual_q = q_data.detach().clone().requires_grad_()
        actual_k = k_data.detach().clone().requires_grad_()
        actual_w = w_data.detach().clone().requires_grad_()
        predict = sparse_indexer_predict(
            actual_q,
            actual_k,
            actual_w,
            topk_indices,
            topk_length=topk_length,
        )
        actual_loss = dsa_indexer_kl_from_distribution(
            actual_q,
            actual_k,
            actual_w,
            target,
            predict,
            topk_indices,
            row_coefficient=row_coefficient,
            valid_query_mask=valid_query_mask,
        )
        actual_loss.backward()

        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_q.grad, reference_q.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(actual_k.grad, reference_k.grad, atol=8e-2, rtol=8e-2)
        torch.testing.assert_close(actual_w.grad, reference_w.grad, atol=5e-2, rtol=5e-2)
        assert torch.count_nonzero(actual_q.grad[:, valid_rows:]) == 0
        assert torch.count_nonzero(actual_w.grad[:, valid_rows:]) == 0
        assert torch.count_nonzero(actual_k.grad[:, valid_rows:]) == 0

    def test_single_layer_convenience_wrapper_matches_distribution_api(self):
        torch.manual_seed(19)
        device = "cuda"
        seq_len = 128
        topk_indices = _packed_topk_indices((128,), topk=128, device=device)
        topk_length = (topk_indices != -1).sum(dim=-1, dtype=torch.int32)
        index_q = torch.randn(1, seq_len, 32, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        index_k = torch.randn(1, seq_len, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        index_weights = (
            torch.randn(1, seq_len, 32, dtype=torch.bfloat16, device=device) * ((32 * 128) ** -0.5)
        ).requires_grad_()
        attn_q = torch.randn(1, seq_len, 64, 576, dtype=torch.bfloat16, device=device)
        attn_k = torch.randn(1, seq_len, 576, dtype=torch.bfloat16, device=device)
        softmax_scale = 576**-0.5
        _, attn_lse = _attention_target_oracle(attn_q, attn_k, topk_indices, softmax_scale)

        target = sparse_attention_target(
            attn_q,
            attn_k,
            attn_lse,
            topk_indices,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
        )
        predict = sparse_indexer_predict(
            index_q,
            index_k,
            index_weights,
            topk_indices,
            topk_length=topk_length,
        )
        expected = dsa_indexer_kl_from_distribution(
            index_q,
            index_k,
            index_weights,
            target,
            predict,
            topk_indices,
            row_coefficient=0.02,
        )
        actual = dsa_indexer_kl_loss(
            index_q,
            index_k,
            index_weights,
            attn_q,
            attn_k,
            attn_lse,
            topk_indices,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            row_coefficient=0.02,
        )

        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
