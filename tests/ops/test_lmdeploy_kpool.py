# Copyright (c) OpenMMLab. All rights reserved.
"""FP8 forward parity, pooling STE gradients and packed KPool boundaries."""

import math

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import get_kpool_topk_indices
from xtuner.v1.ops.sparse_mla.kpool import build_pool_index, kpool_topk_indices, pool_causal_ranges
from xtuner.v1.ops.sparse_mla.lmdeploy_kpool import (
    kpool_compress_backward,
    lmdeploy_kpool_compress,
)


def _oracle_gradients(keys: Tensor, scores: Tensor, ape: Tensor, grad: Tensor) -> tuple[Tensor, ...]:
    values = [value.detach().double().requires_grad_() for value in (keys, scores, ape)]
    width = keys.shape[-1]
    # Explicit FP64 matrix multiplication is independent of the butterfly and
    # the analytical softmax backward in the implementation.
    matrix = torch.tensor(
        [[(-1.0) ** ((row & column).bit_count()) for column in range(width)] for row in range(width)],
        device=keys.device,
        dtype=torch.float64,
    ) / math.sqrt(width)
    probabilities = (values[1] + values[2]).softmax(dim=1)
    pooled = (probabilities * values[0]).sum(dim=1)
    rotated = pooled @ matrix
    return torch.autograd.grad((rotated * grad.double()).sum(), values)


class TestKPoolCompressionBackward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_independent_fp64_autograd(self, dtype: torch.dtype) -> None:
        torch.manual_seed(10)
        keys = torch.randn(7, 4, 8).to(dtype)
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 8)
        grad = torch.randn(7, 8)
        actual = kpool_compress_backward(grad, keys, scores, ape)
        expected = _oracle_gradients(keys, scores, ape, grad)
        for result, reference, original in zip(actual, expected, (keys, scores, ape)):
            assert result.dtype == original.dtype
            torch.testing.assert_close(result.float(), reference.float(), rtol=0.008, atol=0.002)
        if dtype == torch.float32:
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result.double(), reference, rtol=2e-5, atol=2e-6)

    def test_zero_gradient_and_empty_groups(self) -> None:
        for groups in (0, 3):
            keys = torch.randn(groups, 4, 8)
            scores = torch.randn_like(keys)
            ape = torch.randn(4, 8)
            result = kpool_compress_backward(torch.zeros(groups, 8), keys, scores, ape)
            for value in result:
                assert torch.count_nonzero(value) == 0
            assert result[2].shape == ape.shape

    def test_score_gradient_has_softmax_shift_invariance(self) -> None:
        torch.manual_seed(11)
        keys = torch.randn(5, 4, 8)
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 8)
        grad = torch.randn(5, 8)
        result = kpool_compress_backward(grad, keys, scores, ape)
        shifted = kpool_compress_backward(grad, keys, scores + torch.randn(5, 1, 8), ape)
        torch.testing.assert_close(result[1].sum(dim=1), torch.zeros(5, 8), atol=1e-6, rtol=0)
        for actual, expected in zip(result, shifted):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


class TestKPoolCompression:
    @pytest.mark.parametrize("mode", ["extend", "decode"])
    def test_cpu_forward_and_autograd_contract(self, mode: str) -> None:
        pytest.importorskip("lmdeploy.pytorch.nn.kpool")
        from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8

        torch.manual_seed(12)
        keys = torch.randn(5, 4, 8, requires_grad=True)
        scores = torch.randn(5, 4, 8, requires_grad=True)
        ape = torch.randn(4, 8, requires_grad=True)
        result = lmdeploy_kpool_compress(keys, scores, ape, mode=mode)
        pooled = kpool_compress(keys.detach(), scores.detach(), ape.detach(), mode=mode)
        values, scales = kpool_quantize_fp8(pooled, block_size=8, round_scale=True)
        assert torch.equal(result.values.view(torch.uint8), values.view(torch.uint8))
        assert torch.equal(result.scales, scales)
        assert torch.equal(result.dequantized, values.float() * scales)
        assert result.dequantized.requires_grad
        assert not result.values.requires_grad
        assert not result.scales.requires_grad
        grad = torch.randn_like(result.dequantized)
        actual = torch.autograd.grad(result.dequantized, (keys, scores, ape), grad)
        for value, expected in zip(actual, _oracle_gradients(keys, scores, ape, grad)):
            torch.testing.assert_close(value.double(), expected, rtol=2e-5, atol=2e-6)

    @pytest.mark.parametrize("mode", ["extend", "decode"])
    def test_cpu_empty_groups(self, mode: str) -> None:
        pytest.importorskip("lmdeploy.pytorch.nn.kpool")
        keys = torch.empty(0, 4, 8, requires_grad=True)
        scores = torch.empty_like(keys, requires_grad=True)
        ape = torch.randn(4, 8, requires_grad=True)
        result = lmdeploy_kpool_compress(keys, scores, ape, mode=mode)
        assert result.values.shape == (0, 8)
        result.dequantized.sum().backward()
        torch.testing.assert_close(ape.grad, torch.zeros_like(ape))

    def test_rejects_invalid_inputs_before_optional_import(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            lmdeploy_kpool_compress(torch.zeros(2, 4, 8), torch.zeros(2, 4, 7), torch.zeros(4, 8))
        with pytest.raises(ValueError, match="power-of-two"):
            lmdeploy_kpool_compress(torch.zeros(2, 4, 7), torch.zeros(2, 4, 7), torch.zeros(4, 7))
        with pytest.raises(ValueError, match="mode"):
            lmdeploy_kpool_compress(torch.zeros(2, 4, 8), torch.zeros(2, 4, 8), torch.zeros(4, 8), mode="invalid")
        with pytest.raises(TypeError, match="expects"):
            lmdeploy_kpool_compress(torch.zeros(2, 4, 8, dtype=torch.int32), torch.zeros(2, 4, 8), torch.zeros(4, 8))

    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["extend", "decode"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("round_scale", [False, True])
    def test_cuda_forward_is_bitwise_lmdeploy(self, mode: str, dtype: torch.dtype, round_scale: bool) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("lmdeploy.pytorch.kernels.cuda.kpool")
        from lmdeploy.pytorch.kernels.cuda.kpool import compress_kpool

        torch.manual_seed(13)
        keys = torch.randn(31, 4, 128, dtype=dtype, device="cuda", requires_grad=True)
        scores = torch.randn_like(keys, requires_grad=True)
        ape = torch.randn(4, 128, device="cuda", requires_grad=True)
        values, scales = compress_kpool(keys, scores, ape, mode=mode, round_scale=round_scale)
        result = lmdeploy_kpool_compress(keys, scores, ape, mode=mode, round_scale=round_scale)
        assert torch.equal(result.values.view(torch.uint8), values.view(torch.uint8))
        assert torch.equal(result.scales, scales)
        grad = torch.randn_like(result.dequantized)
        actual = torch.autograd.grad(result.dequantized, (keys, scores, ape), grad)
        expected = _oracle_gradients(keys, scores, ape, grad)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value.float(), reference.float(), rtol=0.01, atol=0.005)
        if dtype == torch.float32:
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value.double(), reference, rtol=1e-4, atol=4e-6)


class TestLMDeployKPoolIndexer:
    def test_backend_registration_and_fp32_projection(self) -> None:
        from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer

        assert get_kpool_topk_indices("tilelang") is kpool_topk_indices
        torch.manual_seed(14)
        module = KPoolIndexer(
            hidden_size=16,
            q_lora_rank=8,
            index_head_dim=128,
            index_n_heads=32,
            index_topk=2048,
            index_kpool=4,
            index_kpool_always_select_tail=True,
            indexer_backend="tilelang",
            alignment=512,
        ).to(torch.bfloat16)
        hidden = torch.randn(1, 7, 16, dtype=torch.bfloat16)
        latent = torch.randn(1, 7, 8, dtype=torch.bfloat16)
        captured: list[Tensor] = []

        def capture(q: Tensor, k: Tensor, scores: Tensor, weights: Tensor, *args: object, **kwargs: object) -> Tensor:
            captured.extend((k, weights))
            return torch.zeros(7, 1, 2560, dtype=torch.int32)

        module._topk_indices_fn = capture
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, 7, dtype=torch.long),), device="cpu")
        module(hidden, latent, seq_ctx)
        expected = torch.nn.functional.linear(hidden.float(), module.weights_proj.weight.float())
        assert captured[1].dtype == torch.float32
        torch.testing.assert_close(captured[1], expected, atol=0, rtol=0)
        assert torch.any(expected != module.weights_proj(hidden).float())

    @pytest.mark.gpu
    def test_preserves_lmdeploy_group_order_and_compact_tails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        backend = pytest.importorskip("lmdeploy.pytorch.backends.cuda.kpool")
        from lmdeploy.pytorch.nn.kpool import kpool_expand_selected_groups

        def reverse_groups(logits: Tensor, lengths: Tensor, *, group_topk: int, **kwargs: object) -> Tensor:
            columns = torch.arange(group_topk, device=logits.device)
            return torch.where(columns[None] < lengths[:, None], lengths[:, None] - 1 - columns[None], -1).int()

        monkeypatch.setattr(backend, "kpool_select_groups_cuda", reverse_groups)
        lengths = (9, 17)
        tokens = sum(lengths)
        seq_ctx = SequenceContext.from_input_ids(
            tuple(torch.zeros(1, n, dtype=torch.long) for n in lengths), device="cuda"
        )
        q = torch.randn(1, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
        gates = torch.randn_like(k)
        weights = torch.randn(1, tokens, 32, device="cuda")
        ape = torch.randn(4, 128, device="cuda")
        actual = kpool_topk_indices(
            q, k, gates, weights, ape, seq_ctx, index_head_dim=128, index_topk=2048, query_chunk_size=7
        )
        local_positions = torch.cat([torch.arange(1, n + 1, device="cuda") for n in lengths]).int()
        group_lengths = local_positions // 4
        groups = reverse_groups(torch.empty(tokens, 1, device="cuda"), group_lengths, group_topk=512)
        expected = kpool_expand_selected_groups(groups, group_lengths, 4, 2048, seq_lens=local_positions)
        offsets = torch.cat([torch.zeros(lengths[0]), torch.full((lengths[1],), lengths[0])]).to("cuda").int()
        expected = torch.where(expected >= 0, expected + offsets[:, None], -1)
        expected = torch.nn.functional.pad(expected, (0, 509), value=-1).unsqueeze(1)
        assert actual.dtype == torch.int32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.gpu
    def test_empty_local_queries(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, 1, dtype=torch.long),), device="cuda")
        result = kpool_topk_indices(
            torch.empty(1, 0, 32, 128, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, 128, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, 128, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, 32, device="cuda"),
            torch.zeros(4, 128, device="cuda"),
            seq_ctx,
            index_head_dim=128,
            index_topk=2048,
        )
        assert result.shape == (0, 1, 2560)
        assert result.dtype == torch.int32

    @pytest.mark.gpu
    @pytest.mark.parametrize("query_chunk_size", [257, 1024])
    def test_packed_selection_is_causal_and_matches_fp8_scores(self, query_chunk_size: int) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("lmdeploy.pytorch.nn.kpool")
        from lmdeploy.pytorch.kernels.cuda.kpool import compress_kpool, rotate_kpool_query
        from lmdeploy.pytorch.nn.kpool import kpool_quantize_fp8, kpool_score

        torch.manual_seed(15)
        lengths = (4099, 2111, 9)
        seq_ctx = SequenceContext.from_input_ids(
            tuple(torch.zeros(1, length, dtype=torch.long) for length in lengths), device="cuda"
        )
        tokens = sum(lengths)
        q = torch.randn(1, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn_like(k)
        weights = torch.randn(1, tokens, 32, device="cuda")
        ape = torch.randn(4, 128, device="cuda")
        result = kpool_topk_indices(
            q,
            k,
            scores,
            weights,
            ape,
            seq_ctx,
            index_head_dim=128,
            index_topk=2048,
            query_chunk_size=query_chunk_size,
        )[:, 0]
        assert result.shape == (tokens, 2560)
        assert result.dtype == torch.int32
        pool_index = build_pool_index(seq_ctx, tokens, 4, q.device)
        starts, ends = pool_causal_ranges(seq_ctx, pool_index, tokens, q.device)
        valid_slots = pool_index >= 0
        grouped_k = torch.where(valid_slots[..., None], k[0, pool_index.clamp_min(0)], 0)
        grouped_scores = torch.where(valid_slots[..., None], scores[0, pool_index.clamp_min(0)], 0)
        kv, ks = compress_kpool(grouped_k, grouped_scores, ape, mode="decode", round_scale=True)
        sample_ids = torch.tensor([0, 3, 2051, 4098, 4099, 6150, tokens - 1], device=q.device)
        qv, qs = kpool_quantize_fp8(rotate_kpool_query(q[0, sample_ids]), round_scale=True)
        reference = kpool_score(qv, qs, kv, ks, weights[0, sample_ids] * 32**-0.5)
        pool_ids = torch.arange(pool_index.shape[0], device=q.device)
        reference.masked_fill_(
            (pool_ids[None] < starts[sample_ids, None]) | (pool_ids[None] >= ends[sample_ids, None]), -torch.inf
        )
        document_start = 0
        for length in lengths:
            rows = result[document_start : document_start + length]
            position = torch.arange(document_start, document_start + length, device=q.device)
            assert torch.all((rows == -1) | ((rows >= document_start) & (rows <= position[:, None])))
            expected_counts = (
                torch.minimum((position - document_start + 1) // 4, torch.tensor(512, device=q.device)) * 4
                + (position - document_start + 1) % 4
            )
            torch.testing.assert_close((rows >= 0).sum(dim=1), expected_counts)
            document_start += length
        for offset, row_id in enumerate(sample_ids.tolist()):
            selected = result[row_id]
            selected = selected[selected >= 0]
            assert torch.unique(selected).numel() == selected.numel()
            count = min(int((ends[row_id] - starts[row_id]).item()), 512)
            if not count:
                continue
            # Pool first-token positions are monotonically increasing except
            # unused trailing padding. Recover complete selected groups directly.
            complete_selected = (pool_index[:, :, None] == selected[None, None]).any(dim=-1).all(dim=-1)
            complete_selected &= (pool_ids >= starts[row_id]) & (pool_ids < ends[row_id])
            actual_scores = reference[offset, complete_selected]
            assert actual_scores.numel() == count
            cutoff = reference[offset].topk(count).values[-1]
            assert torch.all(actual_scores >= cutoff - 0.001)
