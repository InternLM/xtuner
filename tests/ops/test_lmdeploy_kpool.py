# Copyright (c) OpenMMLab. All rights reserved.
"""FP8 forward parity, pooling STE gradients and packed KPool boundaries."""

import builtins
import importlib
import math
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import get_kpool_topk_indices
from xtuner.v1.ops.sparse_mla.kpool import (
    _lmdeploy_kpool_is_available,
    build_pool_index,
    kpool_topk_indices,
    pool_causal_ranges,
)
from xtuner.v1.ops.sparse_mla.lmdeploy_kpool import (
    _score_chunk_size,
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["extend", "decode"])
    @pytest.mark.parametrize("round_scale", [False, True])
    def test_cuda_forward_matches_torch_reference(self, mode: str, round_scale: bool) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        reference = pytest.importorskip("lmdeploy.pytorch.nn.kpool")
        torch.manual_seed(83)
        keys = torch.randn(31, 4, 128, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 128, device="cuda")
        actual = lmdeploy_kpool_compress(keys, scores, ape, mode=mode, round_scale=round_scale)
        pooled = reference.kpool_compress(keys, scores, ape, mode=mode)
        values, scales = reference.kpool_quantize_fp8(pooled, block_size=128, round_scale=round_scale)
        # This reference does not call the fused operator. FP32 accumulation
        # differences can straddle a BF16/FP8 midpoint; bitwise parity with the
        # serving operator is checked separately above.
        torch.testing.assert_close(actual.scales, scales, rtol=0.02, atol=0)
        torch.testing.assert_close(actual.dequantized, values.float() * scales, rtol=0.15, atol=0.02)

    @pytest.mark.gpu
    def test_pre_sm90_compression_uses_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        backend = pytest.importorskip("lmdeploy.pytorch.backends.cuda.kpool")
        reference = pytest.importorskip("lmdeploy.pytorch.nn.kpool")
        keys = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 128, device="cuda")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
        with patch.object(backend, "compress_kpool", side_effect=AssertionError("SM90 kernel must not run")):
            actual = lmdeploy_kpool_compress(keys, scores, ape)
        pooled = reference.kpool_compress(keys, scores, ape, mode="decode")
        values, scales = reference.kpool_quantize_fp8(pooled, block_size=128, round_scale=True)
        assert torch.equal(actual.values.view(torch.uint8), values.view(torch.uint8))
        assert torch.equal(actual.scales, scales)


class TestLMDeployKPoolAvailability:
    def test_public_selector_preserves_original_path_without_lmdeploy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bool] = []

        def original_selector(
            q: Tensor, k: Tensor, weights: Tensor, starts: Tensor, ends: Tensor, topk: int, **kwargs: object
        ) -> Tensor:
            calls.append(True)
            columns = torch.arange(topk)[None]
            return torch.where(columns < (ends - starts)[:, None], columns + starts[:, None], -1).int()

        implementation = importlib.import_module("xtuner.v1.ops.sparse_mla.kpool")
        monkeypatch.setattr(implementation, "_lmdeploy_kpool_is_available", lambda device: False)
        monkeypatch.setattr(implementation, "tilelang_indexer_topk_from_ranges", original_selector)
        ctx = SequenceContext.from_input_ids((torch.zeros(1, 9, dtype=torch.long),), device="cpu")
        q = torch.randn(1, 9, 32, 128)
        k = torch.randn(1, 9, 128)
        actual = kpool_topk_indices(
            q,
            k,
            torch.randn_like(k),
            torch.randn(1, 9, 32),
            torch.randn(4, 128),
            ctx,
            index_head_dim=128,
            index_topk=2048,
        )
        assert calls == [True]
        for row in range(9):
            valid = actual[row, 0][actual[row, 0] >= 0].sort().values
            torch.testing.assert_close(valid, torch.arange(row + 1, dtype=torch.int32), atol=0, rtol=0)

    def test_cpu_and_meta_do_not_require_optional_dependencies(self) -> None:
        assert not _lmdeploy_kpool_is_available(torch.device("cpu"))
        assert not _lmdeploy_kpool_is_available(torch.device("meta"))

    def test_missing_lmdeploy_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_import = builtins.__import__

        def without_lmdeploy(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("lmdeploy"):
                raise ModuleNotFoundError("LMDeploy intentionally unavailable")
            return original_import(name, *args, **kwargs)

        _lmdeploy_kpool_is_available.cache_clear()
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
        with patch("builtins.__import__", side_effect=without_lmdeploy):
            assert not _lmdeploy_kpool_is_available(torch.device("cuda", 0))
        _lmdeploy_kpool_is_available.cache_clear()

    def test_pre_sm90_does_not_import_lmdeploy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _lmdeploy_kpool_is_available.cache_clear()
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
        with patch("builtins.__import__", side_effect=AssertionError("unexpected optional import")):
            assert not _lmdeploy_kpool_is_available(torch.device("cuda", 0))
        _lmdeploy_kpool_is_available.cache_clear()

    @pytest.mark.parametrize("heads", [32, 64, 128])
    @pytest.mark.parametrize("pools", [257, 32768, 262145])
    def test_score_chunks_respect_aligned_byte_budget(
        self, monkeypatch: pytest.MonkeyPatch, heads: int, pools: int
    ) -> None:
        envs = pytest.importorskip("lmdeploy.pytorch.envs")
        monkeypatch.setattr(envs, "dsa_indexer_max_logits_mb", 32)
        for requested in (None, 7, 16384):
            rows = _score_chunk_size(pools, heads, requested)
            block_q = 128 // heads
            allocated_rows = math.ceil(rows / block_q) * block_q
            allocated_columns = math.ceil(pools / 256) * 256
            assert allocated_rows * allocated_columns * 4 <= 32 * (1 << 20)
            assert 0 < rows <= (1024 if requested is None else requested)

    def test_score_budget_rejects_less_than_one_aligned_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        envs = pytest.importorskip("lmdeploy.pytorch.envs")
        monkeypatch.setattr(envs, "dsa_indexer_max_logits_mb", 1)
        with pytest.raises(ValueError, match="minimum aligned allocation"):
            _score_chunk_size(262145, 32, None)


class TestLMDeployKPoolIndexer:
    @pytest.mark.parametrize("autocast", [False, True])
    @pytest.mark.parametrize("use_lmdeploy", [False, True])
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_backend_registration_and_fp32_projection(
        self, monkeypatch: pytest.MonkeyPatch, autocast: bool, use_lmdeploy: bool, device: str
    ) -> None:
        from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer

        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA required")
        monkeypatch.setattr(
            "xtuner.v1.model.moe.glm53.nope_dsa_mla._lmdeploy_kpool_is_available", lambda device: use_lmdeploy
        )
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
        ).to(device=device, dtype=torch.bfloat16)
        hidden = torch.randn(1, 7, 16, dtype=torch.bfloat16, device=device)
        latent = torch.randn(1, 7, 8, dtype=torch.bfloat16, device=device)
        captured: list[Tensor] = []

        def capture(q: Tensor, k: Tensor, scores: Tensor, weights: Tensor, *args: object, **kwargs: object) -> Tensor:
            captured.extend((k, weights))
            return torch.zeros(7, 1, 2560, dtype=torch.int32, device=device)

        module._topk_indices_fn = capture
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, 7, dtype=torch.long),), device=device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            module(hidden, latent, seq_ctx)
        expected = torch.nn.functional.linear(hidden.float(), module.weights_proj.weight.float())
        assert captured[1].dtype == torch.float32
        target = expected if use_lmdeploy else module.weights_proj(hidden).float()
        torch.testing.assert_close(captured[1], target, atol=0, rtol=0)
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
        monkeypatch.setattr(
            importlib.import_module("xtuner.v1.ops.sparse_mla.lmdeploy_kpool"),
            "lmdeploy_kpool_compress",
            lambda *args, **kwargs: pytest.fail("Top-K must not allocate an unused dequantized output"),
        )
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
