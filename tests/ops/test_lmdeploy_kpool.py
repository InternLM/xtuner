# Copyright (c) OpenMMLab. All rights reserved.
"""FP8 forward parity, logits storage budgets and packed KPool boundaries."""

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
    _compress_storage,
    _kpool_topk_indices,
    _score_chunk_size,
)


class TestKPoolCompression:
    @pytest.mark.parametrize("mode", ["extend", "decode"])
    def test_cpu_forward_matches_reference(self, mode: str) -> None:
        reference = pytest.importorskip("lmdeploy.pytorch.nn.kpool")
        torch.manual_seed(12)
        keys = torch.randn(5, 4, 8)
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 8)
        values, scales = _compress_storage(keys, scores, ape, mode, True)
        pooled = reference.kpool_compress(keys, scores, ape, mode=mode)
        expected_values, expected_scales = reference.kpool_quantize_fp8(pooled, block_size=8, round_scale=True)
        assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
        assert torch.equal(scales, expected_scales)

    @pytest.mark.parametrize("mode", ["extend", "decode"])
    def test_empty_groups(self, mode: str) -> None:
        keys = torch.empty(0, 4, 8)
        scores = torch.empty_like(keys)
        values, scales = _compress_storage(keys, scores, torch.randn(4, 8), mode, True)
        assert values.shape == (0, 8) and values.dtype == torch.float8_e4m3fn
        assert scales.shape == (0, 1) and scales.dtype == torch.float32

    def test_rejects_invalid_inputs_before_optional_import(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            _compress_storage(torch.zeros(2, 4, 8), torch.zeros(2, 4, 7), torch.zeros(4, 8), "decode", True)
        with pytest.raises(ValueError, match="power-of-two"):
            _compress_storage(torch.zeros(2, 4, 7), torch.zeros(2, 4, 7), torch.zeros(4, 7), "decode", True)
        with pytest.raises(ValueError, match="mode"):
            _compress_storage(torch.zeros(2, 4, 8), torch.zeros(2, 4, 8), torch.zeros(4, 8), "invalid", True)
        with pytest.raises(TypeError, match="expects"):
            _compress_storage(
                torch.zeros(2, 4, 8, dtype=torch.int32), torch.zeros(2, 4, 8), torch.zeros(4, 8), "decode", True
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["extend", "decode"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("round_scale", [False, True])
    def test_cuda_forward_is_bitwise_lmdeploy(self, mode: str, dtype: torch.dtype, round_scale: bool) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        backend = pytest.importorskip("lmdeploy.pytorch.backends.cuda.kpool")
        torch.manual_seed(13)
        keys = torch.randn(31, 4, 128, dtype=dtype, device="cuda")
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 128, device="cuda")
        expected_values, expected_scales = backend.kpool_compress_quantize_cuda(
            keys, scores, ape, mode=mode, round_scale=round_scale
        )
        values, scales = _compress_storage(keys, scores, ape, mode, round_scale)
        assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
        assert torch.equal(scales, expected_scales)

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
        values, scales = _compress_storage(keys, scores, ape, mode, round_scale)
        pooled = reference.kpool_compress(keys, scores, ape, mode=mode)
        expected_values, expected_scales = reference.kpool_quantize_fp8(
            pooled, block_size=128, round_scale=round_scale
        )
        # FP32 accumulation can straddle BF16/FP8 rounding boundaries. Serving
        # operator bitwise parity is checked separately above.
        torch.testing.assert_close(scales, expected_scales, rtol=0.02, atol=0)
        torch.testing.assert_close(
            values.float() * scales, expected_values.float() * expected_scales, rtol=0.15, atol=0.02
        )

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
            values, scales = _compress_storage(keys, scores, ape, "decode", True)
        pooled = reference.kpool_compress(keys, scores, ape, mode="decode")
        expected_values, expected_scales = reference.kpool_quantize_fp8(pooled, block_size=128, round_scale=True)
        assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
        assert torch.equal(scales, expected_scales)


class TestLMDeployKPoolAvailability:
    def test_bf16_override_skips_optional_backend_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_GLM53_KPOOL_BF16", "1")
        _lmdeploy_kpool_is_available.cache_clear()
        try:
            with (
                patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("unexpected CUDA probe")),
                patch("builtins.__import__", side_effect=AssertionError("unexpected optional import")),
            ):
                assert not _lmdeploy_kpool_is_available(torch.device("cuda", 0))
        finally:
            _lmdeploy_kpool_is_available.cache_clear()

    @pytest.mark.gpu
    @pytest.mark.parametrize("local_columns", [False, True])
    @pytest.mark.parametrize("pools", [1024, 1025])
    def test_logits_budget_bounds_actual_storage(
        self, monkeypatch: pytest.MonkeyPatch, local_columns: bool, pools: int
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        backend = pytest.importorskip("lmdeploy.pytorch.backends.cuda.kpool")
        envs = pytest.importorskip("lmdeploy.pytorch.envs")
        if local_columns and not backend._mqa_has_local_columns():
            pytest.skip("Compact MQA API unavailable")
        monkeypatch.setattr(envs, "dsa_indexer_max_logits_mb", 1)
        rows = _score_chunk_size(pools, 32, None, local_columns=local_columns)
        keys = torch.zeros(pools, 128, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.ones(pools, device="cuda")

        def allocate(query_rows: int) -> Tensor:
            return backend._get_deep_gemm().fp8_mqa_logits(
                torch.zeros(query_rows, 32, 128, device="cuda", dtype=torch.float8_e4m3fn),
                (keys, scales),
                torch.ones(query_rows, 32, device="cuda"),
                torch.zeros(query_rows, device="cuda", dtype=torch.int32),
                torch.full((query_rows,), pools, device="cuda", dtype=torch.int32),
                clean_logits=False,
                **({"max_seqlen_k": pools} if local_columns else {}),
            )

        # Check the real backing allocation, including hidden guard columns
        # and padded rows, for a full chunk and an unaligned final chunk.
        for query_rows in (rows, 7):
            logits = allocate(query_rows)
            assert logits.untyped_storage().nbytes() <= 1 << 20
        if not local_columns and pools == 1024:
            # Review regression: 4096 tokens give 1024 pools. The old compact
            # formula chose 256 rows, whose legacy backing store is 1.25 MiB.
            assert allocate(256).untyped_storage().nbytes() == 5 * (1 << 18)

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
    @pytest.mark.parametrize("local_columns", [False, True])
    def test_score_chunks_respect_aligned_byte_budget(
        self, monkeypatch: pytest.MonkeyPatch, heads: int, pools: int, local_columns: bool
    ) -> None:
        envs = pytest.importorskip("lmdeploy.pytorch.envs")
        monkeypatch.setattr(envs, "dsa_indexer_max_logits_mb", 32)
        for requested in (None, 7, 16384):
            rows = _score_chunk_size(pools, heads, requested, local_columns=local_columns)
            block_q = 128 // heads
            allocated_rows = math.ceil(rows / block_q) * block_q
            allocated_columns = math.ceil(pools / 256) * 256 if local_columns else math.ceil((pools + 256) / 64) * 64
            assert allocated_rows * allocated_columns * 4 <= 32 * (1 << 20)
            assert 0 < rows <= (1024 if requested is None else requested)

    @pytest.mark.parametrize("local_columns", [False, True])
    def test_score_budget_rejects_less_than_one_aligned_block(
        self, monkeypatch: pytest.MonkeyPatch, local_columns: bool
    ) -> None:
        envs = pytest.importorskip("lmdeploy.pytorch.envs")
        monkeypatch.setattr(envs, "dsa_indexer_max_logits_mb", 1)
        with pytest.raises(ValueError, match="minimum aligned allocation"):
            _score_chunk_size(262145, 32, None, local_columns=local_columns)


class TestLMDeployKPoolIndexer:
    @pytest.mark.parametrize("autocast", [False, True])
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_bf16_override_restores_projection_and_selector(
        self, monkeypatch: pytest.MonkeyPatch, autocast: bool, device: str
    ) -> None:
        from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer

        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA required")
        monkeypatch.setenv("XTUNER_GLM53_KPOOL_BF16", "1")
        _lmdeploy_kpool_is_available.cache_clear()
        try:
            torch.manual_seed(85)
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
            hidden = torch.randn(1, 9, 16, dtype=torch.bfloat16, device=device)
            latent = torch.randn(1, 9, 8, dtype=torch.bfloat16, device=device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                expected_keys = module.k_norm(module.wk(hidden))
                expected_weights = module.weights_proj(hidden).float()
            calls: list[bool] = []

            def original_selector(
                q: Tensor, k: Tensor, weights: Tensor, starts: Tensor, ends: Tensor, topk: int, **kwargs: object
            ) -> Tensor:
                assert q.dtype == torch.bfloat16
                assert k.dtype == expected_keys.dtype
                calls.append(True)
                columns = torch.arange(topk, device=q.device)[None]
                return torch.where(columns < (ends - starts)[:, None], columns + starts[:, None], -1).int()

            def checked_selection(
                q: Tensor, k: Tensor, scores: Tensor, weights: Tensor, *args: object, **kwargs: object
            ) -> Tensor:
                torch.testing.assert_close(k, expected_keys, atol=0, rtol=0)
                torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
                return kpool_topk_indices(q, k, scores, weights, *args, **kwargs)

            implementation = importlib.import_module("xtuner.v1.ops.sparse_mla.kpool")
            monkeypatch.setattr(implementation, "tilelang_indexer_topk_from_ranges", original_selector)
            module._topk_indices_fn = checked_selection
            ctx = SequenceContext.from_input_ids((torch.zeros(1, 9, dtype=torch.long),), device=device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                result = module(hidden, latent, ctx)
            assert calls == [True]
            for row in range(9):
                valid = result[row, 0][result[row, 0] >= 0].sort().values
                wanted = torch.arange(row + 1, dtype=torch.int32, device=device)
                torch.testing.assert_close(valid, wanted, atol=0, rtol=0)
        finally:
            _lmdeploy_kpool_is_available.cache_clear()

    @pytest.mark.gpu
    @pytest.mark.parametrize("query_chunk_size", [7, 257])
    def test_query_quantization_and_mqa_logits(self, monkeypatch: pytest.MonkeyPatch, query_chunk_size: int) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        backend = pytest.importorskip("lmdeploy.pytorch.backends.cuda.kpool")
        from lmdeploy.pytorch.kernels.cuda.kpool import compress_kpool, rotate_kpool_query
        from lmdeploy.pytorch.nn.kpool import kpool_quantize_fp8, kpool_score

        local_columns = backend._mqa_has_local_columns()
        torch.manual_seed(84)
        lengths = (2051, 13)
        tokens = sum(lengths)
        ctx = SequenceContext.from_input_ids(
            tuple(torch.zeros(1, length, dtype=torch.long) for length in lengths), device="cuda"
        )
        q = torch.randn(1, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn_like(k)
        weights = torch.randn(1, tokens, 32, device="cuda")
        ape = torch.randn(4, 128, device="cuda")
        pool_index = build_pool_index(ctx, tokens, 4, q.device)
        valid = (pool_index >= 0).unsqueeze(-1)
        keys = torch.where(valid, k[0, pool_index.clamp_min(0)], 0)
        gates = torch.where(valid, scores[0, pool_index.clamp_min(0)], 0)
        expected_kv, expected_ks = compress_kpool(keys, gates, ape, mode="decode", round_scale=True)
        expected_qv, expected_qs = kpool_quantize_fp8(rotate_kpool_query(q[0]), round_scale=True)
        real_mqa = backend._get_deep_gemm().fp8_mqa_logits
        offset = 0
        errors: list[Tensor] = []
        magnitudes: list[Tensor] = []

        def checked_mqa(
            query_fp8: Tensor,
            kv: tuple[Tensor, Tensor],
            query_weight: Tensor,
            starts: Tensor,
            ends: Tensor,
            **kwargs: object,
        ) -> Tensor:
            nonlocal offset
            chunk = slice(offset, offset + query_fp8.shape[0])
            assert torch.equal(query_fp8.view(torch.uint8), expected_qv[chunk].view(torch.uint8))
            assert torch.equal(kv[0].view(torch.uint8), expected_kv.view(torch.uint8))
            assert torch.equal(kv[1], expected_ks.flatten())
            expected_weight = weights[0, chunk] * 32**-0.5 * expected_qs[chunk].squeeze(-1) * 128**-0.5
            torch.testing.assert_close(query_weight, expected_weight, atol=0, rtol=0)
            logits = real_mqa(query_fp8, kv, query_weight, starts, ends, **kwargs)
            reference = kpool_score(
                expected_qv[chunk], expected_qs[chunk], expected_kv, expected_ks, weights[0, chunk] * 32**-0.5
            )
            columns = torch.arange(logits.shape[1], device=q.device)[None]
            columns = columns + starts[:, None] if local_columns else columns.expand(logits.shape[0], -1)
            mask = (columns >= starts[:, None]) & (columns < ends[:, None])
            expected = reference.gather(1, columns.clamp_max(reference.shape[1] - 1).long())
            actual_valid, expected_valid = logits[mask].double(), expected[mask].double()
            errors.append((actual_valid - expected_valid).square().sum())
            magnitudes.append((actual_valid.square() + expected_valid.square()).sum())
            offset += query_fp8.shape[0]
            return logits

        monkeypatch.setattr(backend._get_deep_gemm(), "fp8_mqa_logits", checked_mqa)
        result = _kpool_topk_indices(
            q,
            k,
            scores,
            weights,
            ape,
            ctx,
            index_head_dim=128,
            index_topk=2048,
            query_chunk_size=query_chunk_size,
        )
        assert offset == tokens
        assert result.shape == (tokens, 1, 2560)
        # DeepGEMM's own simulated-FP8 reference uses calc_diff < 5e-6:
        # sum((actual - expected)^2) / sum(actual^2 + expected^2).
        # A pointwise relative tolerance is unstable around signed cancellation.
        assert torch.stack(errors).sum() <= 5e-6 * torch.stack(magnitudes).sum()

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
        if not _lmdeploy_kpool_is_available(torch.device("cuda", torch.cuda.current_device())):
            pytest.skip("Requires the LMDeploy FP8 path, compatible dependencies and SM90+")
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
        if not _lmdeploy_kpool_is_available(torch.device("cuda", torch.cuda.current_device())):
            pytest.skip("Requires the LMDeploy FP8 path, compatible dependencies and SM90+")
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
        if not _lmdeploy_kpool_is_available(torch.device("cuda", torch.cuda.current_device())):
            pytest.skip("Requires the LMDeploy FP8 path, compatible dependencies and SM90+")
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
