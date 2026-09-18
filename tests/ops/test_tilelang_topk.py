# Copyright (c) OpenMMLab. All rights reserved.
"""TileLang radix top-k correctness tests."""

import importlib.util

import pytest
import torch


def _tilelang_available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("tilelang") is not None


def _torch_topk(input: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, topk: int) -> torch.Tensor:
    columns = torch.arange(input.shape[1], device=input.device)
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    scores, indices = input.masked_fill(~valid, -torch.inf).topk(topk, dim=-1)
    return indices.masked_fill(scores == -torch.inf, -1).to(torch.int32)


@pytest.mark.skipif(not _tilelang_available(), reason="requires CUDA and TileLang")
class TestTileLangTopK:
    @pytest.mark.parametrize("seq_len", [16384, 32768])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("concentrated", [False, True])
    def test_matches_pytorch_bitwise(self, seq_len: int, packed: bool, concentrated: bool) -> None:
        from xtuner.v1.ops.sparse_mla.tilelang_topk import tl_topk

        topk = 2048
        # Exact, unique FP32 values make index order a strict comparison.
        values = torch.arange(seq_len, device="cuda", dtype=torch.float32)
        if concentrated:
            # All values share the first radix byte: exceed the 8192-entry
            # candidate buffer and exercise dense-prefix refinement.
            values = 1 + values / (1 << 23)
        else:
            values = (values - seq_len // 2) / seq_len
        starts = [0, 0, 1024, seq_len // 4] if packed else [0, 0, 0, 0]
        ends = [1, 1024, seq_len // 2, seq_len] if packed else [seq_len] * 4
        input = torch.stack([torch.roll(values, shifts=row * 977) for row in range(len(starts))])
        starts_tensor = torch.tensor(starts, device="cuda", dtype=torch.int32)
        ends_tensor = torch.tensor(ends, device="cuda", dtype=torch.int32)

        expected = _torch_topk(input, starts_tensor, ends_tensor, topk)
        actual = tl_topk(input, starts_tensor, ends_tensor, topk)

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)

    def test_indexer_switch_matches_pytorch_topk_bitwise(self) -> None:
        from xtuner.v1.ops.sparse_mla.tilelang import _tilelang_dsa_topk_indices_from_ranges

        torch.manual_seed(0)
        seq_len = 64
        q = torch.randn(seq_len, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq_len, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(seq_len, 32, device="cuda", dtype=torch.float32)
        starts = torch.zeros(seq_len, device="cuda", dtype=torch.int32)
        ends = torch.arange(1, seq_len + 1, device="cuda", dtype=torch.int32)

        expected = _tilelang_dsa_topk_indices_from_ranges(q, k, weights, starts, ends, 16, False)
        actual = _tilelang_dsa_topk_indices_from_ranges(q, k, weights, starts, ends, 16, True)

        different = actual != expected
        different_rows = different.flatten(1).any(dim=1)
        assert not different.any(), (
            f"different_elements={different.sum().item()}, "
            f"different_rows={different_rows.sum().item()}, "
            f"first_row={different_rows.nonzero()[0].item()}"
        )
