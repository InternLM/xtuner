# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

import xtuner.v1.float8.float8_gmm_tile_wise as grouped_gemm_module
from xtuner.v1.float8.float8_gmm_tile_wise import ADAPTIVEGEMM_INSTALLED, TileWiseFloat8GroupedLinear
from xtuner.v1.float8.triton_kernels.dual_layout_quant import (
    per_tile_quant_with_trans_per_block,
    per_tile_quant_with_trans_per_tile,
)
from xtuner.v1.float8.triton_kernels.per_tile_quant import per_tile_quant
from xtuner.v1.float8.triton_kernels.trans_quant_per_block import trans_per_block_quant_expand_128x
from xtuner.v1.float8.triton_kernels.trans_quant_per_tile import trans_per_tile_quant_expand_128x


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: dtype {actual.dtype} != {expected.dtype}"
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), f"{name}: bits differ"


@pytest.mark.parametrize(
    ("tokens_per_expert", "hidden_size"),
    [
        ([1], 128),
        ([0, 1, 127, 128, 129], 256),
        ([130, 5, 128], 256),
        ([0, 1, 127, 128, 129], 6144),
    ],
)
def test_per_tile_quant_with_trans_per_block_matches_reference(
    tokens_per_expert: list[int], hidden_size: int
) -> None:
    torch.manual_seed(42)
    sizes = torch.tensor(tokens_per_expert, device="cuda", dtype=torch.int32)
    x = torch.randn(sum(tokens_per_expert), hidden_size, device="cuda", dtype=torch.bfloat16)
    if x.numel() > 0:
        x.view(-1)[0] = 0
        x.view(-1)[-1] = torch.finfo(torch.bfloat16).max

    row_reference = per_tile_quant(x)
    trans_reference = trans_per_block_quant_expand_128x(x, sizes)
    actual = per_tile_quant_with_trans_per_block(x, sizes)

    _assert_bitwise_equal(actual[0], row_reference[0], "row fp8")
    _assert_bitwise_equal(actual[1], row_reference[1], "row scales")
    _assert_bitwise_equal(actual[2], trans_reference[0], "transposed fp8")
    _assert_bitwise_equal(actual[3], trans_reference[1], "transposed scales")
    _assert_bitwise_equal(actual[4], trans_reference[2], "expanded expert sizes")


@pytest.mark.parametrize(
    ("tokens_per_expert", "hidden_size"),
    [
        ([1], 128),
        ([0, 1, 127, 128, 129], 256),
        ([130, 5, 128], 256),
        ([0, 1, 127, 128, 129], 6144),
    ],
)
def test_per_tile_quant_with_trans_per_tile_matches_reference(
    tokens_per_expert: list[int], hidden_size: int
) -> None:
    torch.manual_seed(42)
    sizes = torch.tensor(tokens_per_expert, device="cuda", dtype=torch.int32)
    x = torch.randn(sum(tokens_per_expert), hidden_size, device="cuda", dtype=torch.bfloat16)
    if x.numel() > 0:
        x.view(-1)[0] = 0
        x.view(-1)[-1] = torch.finfo(torch.bfloat16).max

    row_reference = per_tile_quant(x)
    trans_reference = trans_per_tile_quant_expand_128x(x, sizes)
    actual = per_tile_quant_with_trans_per_tile(x, sizes)
    meaningful_tokens = int(trans_reference[2].sum().item())

    _assert_bitwise_equal(actual[0], row_reference[0], "row fp8")
    _assert_bitwise_equal(actual[1], row_reference[1], "row scales")
    _assert_bitwise_equal(actual[2][:, :meaningful_tokens], trans_reference[0][:, :meaningful_tokens], "transposed fp8")
    _assert_bitwise_equal(
        actual[3][:, : meaningful_tokens // 128],
        trans_reference[1][:, : meaningful_tokens // 128],
        "transposed scales",
    )
    _assert_bitwise_equal(actual[4], trans_reference[2], "expanded expert sizes")


@pytest.mark.skipif(not ADAPTIVEGEMM_INSTALLED, reason="requires adaptive_gemm")
def test_grouped_linear_forward_and_backward_match_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    def reference_per_block(
        x: torch.Tensor,
        sizes: torch.Tensor,
        group_size: int = 128,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = per_tile_quant(x, group_size, dtype)
        trans = trans_per_block_quant_expand_128x(x, sizes, group_size, dtype)
        return row[0], row[1], trans[0], trans[1], trans[2]

    def reference_per_tile(
        x: torch.Tensor,
        sizes: torch.Tensor,
        group_size: int = 128,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = per_tile_quant(x, group_size, dtype)
        trans = trans_per_tile_quant_expand_128x(x, sizes, dtype=dtype)
        return row[0], row[1], trans[0], trans[1], trans[2]

    torch.manual_seed(42)
    layer = TileWiseFloat8GroupedLinear(
        128, 128, 2, enable_fused_moe_activation=True
    ).cuda().to(torch.bfloat16)
    sizes = torch.tensor([129, 127], device="cuda", dtype=torch.long)
    grad_output = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    x_reference = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    monkeypatch.setattr(grouped_gemm_module, "per_tile_quant_with_trans_per_block", reference_per_block)
    monkeypatch.setattr(grouped_gemm_module, "per_tile_quant_with_trans_per_tile", reference_per_tile)
    output_reference = layer(x_reference, sizes)
    output_reference.backward(grad_output)
    dx_reference = x_reference.grad.clone()
    dw_reference = layer.weight.grad.clone()
    layer.weight.grad = None

    monkeypatch.setattr(
        grouped_gemm_module,
        "per_tile_quant_with_trans_per_block",
        per_tile_quant_with_trans_per_block,
    )
    monkeypatch.setattr(
        grouped_gemm_module,
        "per_tile_quant_with_trans_per_tile",
        per_tile_quant_with_trans_per_tile,
    )
    x_actual = x_reference.detach().clone().requires_grad_(True)
    output_actual = layer(x_actual, sizes)
    output_actual.backward(grad_output)

    _assert_bitwise_equal(output_actual, output_reference, "grouped linear output")
    _assert_bitwise_equal(x_actual.grad, dx_reference, "grouped linear input gradient")
    _assert_bitwise_equal(layer.weight.grad, dw_reference, "grouped linear weight gradient")
