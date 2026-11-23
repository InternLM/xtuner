# Copyright (c) OpenMMLab. All rights reserved.

from contextlib import contextmanager
from unittest import mock

import pytest
import torch

import xtuner.v1.float8.float8_gmm_tile_wise as grouped_gemm_module
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import (
    ADAPTIVEGEMM_INSTALLED,
    TileWiseFloat8GroupedLinear,
)
from xtuner.v1.float8.triton_kernels.dual_layout_quant import (
    per_tile_quant_with_trans_per_block,
    per_tile_quant_with_trans_per_tile,
)
from xtuner.v1.float8.triton_kernels.per_tile_quant import per_tile_quant
from xtuner.v1.float8.triton_kernels.swiglu_backward import swiglu_backward
from xtuner.v1.float8.triton_kernels.swiglu_dual_layout_quant import (
    swiglu_per_tile_quant_with_trans_per_block,
)
from xtuner.v1.float8.triton_kernels.trans_quant_per_block import trans_per_block_quant_expand_128x
from xtuner.v1.float8.triton_kernels.trans_quant_per_tile import trans_per_tile_quant_expand_128x
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock
from xtuner.v1.ops.act_fn import native_swiglu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: dtype {actual.dtype} != {expected.dtype}"
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), f"{name}: bits differ"


def _original_per_block_quant(
    x: torch.Tensor,
    sizes: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row = per_tile_quant(x, group_size, dtype)
    trans = trans_per_block_quant_expand_128x(x, sizes, group_size, dtype)
    return row[0], row[1], trans[0], trans[1], trans[2]


def _original_per_tile_quant(
    x: torch.Tensor,
    sizes: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row = per_tile_quant(x, group_size, dtype)
    trans = trans_per_tile_quant_expand_128x(x, sizes, group_size, dtype)
    return row[0], row[1], trans[0], trans[1], trans[2]


@contextmanager
def _original_quant_kernels():
    with mock.patch.object(
        grouped_gemm_module,
        "per_tile_quant_with_trans_per_block",
        _original_per_block_quant,
    ), mock.patch.object(
        grouped_gemm_module,
        "per_tile_quant_with_trans_per_tile",
        _original_per_tile_quant,
    ):
        yield


def _original_moe_block_forward(
    block: MoEBlock,
    x: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    gate_up = block.fused_w1w3(x, tokens_per_expert, decoding=False)
    return block.fused_w2(native_swiglu(gate_up), tokens_per_expert, decoding=False)


@pytest.mark.parametrize(
    ("tokens_per_expert", "intermediate_size"),
    [
        ([1], 128),
        ([0, 1, 127, 128, 129], 256),
        ([130, 5, 128], 256),
        ([0, 1, 127, 128, 129], 2048),
    ],
)
def test_swiglu_dual_layout_quant_matches_independent_kernels(
    tokens_per_expert: list[int], intermediate_size: int
) -> None:
    torch.manual_seed(42)
    sizes = torch.tensor(tokens_per_expert, device="cuda", dtype=torch.int32)
    gate_up = torch.randn(
        sum(tokens_per_expert), 2 * intermediate_size, device="cuda", dtype=torch.bfloat16
    )
    if gate_up.numel() > 0:
        gate_up.view(-1)[0] = 0
        gate_up.view(-1)[-1] = -8

    act_reference = native_swiglu(gate_up)
    row_reference = per_tile_quant(act_reference)
    trans_reference = trans_per_block_quant_expand_128x(act_reference, sizes)
    actual = swiglu_per_tile_quant_with_trans_per_block(gate_up, sizes)

    _assert_bitwise_equal(actual[0], row_reference[0], "row fp8")
    _assert_bitwise_equal(actual[1], row_reference[1], "row scales")
    _assert_bitwise_equal(actual[2], trans_reference[0], "transposed fp8")
    _assert_bitwise_equal(actual[3], trans_reference[1], "transposed scales")
    _assert_bitwise_equal(actual[4], trans_reference[2], "expanded expert sizes")


@pytest.mark.parametrize(("tokens", "intermediate_size"), [(1, 128), (257, 256), (1024, 2048)])
def test_swiglu_backward_matches_native_autograd(tokens: int, intermediate_size: int) -> None:
    torch.manual_seed(42)
    gate_up = torch.randn(tokens, 2 * intermediate_size, device="cuda", dtype=torch.bfloat16)
    grad_act = torch.randn(tokens, intermediate_size, device="cuda", dtype=torch.bfloat16)

    gate_up_reference = gate_up.clone().requires_grad_(True)
    grad_reference = torch.autograd.grad(native_swiglu(gate_up_reference), gate_up_reference, grad_act)[0]
    actual = swiglu_backward(gate_up, grad_act)

    _assert_bitwise_equal(actual, grad_reference, "SwiGLU gradient")


@pytest.mark.skipif(not ADAPTIVEGEMM_INSTALLED, reason="requires adaptive_gemm")
def test_moe_block_fused_swiglu_matches_original_path() -> None:
    torch.manual_seed(42)
    block = MoEBlock(
        hidden_size=128,
        moe_intermediate_size=128,
        n_routed_experts=2,
        float8_cfg=Float8Config(
            scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            enable_fused_moe_activation=True,
        ),
        moe_act_fn_cfg=MoEActFnConfig(act_type="swiglu"),
    ).cuda().to(torch.bfloat16)
    sizes = torch.tensor([129, 127], device="cuda", dtype=torch.long)
    grad_output = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    x_reference = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with _original_quant_kernels():
        output_reference = _original_moe_block_forward(block, x_reference, sizes)
        output_reference.backward(grad_output)
    dx_reference = x_reference.grad.clone()
    dw1w3_reference = block.fused_w1w3.weight.grad.clone()
    dw2_reference = block.fused_w2.weight.grad.clone()
    block.zero_grad(set_to_none=True)

    x_actual = x_reference.detach().clone().requires_grad_(True)
    output_actual = block(x_actual, sizes, decoding=False)
    output_actual.backward(grad_output)

    _assert_bitwise_equal(output_actual, output_reference, "MoEBlock output")
    _assert_bitwise_equal(x_actual.grad, dx_reference, "MoEBlock input gradient")
    _assert_bitwise_equal(block.fused_w1w3.weight.grad, dw1w3_reference, "MoEBlock W1W3 gradient")
    _assert_bitwise_equal(block.fused_w2.weight.grad, dw2_reference, "MoEBlock W2 gradient")
