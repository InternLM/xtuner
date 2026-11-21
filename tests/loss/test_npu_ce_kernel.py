# Copyright (c) OpenMMLab. All rights reserved.
"""Run on an Ascend worker with torch_npu and Triton Ascend installed."""

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("triton")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU required", allow_module_level=True)

# Load only the kernel: do not import the full training/distributed stack.
spec = importlib.util.spec_from_file_location(
    "npu_ce_under_test", Path(__file__).resolve().parents[2] / "xtuner/v1/loss/npu_ce.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("reduction", ["sum", "none"])
@pytest.mark.parametrize("has_lse", [False, True])
@pytest.mark.parametrize("vocab", [17, 1025])
def test_native_backward_matches_reference(dtype, reduction, has_lse, vocab):
    # No RNG dependence. The target is in the second vocabulary tile for vocab=1025.
    x = torch.full((3, vocab), -20.0, dtype=dtype, device="npu")
    x[0].fill_(-10.75)
    x[0, 0] = 0.0
    x[0, 1] = -10.75
    x[0, 2:] = -40.0
    x[1, -1] = 0.0
    x[2].fill_(0.0)
    labels = torch.tensor([0, vocab - 1, -100], dtype=torch.int64, device="npu")
    upstream = torch.tensor([2.3862588705e-7, 0.125, 0.25], device="npu", dtype=torch.float32)
    if reduction == "sum":
        upstream = upstream[:1]
    cpu = x.cpu().double()
    lse = torch.logsumexp(cpu, dim=-1)
    saved = lse.clone()
    if not has_lse:
        saved[0] -= cpu[0, 0]
        saved[1] -= cpu[1, -1]
    stats = torch.tensor([1.0, 1.0, 1.0], device="npu", dtype=torch.float32)
    output = torch.empty_like(x)
    module.cross_entropy_backward_kernel_no_weight[(2,)](
        X_ptr=x,
        X_stride=x.stride(0),
        Y_ptr=labels,
        lse_ptr=saved.float().to("npu"),
        grad_output_ptr=upstream,
        grad_output_stride=1 if reduction == "none" else 0,
        dX_ptr=output,
        dX_stride=output.stride(0),
        n_cols=vocab,
        n_rows=3,
        ce_stats_ptr=stats,
        ignore_index=-100,
        reduction=reduction,
        BLOCK_SIZE=1024,
        HAS_LSE=has_lse,
    )
    expected = cpu.softmax(-1)
    expected[0, 0] -= 1
    expected[1, -1] -= 1
    expected *= upstream.cpu().double().reshape(-1, 1)
    expected[2].zero_()
    actual = output.cpu().double()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0)
    # A generic absolute tolerance on the entire tensor would hide this tiny target.
    torch.testing.assert_close(actual[0, 0], expected[0, 0], rtol=0.02, atol=1e-14)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-8)
