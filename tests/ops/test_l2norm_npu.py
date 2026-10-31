"""Regression tests for the runtime-length NPU l2norm."""

import pytest
import torch

from xtuner.v1.ops.gated_deltanet.l2norm_npu import (
    l2norm,
    l2norm_bwd_kernel,
    l2norm_fwd_kernel,
)


requires_npu = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")


def _reference(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The inline fp32 decomposition the GDN path falls back to."""
    original_dtype = x.dtype
    xf = x.float()
    inv_norm = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)
    return (xf * inv_norm).to(original_dtype)


def _triton_cache_size(kernel) -> int:
    return sum(len(entries) for entries in kernel.cache.values())


@requires_npu
class TestL2NormRuntimeLength:
    def test_forward_matches_reference(self):
        # 131072 / 131104 rows are the exact training-time row counts of the
        # k-branch l2norm (32768 / 32776 padded tokens x 4 k-heads); 2100 rows
        # crosses the BT=16 block boundary.
        shapes = [(1, 32768, 4, 128), (1, 32776, 4, 128), (3, 100, 16, 128)]
        for shape in shapes:
            x = (torch.randn(shape, device="npu") * 0.5).bfloat16()
            out = l2norm(x)
            expected = _reference(x)
            assert torch.allclose(out.float(), expected.float(), rtol=1e-3, atol=1e-3), f"shape {shape}"

    def test_backward_matches_reference(self):
        # Rows below BT=16 leave parts of the gradient buffer unwritten in the
        # fla_npu kernel design itself, so backward is only defined for
        # row counts that are multiples of BT; training always satisfies this.
        shapes = [(1, 32768, 4, 128), (1, 32776, 4, 128), (3, 100, 16, 128)]
        for shape in shapes:
            x = (torch.randn(shape, device="npu") * 0.5).bfloat16().requires_grad_(True)
            out = l2norm(x)
            grad = torch.randn(shape, device="npu").bfloat16()
            (dx,) = torch.autograd.grad(out, x, grad)
            (expected,) = torch.autograd.grad(_reference(x), x, grad.float())
            assert torch.allclose(dx.float(), expected.float(), rtol=1e-2, atol=1e-2), f"shape {shape}"

    def test_single_binary_across_row_counts(self):
        # The tgs-spike root cause: fla_npu's kernels inline the row count as
        # tl.constexpr, so the +8-token megabatch variant recompiled mid-run
        # (2-3s bisheng stall per kernel). A runtime-length T alone is not
        # enough — triton still specializes runtime ints by divisibility
        # class, and the variant family flips NB=cdiv(rows, 2048) from
        # 256/64 (div-16) to 257/65 (odd), which recompiled a second binary
        # in run11. With T do_not_specialize'd (and the dead NB dropped),
        # every row count must share exactly one binary per kernel. The row
        # counts must include a non-div-16 value or the divisibility class
        # never changes and the assertion cannot detect a dropped
        # do_not_specialize: 5001 tokens x 4 heads = 20004 rows.
        l2norm(torch.randn(1, 32768, 4, 128, device="npu").bfloat16())
        for tokens in (32776, 32780, 32772):
            l2norm(torch.randn(1, tokens, 4, 128, device="npu").bfloat16())
        x = torch.randn(1, 5001, 4, 128, device="npu").bfloat16().requires_grad_(True)
        # The backward runs on a T % BT != 0 row count purely to pin the
        # compile behavior — its value domain there is undefined (see
        # test_backward_matches_reference), so nothing about dx is asserted.
        torch.autograd.grad(l2norm(x), x, torch.randn(1, 5001, 4, 128, device="npu").bfloat16())
        assert _triton_cache_size(l2norm_fwd_kernel) == 1
        assert _triton_cache_size(l2norm_bwd_kernel) == 1
