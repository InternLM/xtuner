# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn import functional as F

from xtuner.v1.ops.fp32_linear import fp32_linear


class TestFP32Linear:
    @pytest.mark.parametrize("rows", [1, 63, 512, 1025])
    def test_forward_and_gradients(self, rows):
        torch.manual_seed(17)
        x = torch.randn(2, rows, 19, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(13, 19, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(13, dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(2, rows, 13)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = fp32_linear(x, w, b, block_size=512)
        assert y.dtype == torch.float32
        expected = F.linear(x.double(), w.double(), b.double())
        torch.testing.assert_close(y.double(), expected, atol=2e-5, rtol=2e-5)
        grads = torch.autograd.grad(y, (x, w, b), dy, retain_graph=True)
        reference = torch.autograd.grad(expected, (x, w, b), dy.double())
        for actual, oracle in zip(grads, reference):
            torch.testing.assert_close(actual, oracle, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
    )
    def test_sp_geometry_and_fullgraph(self, device):
        torch.manual_seed(42)
        x = torch.randn(1, 2048, 129, device=device, dtype=torch.bfloat16)
        w = torch.randn(31, 129, device=device, dtype=torch.bfloat16)
        reference = fp32_linear(x, w, block_size=512)
        compiled = torch.compile(fp32_linear, fullgraph=True)
        for sp in (1, 2, 4):
            pieces = [fp32_linear(part, w, block_size=512) for part in x.chunk(sp, dim=1)]
            torch.testing.assert_close(torch.cat(pieces, dim=1), reference, atol=0, rtol=0)
        torch.testing.assert_close(compiled(x, w, block_size=512), reference, atol=2e-5, rtol=2e-5)

    def test_empty_and_invalid_block(self):
        x, w = torch.empty(2, 0, 3), torch.empty(4, 3)
        assert fp32_linear(x, w, block_size=512).shape == (2, 0, 4)
        with pytest.raises(ValueError, match="nonnegative"):
            fp32_linear(x, w, block_size=-1)
