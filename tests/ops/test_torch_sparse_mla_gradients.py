"""Gradient accumulation for repeated sparse KV indices."""

import pytest
import torch

from xtuner.v1.ops.sparse_mla import torch_sparse_mla


class TestTorchSparseMLAGradients:
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")],
            ),
        ],
    )
    def test_repeated_bf16_kv_indices_accumulate_in_fp32(self, device: str) -> None:
        # Each query selects one valid key and one padding slot in both KV groups.
        # All 512 value gradients must reach that key; BF16 accumulation stalls at 256.
        q = torch.zeros((512, 2, 2), dtype=torch.bfloat16, device=device)
        kv = torch.zeros((1, 2, 2), dtype=torch.bfloat16, device=device, requires_grad=True)
        indices = torch.tensor([0, -1], dtype=torch.int32, device=device).expand(512, 2, 2)

        result = torch_sparse_mla(q, kv, indices, scaling=1.0)
        result.raw_output.float().sum().backward()

        torch.testing.assert_close(kv.grad, torch.full_like(kv, 512), rtol=0, atol=0)
