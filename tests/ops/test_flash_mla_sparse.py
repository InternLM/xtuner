"""FlashMLA SparseMLA 后端的数值正确性测试。

TestFlashMLASparse
    test_softmax_lse_is_natural_log: 前向导出的 softmax_lse 是自然对数 LSE。
    test_backward_matches_reference: 反向梯度与 PyTorch 稠密参考实现一致。

回归背景：FlashMLA 返回的 LSE 是自然对数，而 TileLang 反向核消费的是 log2 空间的 LSE。
FlashMLA 仓库的 README 描述了以base-2 计算的形式，与实际实现不符合。实际实现返回的lse是自然对数，且反向核消费的是 log2 LSE。
"""

import math

import pytest
import torch

from xtuner.v1.ops.sparse_mla.flash_mla import flash_mla_sparse_mla


_HEAD_DIM = 576
_VALUE_DIM = 512


def _flash_mla_unavailable() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability()[0] < 9:
        return "FlashMLA SparseMLA requires SM90+"
    try:
        import flash_mla  # noqa: F401
    except ImportError:
        return "requires the flash_mla package"
    return None


def _make_inputs(seq_q: int = 128, seq_kv: int = 512, num_heads: int = 64, topk: int = 256):
    torch.manual_seed(0)
    q = torch.randn(seq_q, num_heads, _HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    kv = torch.randn(seq_kv, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    indices = torch.stack([torch.randperm(seq_kv, device="cuda")[:topk] for _ in range(seq_q)])
    return q, kv, indices.to(torch.int32).unsqueeze(1)


def _reference(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scaling: float):
    """稠密 float32 参考实现，返回 (output, natural-log LSE)。"""
    selected = kv.squeeze(1).float()[indices.squeeze(1).long()]
    scores = torch.einsum("qhd,qkd->qhk", q.float(), selected) * scaling
    probs = scores.softmax(dim=-1)
    output = torch.einsum("qhk,qkd->qhd", probs, selected[..., :_VALUE_DIM])
    return output, torch.logsumexp(scores, dim=-1)


@pytest.mark.gpu
@pytest.mark.skipif(_flash_mla_unavailable() is not None, reason=_flash_mla_unavailable() or "")
class TestFlashMLASparse:
    def test_softmax_lse_is_natural_log(self):
        # softmax_lse 必须是自然对数；若误当成 log2 会整体差一个 log2(e) 因子。
        q, kv, indices = _make_inputs()
        scaling = _HEAD_DIM**-0.5

        actual = flash_mla_sparse_mla(q, kv, indices, scaling, value_dim=_VALUE_DIM)
        _, expected_lse = _reference(q, kv, indices, scaling)

        torch.testing.assert_close(actual.softmax_lse, expected_lse, rtol=1e-3, atol=1e-3)
        assert not torch.allclose(actual.softmax_lse, expected_lse * math.log2(math.e), rtol=1e-2)

    def test_backward_matches_reference(self):
        # 反向走的是 TileLang 核（消费 log2 LSE），梯度量级必须与参考实现一致。
        q, kv, indices = _make_inputs()
        scaling = _HEAD_DIM**-0.5
        grad_output = torch.randn(q.shape[0], q.shape[1], _VALUE_DIM, dtype=torch.bfloat16, device="cuda")

        q_flash = q.clone().requires_grad_(True)
        kv_flash = kv.clone().requires_grad_(True)
        flash_mla_sparse_mla(q_flash, kv_flash, indices, scaling, value_dim=_VALUE_DIM).raw_output.backward(
            grad_output
        )

        q_ref = q.clone().float().requires_grad_(True)
        kv_ref = kv.clone().float().requires_grad_(True)
        _reference(q_ref, kv_ref, indices, scaling)[0].backward(grad_output.float())

        for actual, expected in ((q_flash.grad, q_ref.grad), (kv_flash.grad, kv_ref.grad)):
            relative_error = (actual.float() - expected).norm() / expected.norm()
            assert relative_error < 1e-2, f"gradient relative error {relative_error:.3e} is too large"
