# Copyright (c) OpenMMLab. All rights reserved.
"""KDA gate prefix 的 head 切分精度与实际前反向。

TestKDAForwardPrecision
    test_prefix_matches_reference_and_head_slices  短序列、尾块及 packed 文档的独立参考
    test_chunk_output_is_stable_across_head_slices  2K 下完整 heads 与 SP=2/4 head 切片一致
    test_compiled_forward_and_gradients            fullgraph 编译保持输出及全部输入梯度
"""

import pytest
import torch
import torch.nn.functional as F
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices

from xtuner.v1.ops.kda.chunk_cumsum import chunk_cumsum
from xtuner.v1.ops.kda.chunk_kda import chunk_kda


@pytest.mark.gpu
class TestKDAForwardPrecision:
    """验证实际 KDA 入口不会随 head 数选择不同的 gate prefix 加法顺序。"""

    @pytest.mark.parametrize("lengths", [(37,), (64,), (65,), (769, 1279)])
    @pytest.mark.parametrize("packed", [False, True])
    def test_prefix_matches_reference_and_head_slices(self, lengths: tuple[int, ...], packed: bool) -> None:
        # FP64 cumsum 作为独立 oracle；文档边界和尾部必须按 64-token 块重置。
        torch.manual_seed(179)
        boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        length = sum(lengths)
        gate = -5 * torch.rand(1, length, 64, 128, device="cuda")
        cu = boundaries if packed else None
        indices = prepare_chunk_indices(cu, 64) if cu is not None else None
        expected = torch.empty_like(gate)
        start = 0
        for size in lengths if packed else (length,):
            for offset in range(0, size, 64):
                stop = min(start + offset + 64, start + size)
                values = gate[:, start + offset : stop]
                expected[:, start + offset : stop] = (values.double().cumsum(1) * RCP_LN2).float()
            start += size
        actual = chunk_cumsum(gate, cu, indices, RCP_LN2)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-7)
        for split in (2, 4):
            partitioned = torch.cat(
                [chunk_cumsum(part.contiguous(), cu, indices, RCP_LN2) for part in gate.chunk(split, dim=2)], dim=2
            )
            torch.testing.assert_close(actual, partitioned, atol=0, rtol=0)

    def test_chunk_output_is_stable_across_head_slices(self) -> None:
        # 固定 q/k/v/g/beta，模拟 Ulysses 只切 heads；覆盖 2K 和非整块文档边界。
        torch.manual_seed(180)
        shape = (1, 2048, 64, 128)
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        gate = -5 * torch.rand(shape, device="cuda")
        beta = torch.rand(shape[:-1], device="cuda")
        cu = torch.tensor([0, 769, 2048], device="cuda", dtype=torch.int32)
        kwargs = {"cu_seqlens": cu, "safe_gate": True, "transpose_state_layout": True}
        actual = chunk_kda(q, k, v, gate, beta, **kwargs)[0]
        for split in (2, 4):
            partitioned = torch.cat(
                [
                    chunk_kda(*(p.contiguous() for p in parts), **kwargs)[0]
                    for parts in zip(*(value.chunk(split, dim=2) for value in (q, k, v, gate, beta)))
                ],
                dim=2,
            )
            torch.testing.assert_close(actual, partitioned, atol=1e-6, rtol=0)

    def test_forward_and_gradients_match_recurrent_reference(self) -> None:
        # 独立 PyTorch 逐 token 递推，检查新 prefix 与原 FLA backward 的数学契约。
        torch.manual_seed(182)
        shape = (1, 65, 4, 16)
        values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        values += [-torch.rand(shape, device="cuda"), torch.rand(shape[:-1], device="cuda")]
        inputs = [v.detach().clone().requires_grad_() for v in values]
        reference_inputs = [v.detach().clone().requires_grad_() for v in values]
        cu = torch.tensor([0, 37, 65], device="cuda", dtype=torch.int32)
        actual = chunk_kda(*inputs, cu_seqlens=cu, safe_gate=True, transpose_state_layout=True)[0]

        q, k, v, gate, beta = reference_inputs
        q = F.normalize(q.float(), dim=-1).to(q.dtype).float()
        k = F.normalize(k.float(), dim=-1).to(k.dtype).float()
        outputs = []
        for start, stop in [(0, 37), (37, 65)]:
            state = torch.zeros(4, 16, 16, device="cuda")
            for token in range(start, stop):
                state = state * gate[0, token].exp().unsqueeze(-1)
                key = k[0, token]
                delta = (v[0, token].float() - (state * key.unsqueeze(-1)).sum(-2)) * beta[0, token, :, None]
                state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
                outputs.append((state * q[0, token].unsqueeze(-1)).sum(-2) * 16**-0.5)
        expected = torch.stack(outputs).unsqueeze(0).to(v.dtype)
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=2e-2)
        upstream = torch.randn_like(actual)
        gradients = torch.autograd.grad(actual, inputs, upstream)
        reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
        for name, gradient, reference in zip(("q", "k", "v", "g", "beta"), gradients, reference_gradients):
            assert torch.isfinite(gradient).all(), name
            relative_error = (gradient.float() - reference.float()).norm() / reference.float().norm()
            assert relative_error < 2e-2, (name, relative_error)

    def test_compiled_forward_and_gradients(self) -> None:
        # compile 必须复用真实 prefix，且 q/k/v/g/beta 的梯度全部保持一致。
        torch.manual_seed(181)
        shape = (1, 65, 4, 16)
        values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        values += [-torch.rand(shape, device="cuda"), torch.rand(shape[:-1], device="cuda")]
        eager_inputs = [v.detach().clone().requires_grad_() for v in values]
        compiled_inputs = [v.detach().clone().requires_grad_() for v in values]
        cu = torch.tensor([0, 37, 65], device="cuda", dtype=torch.int32)

        def forward(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            return chunk_kda(q, k, v, g, b, cu_seqlens=cu, safe_gate=True, transpose_state_layout=True)[0]

        expected = forward(*eager_inputs)
        actual = torch.compile(forward, fullgraph=True)(*compiled_inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        upstream = torch.randn_like(expected)
        gradients = torch.autograd.grad(actual, compiled_inputs, upstream)
        reference_gradients = torch.autograd.grad(expected, eager_inputs, upstream)
        for actual_gradient, reference in zip(gradients, reference_gradients):
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, reference, atol=0, rtol=0)
