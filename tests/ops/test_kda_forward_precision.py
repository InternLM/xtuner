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
from fla.ops.kda import chunk_kda as fla_chunk_kda
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    @pytest.mark.parametrize(
        "batch,packed,safe_gate,transpose,scale,lower_bound",
        [
            (1, True, True, True, None, None),
            (2, False, False, False, 0.17, None),
            (2, False, True, True, 0.25, -3.0),
            (1, True, False, False, 0.17, None),
            (1, False, False, True, None, None),
            (1, True, True, False, 0.3, -7.0),
        ],
        ids=["default-packed", "batch2-unsafe", "batch2-bounded", "packed-unsafe", "unpacked", "packed-untransposed"],
    )
    def test_forward_and_gradients_match_recurrent_reference(
        self, dtype, batch, packed, safe_gate, transpose, scale, lower_bound
    ) -> None:
        """Check supported options against an independent token-by-token recurrence."""
        torch.manual_seed(182)
        shape = (batch, 65, 4, 16)
        # Rectangular states expose layout mistakes hidden by K == V.
        values = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(2)]
        values += [torch.randn(batch, 65, 4, 24, device="cuda", dtype=dtype)]
        values += [-torch.rand(shape, device="cuda") * abs(lower_bound or -5.0), torch.rand(shape[:-1], device="cuda")]
        inputs = [v.detach().clone().requires_grad_() for v in values]
        reference_inputs = [v.detach().clone().requires_grad_() for v in values]
        native_inputs = [v.detach().clone().requires_grad_() for v in values]
        cu = torch.tensor([0, 37, 65], device="cuda", dtype=torch.int32) if packed else None
        actual = chunk_kda(
            *inputs,
            cu_seqlens=cu,
            safe_gate=safe_gate,
            transpose_state_layout=transpose,
            scale=scale,
            lower_bound=lower_bound,
        )[0]
        native = fla_chunk_kda(
            *native_inputs,
            cu_seqlens=cu,
            safe_gate=safe_gate,
            transpose_state_layout=transpose,
            scale=scale,
            lower_bound=lower_bound,
            use_qk_l2norm_in_kernel=True,
        )[0]

        q, k, v, gate, beta = reference_inputs
        q = F.normalize(q.float(), dim=-1).to(q.dtype).float()
        k = F.normalize(k.float(), dim=-1).to(k.dtype).float()
        outputs = []
        for sample in range(batch):
            sample_outputs = []
            for start, stop in [(0, 37), (37, 65)] if packed else [(0, 65)]:
                state = torch.zeros(4, 16, 24, device="cuda")
                for token in range(start, stop):
                    state = state * gate[sample, token].exp().unsqueeze(-1)
                    key = k[sample, token]
                    delta = v[sample, token].float() - (state * key.unsqueeze(-1)).sum(-2)
                    delta = delta * beta[sample, token, :, None]
                    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
                    sample_outputs.append(
                        (state * q[sample, token].unsqueeze(-1)).sum(-2) * (scale if scale is not None else 16**-0.5)
                    )
            outputs.append(torch.stack(sample_outputs))
        expected = torch.stack(outputs).to(v.dtype)
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=2e-2)
        torch.manual_seed(5678)
        upstream = torch.randn_like(actual)
        gradients = torch.autograd.grad(actual, inputs, upstream)
        reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
        native_gradients = torch.autograd.grad(native, native_inputs, upstream)
        for name, gradient, reference, original in zip(
            ("q", "k", "v", "g", "beta"), gradients, reference_gradients, native_gradients
        ):
            assert torch.isfinite(gradient).all(), name
            relative_error = (gradient.float() - reference.float()).norm() / reference.float().norm()
            limit = 2e-2 if dtype == torch.bfloat16 else 3e-3
            # Native FLA's FP32 safe-gate backward has ~0.5% dg error here; its
            # triangular solve uses TF32. Retain the stricter native parity check.
            if dtype == torch.float32 and safe_gate and name == "g":
                limit = 1e-2
            assert relative_error < limit, (name, relative_error)
            native_error = (gradient.float() - original.float()).norm() / original.float().norm()
            assert native_error < (1e-3 if dtype == torch.bfloat16 else 1e-4), (name, native_error)

    @pytest.mark.parametrize("packed", [True, False], ids=["default-packed", "custom-batch2"])
    def test_compiled_forward_and_gradients(self, packed: bool) -> None:
        # compile 必须复用真实 prefix，且 q/k/v/g/beta 的梯度全部保持一致。
        torch.manual_seed(181)
        batch = 1 if packed else 2
        shape = (batch, 65, 4, 16)
        values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        values += [torch.randn(batch, 65, 4, 24, device="cuda", dtype=torch.bfloat16)]
        values += [-torch.rand(shape, device="cuda"), torch.rand(shape[:-1], device="cuda")]
        eager_inputs = [v.detach().clone().requires_grad_() for v in values]
        compiled_inputs = [v.detach().clone().requires_grad_() for v in values]
        cu = torch.tensor([0, 37, 65], device="cuda", dtype=torch.int32) if packed else None

        def forward(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            return chunk_kda(
                q,
                k,
                v,
                g,
                b,
                cu_seqlens=cu,
                safe_gate=packed,
                transpose_state_layout=packed,
                scale=None if packed else 0.17,
                lower_bound=None if packed else -3.0,
            )[0]

        expected = forward(*eager_inputs)
        actual = torch.compile(forward, fullgraph=True)(*compiled_inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        upstream = torch.randn_like(expected)
        gradients = torch.autograd.grad(actual, compiled_inputs, upstream)
        reference_gradients = torch.autograd.grad(expected, eager_inputs, upstream)
        for actual_gradient, reference in zip(gradients, reference_gradients):
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, reference, atol=0, rtol=0)
