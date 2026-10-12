"""Direct sequence-SP operator parity, including document and rank boundaries."""

import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.ops.kda.causal_conv1d import causal_conv1d
from xtuner.v1.ops.kda.sequence_parallel import sequence_causal_conv1d, sequence_chunk_kda
from xtuner.v1.utils.test_utils import init_data_mesh


def _assert_parity(actual, expected, dtype, name):
    assert actual is not None and expected is not None, name
    assert torch.isfinite(actual).all(), name
    actual, expected = actual.float(), expected.float()
    if not bool(expected.any()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)
        return
    limit = 2e-2 if dtype == torch.bfloat16 else 3e-3
    error = actual - expected
    assert error.norm() / expected.norm() < limit, name
    assert error.abs().max() / expected.abs().max() < 2 * limit, name


class TestKDASequenceOps(DistributedTestBase):
    def test_convolution_output_and_all_gradients(self, device="cuda"):
        self.create_pg(device)
        mesh = init_data_mesh(device, self.world_size)["sp"]
        rank, group = mesh.get_local_rank(), mesh.get_group()
        total, length = 128, 128 // self.world_size
        selection = slice(rank * length, (rank + 1) * length)
        layouts = ([total], [length - 2, 5, total - length - 3], [length, total - length])
        for dtype in (torch.float32, torch.bfloat16):
            for lengths in layouts:
                for width, activation in ((1, None), (1, "silu"), (4, None), (4, "silu")):
                    torch.manual_seed(1234)
                    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
                    full = torch.randn(1, total, 12, device=device, dtype=dtype, requires_grad=True)
                    weight = torch.randn(12, width, device=device, requires_grad=True)
                    bias = torch.randn(12, device=device, requires_grad=True)
                    local = full.detach()[:, selection].clone().requires_grad_()
                    local_weight = weight.detach().clone().requires_grad_()
                    local_bias = bias.detach().clone().requires_grad_()

                    def forward(x, w, b):
                        return sequence_causal_conv1d(x, w, b, activation, cu, group.group_name)

                    compiled = (
                        dtype == torch.bfloat16 and lengths == layouts[1] and width == 4 and activation == "silu"
                    )
                    runner = torch.compile(forward, fullgraph=True) if compiled else forward
                    expected = causal_conv1d(full, weight, bias, activation=activation, cu_seqlens=cu)[0]
                    output = runner(local, local_weight, local_bias)
                    _assert_parity(output, expected[:, selection], dtype, "convolution output")
                    torch.manual_seed(5678)
                    gradient = torch.zeros_like(expected)
                    gradient[:, -length : -length + 8] = torch.randn_like(gradient[:, :8])
                    expected.backward(gradient)
                    output.backward(gradient[:, selection].contiguous())
                    _assert_parity(local.grad, full.grad[:, selection], dtype, "convolution dx")
                    if lengths == [total] and width > 1 and rank == self.world_size - 2:
                        assert local.grad[:, -3:].abs().sum() > 0, "Missing halo gradient"
                    for actual, reference, name in (
                        (local_weight.grad, weight.grad, "convolution dw"),
                        (local_bias.grad, bias.grad, "convolution db"),
                    ):
                        assert actual.dtype == torch.float32, name
                        torch.distributed.all_reduce(actual, group=group)
                        _assert_parity(actual, reference, dtype, name)

    def test_chunk_output_and_all_gradients(self, device="cuda"):
        self.create_pg(device)
        mesh = init_data_mesh(device, self.world_size)["sp"]
        rank, group = mesh.get_local_rank(), mesh.get_group()
        total, length = 128, 128 // self.world_size
        selection = slice(rank * length, (rank + 1) * length)
        layouts = ([total], [length - 2, 5, total - length - 3], [length, total - length])
        cases = (
            (torch.float32, 0, False, False),
            (torch.float32, 1, True, True),
            (torch.float32, 2, False, True),
            (torch.bfloat16, 0, True, True),
            (torch.bfloat16, 1, False, False),
            (torch.bfloat16, 1, True, True),
            (torch.bfloat16, 2, True, False),
        )
        for dtype, layout, safe_gate, transpose in cases:
            torch.manual_seed(4321)
            lengths = layouts[layout]
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
            shape = (1, total, 2, 16)
            values = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
            values += [torch.randn(1, total, 2, 24, device=device, dtype=dtype)]
            # Slow decay makes an omitted recurrent boundary gradient observable.
            values += [-0.01 * torch.rand(shape, device=device), torch.rand(shape[:-1], device=device)]
            full = [value.detach().clone().requires_grad_() for value in values]
            local = [value[:, selection].clone().requires_grad_() for value in values]

            def forward(q, k, v, g, beta):
                return sequence_chunk_kda(q, k, v, g, beta, cu, group.group_name, safe_gate, transpose)

            compiled = dtype == torch.bfloat16 and layout == 1 and safe_gate and transpose
            runner = torch.compile(forward, fullgraph=True) if compiled else forward
            expected = fla_chunk_kda(
                *full,
                cu_seqlens=cu,
                safe_gate=safe_gate,
                transpose_state_layout=transpose,
                use_qk_l2norm_in_kernel=True,
            )[0]
            output = runner(*local)
            _assert_parity(output, expected[:, selection], dtype, "chunk output")
            torch.manual_seed(5678)
            gradient = torch.zeros_like(expected)
            gradient[:, -length : -length + 8] = torch.randn_like(gradient[:, :8])
            expected.backward(gradient)
            output.backward(gradient[:, selection].contiguous())
            for actual, reference, name in zip(local, full, ("q", "k", "v", "g", "beta")):
                _assert_parity(actual.grad, reference.grad[:, selection], dtype, "chunk d" + name)
            if layout == 0 and rank < self.world_size - 1:
                assert local[0].grad is not None
                # Earlier queries do not affect later outputs, but earlier k/v do.
                assert local[1].grad.abs().sum() > 0 and local[2].grad.abs().sum() > 0

    @property
    def world_size(self) -> int:
        return 2


class TestKDASequenceOps4(TestKDASequenceOps):
    @property
    def world_size(self) -> int:
        return 4
