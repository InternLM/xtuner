"""Direct sequence-SP operator parity, including document and rank boundaries."""

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda as fla_chunk_kda
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.ops.kda.causal_conv1d import causal_conv1d
from xtuner.v1.ops.kda.sequence_parallel import sequence_causal_conv1d, sequence_chunk_kda
from xtuner.v1.utils.test_utils import init_data_mesh


def _assert_parity(actual, expected, dtype, name, limit=None):
    assert actual is not None and expected is not None, name
    actual, expected = actual.detach().float(), expected.detach().float()
    if limit is None:
        limit = 2e-2 if dtype == torch.bfloat16 else 3e-3
    error = actual - expected
    if bool(expected.any()):
        relative_l2 = float(error.norm() / expected.norm())
        relative_max = float(error.abs().max() / expected.abs().max())
        valid = relative_l2 < limit and relative_max < 2 * limit
    else:
        relative_l2 = relative_max = float(error.abs().max())
        valid = relative_max == 0
    metrics = torch.tensor([int(not valid), relative_l2, relative_max], device=actual.device)
    # Every rank must stop on an assertion; otherwise peers enter the next collective alone.
    torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.MAX)
    assert metrics[0].item() == 0, (
        f"{name}: maximum-rank relative_l2={metrics[1].item()}, relative_max={metrics[2].item()}, limit={limit}"
    )


def _recurrent_reference(inputs, lengths):
    """Independent token recurrence with an explicit state reset per document."""
    q, k, v, gate, beta = inputs
    q = F.normalize(q.float(), dim=-1).to(q.dtype).float()
    k = F.normalize(k.float(), dim=-1).to(k.dtype).float()
    heads, key_dim, value_dim = q.shape[2], q.shape[3], v.shape[3]
    outputs, start = [], 0
    for length in lengths:
        state = torch.zeros(heads, key_dim, value_dim, device=q.device)
        for token in range(start, start + length):
            state = state * gate[0, token].exp().unsqueeze(-1)
            key = k[0, token]
            delta = v[0, token].float() - (state * key.unsqueeze(-1)).sum(-2)
            delta = delta * beta[0, token, :, None]
            state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q[0, token].unsqueeze(-1)).sum(-2) * key_dim**-0.5)
        start += length
    return torch.stack(outputs)[None].to(v.dtype)


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
            label = f"{dtype} layout={layout} safe_gate={safe_gate} transpose={transpose}"
            torch.manual_seed(4321)
            lengths = layouts[layout]
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
            shape = (1, total, 2, 16)
            values = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
            values += [torch.randn(1, total, 2, 24, device=device, dtype=dtype)]
            # Slow decay makes an omitted recurrent boundary gradient observable.
            values += [-0.01 * torch.rand(shape, device=device), torch.rand(shape[:-1], device=device)]
            full = [value.detach().clone().requires_grad_() for value in values]
            oracle = [value.detach().clone().requires_grad_() for value in values]
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
            independent = _recurrent_reference(oracle, lengths)
            output = runner(*local)
            _assert_parity(output, expected[:, selection], dtype, label + " output")
            oracle_limit = 2e-2 if dtype == torch.bfloat16 else 5e-3
            _assert_parity(output, independent[:, selection], dtype, label + " oracle output", oracle_limit)
            torch.manual_seed(5678)
            gradient = torch.zeros_like(expected)
            gradient[:, -length : -length + 8] = torch.randn_like(gradient[:, :8])
            expected.backward(gradient)
            independent.backward(gradient)
            output.backward(gradient[:, selection].contiguous())
            for actual, reference, recurrence, name in zip(local, full, oracle, ("q", "k", "v", "g", "beta")):
                # FP32 SP4 dg/dbeta differ from SP1 by ~0.33%, while native FLA
                # itself differs from the recurrence by up to ~0.38% here.
                # Retain the 0.3% native budget for dq/dk/dv and check both
                # implementations against the independent 0.5% oracle budget.
                native_limit = 5e-3 if dtype == torch.float32 and name in ("g", "beta") else None
                _assert_parity(actual.grad, reference.grad[:, selection], dtype, label + " d" + name, native_limit)
                _assert_parity(
                    actual.grad, recurrence.grad[:, selection], dtype, label + " oracle d" + name, oracle_limit
                )
                _assert_parity(
                    reference.grad[:, selection],
                    recurrence.grad[:, selection],
                    dtype,
                    label + " native oracle d" + name,
                    oracle_limit,
                )
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
