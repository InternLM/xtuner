import pytest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_distributed import DistributedTestBase
from torch.utils.checkpoint import checkpoint

from xtuner.v1.module.decoder_layer.mhc import MHCConfig, hc_pre
from xtuner.v1.ops.mhc_pre_norm_fn import mhc_pre_norm_fn


def _relative_l2(actual, expected):
    return (actual - expected).norm() / expected.norm().clamp_min(1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mhc_pre_norm_fn_matches_fp32_forward_and_backward(batch_size):
    torch.manual_seed(0)
    x = torch.randn(batch_size, 7, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_fn = fn.detach().clone().float().requires_grad_()

    actual = mhc_pre_norm_fn(x, fn, 1e-5)
    flat = ref_x.flatten(2).float()
    expected = F.linear(F.rms_norm(flat, (flat.shape[-1],), weight=None, eps=1e-5), ref_fn)

    assert actual.shape == expected.shape
    assert _relative_l2(actual, expected) < 2e-4
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert _relative_l2(x.grad, ref_x.grad) < 2e-4
    assert _relative_l2(fn.grad, ref_fn.grad) < 5e-3


@pytest.mark.gpu
def test_hc_pre_fused_path_matches_reference():
    torch.manual_seed(1)
    x = torch.randn(1, 7, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = torch.randn(3, device="cuda", dtype=torch.float32, requires_grad=True)
    base = torch.randn(24, device="cuda", dtype=torch.float32, requires_grad=True)
    tensors = (x, fn, scale, base)
    refs = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)

    actual = hc_pre(*tensors, 4, 20, 1e-6, norm_eps=1e-5, fused_pre=True)
    expected = hc_pre(*refs, 4, 20, 1e-6, norm_eps=1e-5, fused_pre=False)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.shape == expected_tensor.shape
        assert _relative_l2(actual_tensor, expected_tensor) < 2e-4

    sum(tensor.square().sum() for tensor in actual).backward()
    sum(tensor.square().sum() for tensor in expected).backward()
    for tensor, ref in zip(tensors, refs, strict=True):
        assert _relative_l2(tensor.grad, ref.grad) < 5e-3


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [8192, 16384])
def test_mhc_pre_norm_fn_real_shape_error_budget(seq_len):
    torch.manual_seed(2)
    x = torch.randn(1, seq_len, 4, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(24, 16384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_fn = fn.detach().clone().float().requires_grad_()

    actual = mhc_pre_norm_fn(x, fn, 1e-5)
    flat = ref_x.flatten(2).float()
    expected = F.linear(F.rms_norm(flat, (flat.shape[-1],), weight=None, eps=1e-5), ref_fn)
    assert actual.shape == expected.shape
    actual.square().sum().backward()
    expected.square().sum().backward()

    metrics = {
        "output": _relative_l2(actual, expected).item(),
        "x_grad": _relative_l2(x.grad, ref_x.grad).item(),
        "fn_grad": _relative_l2(fn.grad, ref_fn.grad).item(),
    }
    print(f"seq_len={seq_len} relative_l2={metrics}")
    assert metrics["output"] < 2e-4
    assert metrics["x_grad"] < 2e-4
    assert metrics["fn_grad"] < 5e-3


@pytest.mark.gpu
def test_mhc_pre_norm_fn_compile_matches_eager_forward_and_backward():
    torch.manual_seed(3)
    x = torch.randn(1, 7, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_fn = fn.detach().clone().requires_grad_()

    compiled = torch.compile(lambda x, fn: mhc_pre_norm_fn(x, fn, 1e-5), fullgraph=True)
    actual = compiled(x, fn)
    expected = mhc_pre_norm_fn(ref_x, ref_fn, 1e-5)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(fn.grad, ref_fn.grad)


@pytest.mark.gpu
def test_mhc_pre_norm_fn_checkpoint_matches_direct():
    torch.manual_seed(31)
    x = torch.randn(1, 7, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_fn = fn.detach().clone().requires_grad_()

    actual = checkpoint(mhc_pre_norm_fn, x, fn, 1e-5, use_reentrant=False)
    expected = mhc_pre_norm_fn(ref_x, ref_fn, 1e-5)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, ref_x.grad)
    assert torch.equal(fn.grad, ref_fn.grad)


@pytest.mark.gpu
def test_hc_pre_fused_path_is_bitwise_deterministic():
    torch.manual_seed(4)
    inputs = (
        torch.randn(1, 7, 4, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(24, 512, device="cuda", dtype=torch.bfloat16),
        torch.randn(3, device="cuda", dtype=torch.float32),
        torch.randn(24, device="cuda", dtype=torch.float32),
    )
    reference = hc_pre(*inputs, 4, 20, 1e-6, norm_eps=1e-5, fused_pre=True)
    cotangents = tuple(torch.randn_like(tensor) for tensor in reference)

    runs = []
    for _ in range(2):
        tensors = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        outputs = hc_pre(*tensors, 4, 20, 1e-6, norm_eps=1e-5, fused_pre=True)
        torch.autograd.backward(outputs, cotangents)
        runs.append((*outputs, *(tensor.grad for tensor in tensors)))
    for first, second in zip(runs[0], runs[1], strict=True):
        assert torch.equal(first, second)


def test_fused_pre_defaults_off():
    assert MHCConfig(hc_mult=4).fused_pre is False


@pytest.mark.gpu
@pytest.mark.parametrize("invalid_kind", ["cpu", "dtype", "mix_size", "hidden_alignment"])
def test_mhc_pre_norm_fn_rejects_unsupported_inputs(invalid_kind):
    device = "cpu" if invalid_kind == "cpu" else "cuda"
    x = torch.randn(1, 7, 4, 128, device=device, dtype=torch.bfloat16)
    fn = torch.randn(24, 512, device=device, dtype=torch.bfloat16)
    if invalid_kind == "dtype":
        x = x.float()
    elif invalid_kind == "mix_size":
        fn = torch.randn(33, 512, device=device, dtype=torch.bfloat16)
    elif invalid_kind == "hidden_alignment":
        x = torch.randn(1, 7, 4, 127, device=device, dtype=torch.bfloat16)
        fn = torch.randn(24, 508, device=device, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        mhc_pre_norm_fn(x, fn, 1e-5)


@pytest.mark.gpu
def test_mhc_pre_norm_fn_accepts_noncontiguous_fn():
    torch.manual_seed(6)
    x = torch.randn(1, 7, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(24, 1024, device="cuda", dtype=torch.bfloat16)[:, ::2].requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    ref_fn = fn.detach().clone().float().requires_grad_()

    actual = mhc_pre_norm_fn(x, fn, 1e-5)
    flat = ref_x.flatten(2).float()
    expected = F.linear(F.rms_norm(flat, (flat.shape[-1],), weight=None, eps=1e-5), ref_fn)
    assert _relative_l2(actual, expected) < 2e-4
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert _relative_l2(x.grad, ref_x.grad) < 2e-4
    assert _relative_l2(fn.grad, ref_fn.grad) < 5e-3


class TestMHCPreNormFnSPParity(DistributedTestBase):
    @pytest.mark.gpu
    def test_fused_sp2_matches_full_sequence_forward_and_backward(self, device="cuda"):
        self.create_pg(device)
        torch.manual_seed(5)
        x = torch.randn(1, 8, 4, 128, device=device, dtype=torch.bfloat16)
        fn = torch.randn(24, 512, device=device, dtype=torch.bfloat16)
        out_grad = torch.randn(1, 8, 24, device=device, dtype=torch.float32)
        for tensor in (x, fn, out_grad):
            torch.distributed.broadcast(tensor, src=0)

        full_x = x.detach().clone().requires_grad_()
        full_fn = fn.detach().clone().requires_grad_()
        expected = mhc_pre_norm_fn(full_x, full_fn, 1e-5)
        expected.backward(out_grad)

        rank = torch.distributed.get_rank()
        start = rank * (x.shape[1] // self.world_size)
        end = start + x.shape[1] // self.world_size
        local_x = x[:, start:end].detach().clone().requires_grad_()
        local_fn = fn.detach().clone().requires_grad_()
        actual = mhc_pre_norm_fn(local_x, local_fn, 1e-5)
        actual.backward(out_grad[:, start:end])

        torch.testing.assert_close(actual, expected[:, start:end])
        torch.testing.assert_close(local_x.grad, full_x.grad[:, start:end])
        torch.distributed.all_reduce(local_fn.grad)
        relative_l2 = _relative_l2(local_fn.grad, full_fn.grad).item()
        max_abs = (local_fn.grad - full_fn.grad).abs().max().item()
        print(f"rank={rank} fn_grad_relative_l2={relative_l2:.8f} max_abs={max_abs:.8f}")
        # BF16 分片梯度先转回 FP32、再转为 BF16 后求和，沿用真实形状权重梯度预算。
        assert relative_l2 < 5e-3

    @property
    def world_size(self) -> int:
        return 2
