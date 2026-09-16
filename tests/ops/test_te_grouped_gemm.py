"""Correctness tests for the XTuner TE grouped-GEMM adapter."""

import os

import pytest
import torch

import xtuner.v1.ops.moe.cuda.group_gemm_te as adapter
from xtuner.v1.ops.moe.cuda.group_gemm_te import (
    general_grouped_gemm,
    te_grouped_gemm,
)


def test_te_grouped_gemm_forward_backward_reference(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    x = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 6, 4, requires_grad=True)
    y = te_grouped_gemm(x, weight, counts)
    reference = torch.cat((x[:2] @ weight[0].T, x[2:] @ weight[2].T))
    torch.testing.assert_close(y, reference)
    (y.square().sum()).backward()
    assert x.grad is not None and weight.grad is not None
    assert torch.count_nonzero(weight.grad[1]) == 0


def test_te_grouped_gemm_ultraep_replica_gradient(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    counts = torch.tensor([2, 0, 1, 2], dtype=torch.int64)
    x = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 6, 4, requires_grad=True)
    replica = torch.randn(1, 6, 4)
    replica_grad = torch.zeros_like(replica, dtype=torch.float32)
    y = te_grouped_gemm(x, weight, counts, replica_weight=replica, replica_grad=replica_grad)
    reference = torch.cat((x[:2] @ weight[0].T, x[2:3] @ weight[2].T, x[3:] @ replica[0].T))
    torch.testing.assert_close(y, reference)
    y.sum().backward()
    assert float(replica_grad.abs().sum()) > 0
    assert replica_grad.dtype is torch.float32


def test_te_grouped_gemm_prefers_host_counts(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    seen: list[object] = []
    real = adapter.general_grouped_gemm

    def spy(*args, **kwargs):
        seen.append(kwargs["m_splits"])
        return real(*args, **kwargs)

    monkeypatch.setattr(adapter, "general_grouped_gemm", spy)

    device_counts = torch.tensor([9, 9, 9], dtype=torch.int64)
    host_counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    x = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 6, 4, requires_grad=True)

    y = te_grouped_gemm(x, weight, device_counts, tokens_per_expert_cpu=host_counts)
    y.sum().backward()

    assert seen, "general_grouped_gemm was never called"
    assert all(list(split) == [2, 0, 3] for split in seen)
    torch.testing.assert_close(y, torch.cat((x[:2] @ weight[0].T, x[2:] @ weight[2].T)))


def test_te_grouped_gemm_falls_back_to_device_counts(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    seen: list[object] = []
    real = adapter.general_grouped_gemm

    def spy(*args, **kwargs):
        seen.append(kwargs["m_splits"])
        return real(*args, **kwargs)

    monkeypatch.setattr(adapter, "general_grouped_gemm", spy)

    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    x = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 6, 4, requires_grad=True)

    te_grouped_gemm(x, weight, counts).sum().backward()

    assert seen and all(list(split) == [2, 0, 3] for split in seen)


def test_cuda_backend_refuses_to_fall_back_to_reference(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "cutlass")
    if not adapter.TE_GROUPED_GEMM_INSTALLED:
        with pytest.raises(ImportError, match="te_grouped_gemm is not installed"):
            adapter._require_native("cutlass")
        return
    monkeypatch.setattr(adapter._te_pkg, "_NATIVE", None)
    monkeypatch.setattr(adapter._te_pkg, "_NATIVE_TRIED", True)
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._require_native("cutlass")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    adapter._require_native("torch")


def test_general_grouped_gemm_takes_a_weight_list(monkeypatch):
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    weights = [torch.randn(3, 2), torch.randn(3, 2)]
    inputs = [torch.randn(4, 2), torch.randn(1, 2)]
    dest = [torch.empty(4, 3), torch.empty(1, 3)]
    outputs, bias, gelu = general_grouped_gemm(
        weights, inputs, dest, layout="TN", m_splits=[4, 1]
    )
    assert outputs is dest
    torch.testing.assert_close(dest[0], inputs[0] @ weights[0].T)
    torch.testing.assert_close(dest[1], inputs[1] @ weights[1].T)
    assert len(bias) == len(gelu) == 2


def test_selected_backend_cutlass_overrides_te_env(monkeypatch):
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "cutlass")
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "0")
    assert adapter.selected_backend() == "cutlass"
    assert os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] == "1"


def test_get_group_gemm_defaults_to_triton_on_cuda(monkeypatch):
    monkeypatch.delenv("XTUNER_GROUP_GEMM", raising=False)
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    from xtuner.v1.ops.moe import get_group_gemm
    from xtuner.v1.ops.moe.protocol import cpu_group_gemm
    from xtuner.v1.utils import get_device

    gemm = get_group_gemm()
    if get_device() != "cuda":
        assert gemm is cpu_group_gemm
        return

    assert gemm.__name__ == "triton_group_gemm"
    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    x = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 6, 4, requires_grad=True)
    replica = torch.randn(1, 6, 4)
    replica_grad = torch.zeros_like(replica, dtype=torch.float32)
    counts_with_replica = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    x_rep = torch.randn(6, 4, requires_grad=True)
    y = gemm(
        x_rep,
        weight,
        counts_with_replica,
        tokens_per_expert_cpu=counts_with_replica,
        replica_weight=replica,
        replica_grad=replica_grad,
    )
    torch.testing.assert_close(
        y,
        torch.cat(
            (
                x_rep[:2] @ weight[0].T,
                x_rep[2:5] @ weight[2].T,
                x_rep[5:] @ replica[0].T,
            )
        ),
    )
    y_master = gemm(x, weight, counts, tokens_per_expert_cpu=counts)
    torch.testing.assert_close(y_master, torch.cat((x[:2] @ weight[0].T, x[2:] @ weight[2].T)))


def test_get_group_gemm_selects_te_adapter(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")
    from xtuner.v1.ops.moe import get_group_gemm
    from xtuner.v1.ops.moe.protocol import cpu_group_gemm
    from xtuner.v1.utils import get_device

    gemm = get_group_gemm()
    if get_device() != "cuda":
        assert gemm is cpu_group_gemm
        return

    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    x = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 6, 4, requires_grad=True)
    y = gemm(x, weight, counts, tokens_per_expert_cpu=counts)
    torch.testing.assert_close(y, torch.cat((x[:2] @ weight[0].T, x[2:] @ weight[2].T)))
    y.sum().backward()
    assert x.grad is not None and weight.grad is not None


@pytest.mark.parametrize("backend", ["triton", "triton_dual"])
def test_get_group_gemm_selects_triton_when_requested(monkeypatch, backend):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", backend)
    from xtuner.v1.ops.moe import get_group_gemm
    from xtuner.v1.ops.moe.protocol import cpu_group_gemm
    from xtuner.v1.utils import get_device

    gemm = get_group_gemm()
    if get_device() == "cpu":
        assert gemm is cpu_group_gemm
        return
    assert gemm.__name__ == "triton_group_gemm"


def test_get_group_gemm_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "foo")
    from xtuner.v1.ops.moe import get_group_gemm

    with pytest.raises(ValueError, match="XTUNER_GROUP_GEMM"):
        get_group_gemm()


def test_triton_group_gemm_requires_matching_ultraep_replica_args():
    from xtuner.v1.ops.moe.cuda.group_gemm import triton_group_gemm

    x = torch.randn(3, 4)
    weight = torch.randn(2, 6, 4)
    counts = torch.tensor([1, 2], dtype=torch.int64)
    replica = torch.randn(1, 6, 4)
    with pytest.raises(ValueError, match="must be provided together"):
        triton_group_gemm(x, weight, counts, replica_weight=replica)


def test_cutlass_group_gemm_rejects_ultraep_replica():
    from xtuner.v1.ops.moe.cuda.group_gemm_cutlass import cutlass_group_gemm

    x = torch.randn(3, 4)
    weight = torch.randn(2, 6, 4)
    counts = torch.tensor([1, 2], dtype=torch.int64)
    replica = torch.randn(1, 6, 4)
    replica_grad = torch.zeros_like(replica, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="does not support UltraEP replica weights"):
        cutlass_group_gemm(x, weight, counts, replica_weight=replica, replica_grad=replica_grad)


def _require_native_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not adapter.TE_GROUPED_GEMM_INSTALLED:
        pytest.skip("te_grouped_gemm is not installed")
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "cublas")
    monkeypatch.setattr(adapter._te_pkg, "_NATIVE_TRIED", False)
    monkeypatch.setattr(adapter._te_pkg, "_NATIVE", None)
    native = adapter._load_native()
    if native is None or not hasattr(native, "te_general_grouped_gemm"):
        pytest.skip("te_grouped_gemm._C is not available")
    return native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_grouped_gemm_cuda_matches_torch_reference(monkeypatch):
    _require_native_cuda(monkeypatch)
    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    x = torch.randn(5, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(3, 6, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = te_grouped_gemm(x, weight, counts, tokens_per_expert_cpu=counts)
    reference = torch.cat((x[:2] @ weight[0].T, x[2:] @ weight[2].T))
    torch.testing.assert_close(y, reference, atol=2e-2, rtol=2e-2)
    y.square().sum().backward()
    assert x.grad is not None and weight.grad is not None
    assert torch.count_nonzero(weight.grad[1]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_grouped_gemm_cuda_ultraep_replica_gradient(monkeypatch):
    _require_native_cuda(monkeypatch)
    counts = torch.tensor([2, 0, 1, 2], dtype=torch.int64)
    x = torch.randn(5, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(3, 6, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    replica = torch.randn(1, 6, 4, device="cuda", dtype=torch.bfloat16)
    replica_grad = torch.zeros_like(replica, dtype=torch.float32)
    y = te_grouped_gemm(
        x,
        weight,
        counts,
        tokens_per_expert_cpu=counts,
        replica_weight=replica,
        replica_grad=replica_grad,
    )
    reference = torch.cat((x[:2] @ weight[0].T, x[2:3] @ weight[2].T, x[3:] @ replica[0].T))
    torch.testing.assert_close(y, reference, atol=2e-2, rtol=2e-2)
    y.sum().backward()
    assert float(replica_grad.abs().sum()) > 0
    assert replica_grad.dtype is torch.float32
    assert replica_grad.device == x.device
