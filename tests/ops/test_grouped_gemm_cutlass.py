import pytest
import torch


pytest.importorskip("grouped_gemm_backend")

from grouped_gemm import backend

from xtuner.v1.ops.moe.cuda.group_gemm_cutlass import cutlass_group_gemm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("use_cutlass", [False, True])
def test_grouped_gemm_wrapper_supports_natural_gradients(
    compile: bool,
    use_cutlass: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "use_cutlass", use_cutlass)
    grouped_gemm = torch.compile(cutlass_group_gemm, fullgraph=True) if compile else cutlass_group_gemm

    for sizes in ([2, 0, 5], [1, 3, 3]):
        torch.manual_seed(0)
        counts = torch.tensor(sizes, device="cuda", dtype=torch.int32)
        hidden_states = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        hidden_states_ref = hidden_states.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()

        output = grouped_gemm(hidden_states, weight, counts)
        expected_groups = []
        offset = 0
        for expert, size in enumerate(sizes):
            expected_groups.append(hidden_states_ref[offset : offset + size] @ weight_ref[expert].T)
            offset += size
        expected = torch.cat(expected_groups)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        expected.backward(grad_output)

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(hidden_states.grad, hidden_states_ref.grad)
        torch.testing.assert_close(weight.grad, weight_ref.grad)
        if sizes[1] == 0:
            torch.testing.assert_close(weight.grad[1], torch.zeros_like(weight.grad[1]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile", [False, True])
def test_cublas_path_accepts_preallocated_gradient(
    compile: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "use_cutlass", False)
    hidden_states = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([2], device="cuda", dtype=torch.int32)
    grad_weight_out = torch.empty_like(weight)
    grouped_gemm = torch.compile(cutlass_group_gemm, fullgraph=True) if compile else cutlass_group_gemm

    output = grouped_gemm(hidden_states, weight, counts, grad_weight_out=grad_weight_out)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    expected = grad_output.T @ hidden_states.detach()
    torch.testing.assert_close(grad_weight_out[0], expected)
    torch.testing.assert_close(weight.grad, grad_weight_out)
