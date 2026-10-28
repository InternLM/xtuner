import pytest
import torch
from torch import nn

from xtuner.v1.ops.moe.cuda import cutlass_group_gemm
from xtuner.v1.ops.moe.cuda.group_gemm import triton_group_gemm
from xtuner.v1.ops.moe.cuda.route_weight import route_weight_rows_backward


@pytest.fixture(params=[triton_group_gemm, cutlass_group_gemm], ids=["triton", "cutlass"])
def grouped_gemm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    implementation = request.param
    if implementation is None:
        pytest.skip("requires grouped_gemm")
    if implementation is cutlass_group_gemm:
        from grouped_gemm import backend

        monkeypatch.setattr(backend, "use_cutlass", True)
    return implementation


@pytest.mark.parametrize("compile", [False, True])
def test_grouped_gemm_backward_returns_natural_bf16_weight_gradient(compile: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(17)
    counts = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    x = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = torch.cat(
        (
            x_ref[:2] @ weight_ref[0].T,
            x_ref[2:5] @ weight_ref[2].T,
            x_ref[5:] @ weight_ref[3].T,
        )
    )
    expected.backward(grad_output)

    grouped_gemm = triton_group_gemm
    if compile:
        grouped_gemm = torch.compile(grouped_gemm, fullgraph=True)
    actual = grouped_gemm(x, weight, counts)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(weight.grad[1], torch.zeros_like(weight.grad[1]), rtol=0, atol=0)


@pytest.mark.parametrize("compile", [False, True])
def test_parameter_owns_preallocated_grouped_gemm_gradient_without_copy(grouped_gemm, compile: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    counts = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    hidden_states = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = nn.Parameter(torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16))
    target_storage = torch.empty_like(weight)
    target = target_storage.new_empty(0).set_(
        target_storage.untyped_storage(),
        target_storage.storage_offset(),
        target_storage.shape,
        target_storage.stride(),
    )
    seen_grad_pointers: list[int] = []
    weight.register_post_accumulate_grad_hook(lambda parameter: seen_grad_pointers.append(parameter.grad.data_ptr()))

    if compile:
        grouped_gemm = torch.compile(grouped_gemm, fullgraph=True)
    output = grouped_gemm(hidden_states, weight, counts, grad_weight_out=target)
    del target
    output.sum().backward()

    assert weight.grad is not None
    assert weight.grad.data_ptr() == target_storage.data_ptr()
    assert seen_grad_pointers == [target_storage.data_ptr()]
    torch.testing.assert_close(weight.grad[1], torch.zeros_like(weight.grad[1]), rtol=0, atol=0)


@pytest.mark.parametrize("compile", [False, True])
def test_retained_gradient_target_uses_parameter_copy_path(grouped_gemm, compile: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    counts = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    hidden_states = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = nn.Parameter(torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16))
    target = torch.empty_like(weight)

    if compile:
        grouped_gemm = torch.compile(grouped_gemm, fullgraph=True)
    output = grouped_gemm(hidden_states, weight, counts, grad_weight_out=target)
    output.sum().backward()

    assert weight.grad is not None
    assert weight.grad.data_ptr() != target.data_ptr()
    torch.testing.assert_close(weight.grad, target, rtol=0, atol=0)


@pytest.mark.parametrize("compile", [False, True])
def test_preallocated_grouped_gemm_gradient_covers_zero_token_batch(grouped_gemm, compile: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    counts = torch.zeros(4, device="cuda", dtype=torch.int32)
    hidden_states = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = nn.Parameter(torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16))
    target_storage = torch.full_like(weight, 7)
    target = target_storage.new_empty(0).set_(
        target_storage.untyped_storage(),
        target_storage.storage_offset(),
        target_storage.shape,
        target_storage.stride(),
    )

    if compile:
        grouped_gemm = torch.compile(grouped_gemm, fullgraph=True)
    output = grouped_gemm(hidden_states, weight, counts, grad_weight_out=target)
    del target
    output.sum().backward()

    assert output.shape == (0, 256)
    assert weight.grad is not None
    assert weight.grad.data_ptr() == target_storage.data_ptr()
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight.grad), rtol=0, atol=0)


def test_fused_route_weight_backward_returns_bf16_rows_and_fp32_weights() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(23)
    grad_weighted = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    expert_output = torch.randn_like(grad_weighted)
    route_weights = torch.randn(7, device="cuda", dtype=torch.float32)

    grad_expert, grad_route = route_weight_rows_backward(
        grad_weighted,
        expert_output,
        route_weights,
    )
    # grouped-gemm's BF16 unpermute backward first rounds the FP32 router
    # weight to BF16, and its route-gradient dot rounds every product to BF16
    # before the FP32 reduction. MoonEP must preserve that public numerical
    # contract when it fuses the same operation into combine backward.
    expected_expert = grad_weighted * route_weights.bfloat16()[:, None]
    expected_route = (grad_weighted * expert_output).float().sum(dim=-1)

    assert grad_expert.dtype is torch.bfloat16
    assert grad_route.dtype is torch.float32
    torch.testing.assert_close(grad_expert, expected_expert, rtol=0, atol=0)
    torch.testing.assert_close(grad_route, expected_route, rtol=1e-5, atol=1e-4)
