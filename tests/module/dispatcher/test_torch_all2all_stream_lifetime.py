from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

import xtuner.v1.module.dispatcher.torch_all2all as all2all
from xtuner.v1.module.dispatcher.torch_all2all import _AsyncCombine, _AsyncDispatch, _record_stream


@pytest.mark.parametrize("wrapped", [False, True])
def test_record_stream_does_not_wait_for_collective(wrapped: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real ACT dispatch, stubbing only the CPU stream kernel and collective wait."""
    tensor = torch.ones(4)
    output = funcol.AsyncCollectiveTensor(tensor) if wrapped else tensor
    recorded = []
    waits = []
    monkeypatch.setattr(funcol, "wait_tensor", lambda value: waits.append(value) or value)
    library = torch.library.Library("aten", "IMPL", "CPU")
    try:
        library.impl("record_stream", lambda value, stream: recorded.append(value))
        _record_stream(output, torch.Stream(device="cpu"))
    finally:
        library._destroy()
    assert not waits
    assert len(recorded) == 1 and recorded[0] is tensor
    if wrapped:
        assert not output.completed


@pytest.mark.gpu
class TestAsyncAll2AllStreamLifetime:
    @pytest.mark.parametrize("backward", [_AsyncCombine.backward, _AsyncDispatch.backward])
    def test_backward_output_is_not_reused_before_compute_reads(
        self, backward: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return a communication-stream output only after registering its compute-stream use."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")

        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        compute_stream = torch.cuda.current_stream()
        comm_stream = torch.cuda.Stream()
        grad_output = torch.full((1 << 22,), 1.0, device="cuda", dtype=torch.bfloat16)

        def fake_get_world_size(group: Any = None) -> int:
            return 2

        def fake_all_to_all_single(output: torch.Tensor, input_: torch.Tensor, **kwargs: Any) -> None:
            output.copy_(input_)

        monkeypatch.setattr(dist, "get_world_size", fake_get_world_size)
        monkeypatch.setattr(dist, "all_to_all_single", fake_all_to_all_single)
        ctx = SimpleNamespace(
            input_shape=grad_output.shape,
            comm_stream=comm_stream,
            group=None,
            input_split_sizes=[grad_output.numel() // 2, grad_output.numel() // 2],
            output_split_sizes=[grad_output.numel() // 2, grad_output.numel() // 2],
            backward_previous_event=None,
            backward_finished_event=None,
        )

        result = backward(ctx, grad_output)
        output = result[0]
        comm_done = torch.cuda.Event()
        comm_done.record(comm_stream)
        compute_stream.wait_event(comm_done)
        output_ptr = output.data_ptr()
        sums = [output.sum() for _ in range(2000)]
        compute_done = torch.cuda.Event()
        compute_done.record(compute_stream)
        del output
        del result

        assert not compute_done.query()
        with torch.cuda.stream(comm_stream):
            reused = torch.empty_like(grad_output)

        assert reused.data_ptr() != output_ptr
        compute_stream.synchronize()
        comm_stream.synchronize()
        assert all(value.item() == grad_output.numel() for value in sums)
        torch.cuda.empty_cache()

    def test_combine_saved_activation_is_not_reused_during_backward(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Autograd releases saved forward storage before its queued GPU reads finish."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")

        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        compute_stream = torch.cuda.current_stream()
        comm_stream = torch.cuda.Stream()
        hidden_states = torch.ones(1 << 22, device="cuda", dtype=torch.bfloat16)
        previous = torch.cuda.Event()
        finished = torch.cuda.Event()
        previous.record(compute_stream)

        def fake_collective(value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return funcol.AsyncCollectiveTensor(value.clone())

        monkeypatch.setattr(all2all, "all_to_all_single_autograd", fake_collective)
        output = _AsyncCombine.forward(
            SimpleNamespace(), hidden_states, [], [], previous, finished, None, None, comm_stream, None
        )
        assert not output.completed
        compute_stream.wait_event(finished)
        activation = output._get_acs_underlying_tensor()
        activation_ptr = activation.data_ptr()
        # Warm up reduction allocations and autograd before the asynchronous window.
        assert activation.sum().item() == hidden_states.numel()
        anchor = torch.ones((), device="cuda", requires_grad=True)
        anchor.clone().backward()
        anchor.grad = None
        sums = []

        class SaveActivation(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, anchor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(value)
                return anchor.clone()

            @staticmethod
            def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
                (value,) = ctx.saved_tensors
                # Keep the read queued while autograd releases its saved tensor.
                torch.cuda._sleep(100_000_000)
                sums.append(value.sum())
                return grad, None

        loss = SaveActivation.apply(anchor, activation)
        del activation
        del output
        loss.backward()
        del loss
        compute_done = torch.cuda.Event()
        compute_done.record(compute_stream)
        assert not compute_done.query()

        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            with torch.cuda.stream(comm_stream):
                reused = torch.empty_like(hidden_states)
            # In the regression, this allocation reuses the activation and fills it with NaN.
            assert reused.data_ptr() != activation_ptr
            compute_stream.synchronize()
            comm_stream.synchronize()
            assert all(value.item() == hidden_states.numel() for value in sums)
        finally:
            torch.cuda.synchronize()
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
