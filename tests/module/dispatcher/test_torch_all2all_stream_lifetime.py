from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from xtuner.v1.module.dispatcher.torch_all2all import _AsyncCombine, _AsyncDispatch


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
