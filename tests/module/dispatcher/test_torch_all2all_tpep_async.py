import pytest
import torch
import torch.distributed as dist

from xtuner.v1.module.dispatcher import torch_all2all
from xtuner.v1.module.dispatcher.torch_all2all_tpep import (
    TorchAll2AllTPEPDispatcher,
    _async_tp_all_gather,
    _async_tp_reduce_scatter_sum,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for stream assertions.")


class _FakeTPGroup:
    def __init__(self, size: int = 2, rank: int = 0) -> None:
        self._size = size
        self.rank = rank

    def size(self) -> int:
        return self._size


class _FakeEPGroup(_FakeTPGroup):
    pass


def _stream_id() -> int:
    return torch.cuda.current_stream().cuda_stream


def test_async_tpep_dispatch_returns_tp_gathered_payload(monkeypatch) -> None:
    dispatcher = TorchAll2AllTPEPDispatcher(
        n_routed_experts=4,
        ep_group=_FakeEPGroup(size=1),  # type: ignore[arg-type]
        tp_group=_FakeTPGroup(size=2),  # type: ignore[arg-type]
    )

    def fake_get_rank(group=None) -> int:
        return getattr(group, "rank", 0)

    def fake_all_to_all_single(output, input, *args, **kwargs) -> None:
        output.copy_(input)

    def fake_ep_all_to_all_single_autograd(input, *args, **kwargs):
        return input.clone()

    def fake_all_gather_into_tensor(output, input, group=None) -> None:
        if output.numel() == 2 and input.numel() == 1:
            output.fill_(input.item())
        else:
            output[0].copy_(input)
            output[1].copy_(input)

    def fake_all_gather(chunks, tensor, group=None) -> None:
        chunks[0].copy_(tensor)
        chunks[1].copy_(tensor + 10)

    monkeypatch.setattr(dist, "get_rank", fake_get_rank)
    monkeypatch.setattr(dist, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setattr(torch_all2all, "all_to_all_single_autograd", fake_ep_all_to_all_single_autograd)
    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    hidden = torch.randn(32, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    topk_ids = torch.randint(0, 4, (32, 1), device="cuda", dtype=torch.float32)
    topk_weights = torch.ones(32, 1, device="cuda", dtype=torch.float32)
    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, async_op=True)

    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        async_op=True,
    )
    torch.cuda.current_stream().wait_event(dispatched["forward_finished_event"])
    torch.cuda.synchronize()

    # 中文注释：TP 通信的归属边界是 dispatch，postprocess 只能看到已经 gather 好的 token。
    assert dispatched["hidden_states"].shape == (64, 128)
    assert dispatched["output_splits_tp"] == [32, 32]
    torch.testing.assert_close(dispatched["hidden_states"][32:], pre_dispatched["hidden_states"] + 10)


def test_async_tpep_combine_owns_tp_reduce_scatter(monkeypatch) -> None:
    dispatcher = TorchAll2AllTPEPDispatcher(
        n_routed_experts=4,
        ep_group=_FakeEPGroup(size=1),  # type: ignore[arg-type]
        tp_group=_FakeTPGroup(size=2),  # type: ignore[arg-type]
    )

    def fake_get_rank(group=None) -> int:
        return getattr(group, "rank", 0)

    def fake_all_to_all_single(output, input, *args, **kwargs) -> None:
        output.copy_(input)

    def fake_ep_all_to_all_single_autograd(input, *args, **kwargs):
        return input.clone()

    def fake_all_gather_into_tensor(output, input, group=None) -> None:
        if output.numel() == 2 and input.numel() == 1:
            output.fill_(input.item())
        else:
            output[0].copy_(input)
            output[1].copy_(input)

    def fake_reduce_scatter_tensor(output, input, op=None, group=None) -> None:
        output.copy_(input[: output.shape[0]])

    def fake_reduce_scatter(output, input_list, op=None, group=None) -> None:
        output.copy_(input_list[getattr(group, "rank", 0)])

    def fake_all_reduce(tensor, op=None, group=None) -> None:
        raise AssertionError("TP ReduceScatterSum should not use all_reduce + slice")

    def fake_all_gather(chunks, tensor, group=None) -> None:
        chunks[0].copy_(tensor)
        chunks[1].copy_(tensor + 10)

    monkeypatch.setattr(dist, "get_rank", fake_get_rank)
    monkeypatch.setattr(dist, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setattr(torch_all2all, "all_to_all_single_autograd", fake_ep_all_to_all_single_autograd)
    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)
    monkeypatch.setattr(dist, "reduce_scatter", fake_reduce_scatter)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    hidden = torch.randn(32, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    topk_ids = torch.randint(0, 4, (32, 1), device="cuda", dtype=torch.float32)
    topk_weights = torch.ones(32, 1, device="cuda", dtype=torch.float32)
    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, async_op=True)
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        async_op=True,
    )
    post_dispatched = dispatcher.dispatch_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        async_op=True,
    )

    pre_combined = dispatcher.combine_preprocess(
        hidden_states=post_dispatched["hidden_states"],
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        async_op=True,
    )
    torch.cuda.current_stream().wait_event(pre_combined["forward_finished_event"])
    torch.cuda.synchronize()

    # 中文注释：preprocess 只做本地 layout，还保持 TP-gather 后的完整 token 数。
    assert pre_combined["hidden_states"].shape == (64, 128)

    combined = dispatcher.combine(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        async_op=True,
    )
    torch.cuda.current_stream().wait_event(combined["forward_finished_event"])
    torch.cuda.synchronize()

    # 中文注释：TP ReduceScatter 属于 combine，combine 后才回到当前 TP rank 的 token slice。
    assert combined["hidden_states"].shape == (32, 128)


def test_async_tp_all_gather_uses_comm_stream(monkeypatch) -> None:
    comm_stream = torch.cuda.Stream()
    group = _FakeTPGroup()
    calls: list[tuple[str, int]] = []

    def fake_get_rank(group=None) -> int:
        return getattr(group, "rank", 0)

    def fake_all_gather(chunks, tensor, group=None) -> None:
        calls.append(("all_gather", _stream_id()))
        for chunk in chunks:
            chunk.copy_(tensor[: chunk.shape[0]])

    def fake_reduce_scatter_tensor(output, input, op=None, group=None) -> None:
        calls.append(("reduce_scatter_tensor", _stream_id()))
        output.copy_(input[: output.shape[0]])

    def fake_all_reduce(tensor, op=None, group=None) -> None:
        raise AssertionError("TP AllGather backward should use reduce_scatter")

    monkeypatch.setattr(dist, "get_rank", fake_get_rank)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    hidden = torch.randn(2, 3, device="cuda", requires_grad=True)
    forward_previous_event = torch.cuda.Event()
    forward_finished_event = torch.cuda.Event()
    backward_previous_event = torch.cuda.Event()
    backward_finished_event = torch.cuda.Event()
    forward_previous_event.record()

    out = _async_tp_all_gather(
        hidden,
        all_sizes=[2, 2],
        tp_group=group,  # type: ignore[arg-type]
        forward_previous_event=forward_previous_event,
        forward_finished_event=forward_finished_event,
        backward_previous_event=backward_previous_event,
        backward_finished_event=backward_finished_event,
        comm_stream=comm_stream,
    )
    torch.cuda.current_stream().wait_event(forward_finished_event)
    loss = out.sum()

    # 中文注释：直接调用私有 helper 时没有 dispatcher hook，这里手动模拟梯度已就绪事件。
    backward_previous_event.record()
    loss.backward()
    torch.cuda.current_stream().wait_event(backward_finished_event)
    torch.cuda.synchronize()

    assert hidden.grad is not None
    assert calls == [
        ("all_gather", comm_stream.cuda_stream),
        ("reduce_scatter_tensor", comm_stream.cuda_stream),
    ]


def test_async_tp_reduce_scatter_uses_comm_stream(monkeypatch) -> None:
    comm_stream = torch.cuda.Stream()
    group = _FakeTPGroup()
    calls: list[tuple[str, int]] = []

    def fake_get_rank(group=None) -> int:
        return getattr(group, "rank", 0)

    def fake_reduce_scatter(output, input_list, op=None, group=None) -> None:
        calls.append(("reduce_scatter", _stream_id()))
        output.copy_(input_list[getattr(group, "rank", 0)])

    def fake_all_reduce(tensor, op=None, group=None) -> None:
        raise AssertionError("TP ReduceScatterSum should use reduce_scatter")

    def fake_all_gather(chunks, tensor, group=None) -> None:
        calls.append(("all_gather", _stream_id()))
        for chunk in chunks:
            chunk.copy_(tensor[:1].expand_as(chunk))

    monkeypatch.setattr(dist, "get_rank", fake_get_rank)
    monkeypatch.setattr(dist, "reduce_scatter", fake_reduce_scatter)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    hidden = torch.randn(4, 3, device="cuda", requires_grad=True)
    forward_previous_event = torch.cuda.Event()
    forward_finished_event = torch.cuda.Event()
    backward_previous_event = torch.cuda.Event()
    backward_finished_event = torch.cuda.Event()
    forward_previous_event.record()

    out = _async_tp_reduce_scatter_sum(
        hidden,
        all_sizes=[1, 3],
        tp_group=group,  # type: ignore[arg-type]
        forward_previous_event=forward_previous_event,
        forward_finished_event=forward_finished_event,
        backward_previous_event=backward_previous_event,
        backward_finished_event=backward_finished_event,
        comm_stream=comm_stream,
    )
    torch.cuda.current_stream().wait_event(forward_finished_event)
    loss = out.sum()

    backward_previous_event.record()
    loss.backward()
    torch.cuda.current_stream().wait_event(backward_finished_event)
    torch.cuda.synchronize()

    assert hidden.grad is not None
    assert calls == [
        ("reduce_scatter", comm_stream.cuda_stream),
        ("all_gather", comm_stream.cuda_stream),
    ]
