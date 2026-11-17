"""Small allocation/checkpoint regressions; no model checkpoint is required.

CPU tests mock only pinned allocation and the accelerator boundary. GPU tests can
also run distributed with torchrun --nproc_per_node=2 -m pytest ... -k GPU.
"""

import gc
import importlib
import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.testing._internal.distributed.fake_pg import FakeStore

from xtuner.v1.optim.swap_adamw import SwapAdamW
from xtuner.v1.utils.pinned_memory import plan_pinned_slabs


swap_module = importlib.import_module("xtuner.v1.optim.swap_adamw")


def _rounded(size: int) -> int:
    return 1 << (size - 1).bit_length() if size else 0


def _payload_bytes(value: Any) -> int:
    stream = io.BytesIO()
    torch.save(value, stream)
    with zipfile.ZipFile(stream) as archive:
        return sum(info.file_size for info in archive.infolist() if "/data/" in info.filename)


@pytest.fixture
def cpu_allocation(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    allocations: list[int] = []
    original_empty = torch.empty

    def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.pop("pin_memory", False):
            assert kwargs["device"] == "cpu"
            tensor = original_empty(*args, **kwargs)
            allocations.append(tensor.numel() * tensor.element_size())
            return tensor
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(swap_module, "DEVICE", "cpu")
    monkeypatch.setattr(
        swap_module, "DEVICE_MODULE", SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None)
    )
    return allocations


@pytest.fixture
def fake_cpu_mesh() -> Iterator[DeviceMesh]:
    # The trailing rank covers both uneven and empty shards without a GPU or subprocess.
    dist.init_process_group("fake", rank=1, world_size=2, store=FakeStore())
    try:
        yield init_device_mesh("cpu", (2,))
    finally:
        dist.destroy_process_group()


class TestPinnedSlabPlan:
    def test_deterministic_best_fit_and_bounds(self) -> None:
        sizes = [0, 6, 4, 3, 7, 12, 1]
        plan = plan_pinned_slabs(sizes, cap_bytes=10)
        assert plan == [[5], [4, 3, 0], [1, 2], [6]]
        assert sorted(index for slab in plan for index in slab) == list(range(len(sizes)))
        assert plan_pinned_slabs([], 10) == []
        assert plan_pinned_slabs([0, 0], 10) == [[0, 1]]
        assert plan_pinned_slabs(sizes, 10) == plan

    def test_expert_histogram_reduces_allocator_rounding_without_allocating(self) -> None:
        sizes = [268443648] * 75 + [134217728] * 75 + [317202432] * 2 + [4, 4096, 8192, 0]
        # Two moments, with identical dtype/device, as in the measured shape family.
        sizes *= 2
        plan = plan_pinned_slabs(sizes)
        totals = [sum(sizes[index] for index in slab) for slab in plan]
        independent = sum(map(_rounded, sizes))
        pooled = sum(map(_rounded, totals))
        assert all(total <= 1 << 30 for total in totals)
        assert sum(totals) == sum(sizes)
        assert pooled < 0.8 * independent
        assert pooled < 1.15 * sum(sizes)

    def test_small_total_and_oversized_requests(self) -> None:
        sizes = [16, 8, 4, 0, (1 << 30) + 4]
        plan = plan_pinned_slabs(sizes)
        assert plan == [[4], [0, 1, 2, 3]]
        assert sum(map(_rounded, [sum(sizes[i] for i in slab) for slab in plan])) == (1 << 31) + 32


@pytest.mark.usefixtures("cpu_allocation")
class TestSwapAdamWStorageCPU:
    def test_direct_allocation_layout_aliases_and_amsgrad(self, cpu_allocation: list[int]) -> None:
        params = [
            torch.nn.Parameter(torch.empty(3, 5).t()),
            torch.nn.Parameter(torch.empty(2, 3, 4, 5).to(memory_format=torch.channels_last)),
            torch.nn.Parameter(torch.empty(0, 3)),
            torch.nn.Parameter(torch.empty(6, 8)[:, ::2]),
            torch.nn.Parameter(torch.empty(7, dtype=torch.float64)),
        ]
        optimizer = SwapAdamW([{"params": params[:3], "amsgrad": True}, {"params": params[3:]}])
        assert len(cpu_allocation) == 2  # one small slab per dtype, not one allocation per moment
        assert sum(cpu_allocation) == sum(
            p.numel() * p.element_size() * (3 if i < 3 else 2) for i, p in enumerate(params)
        )
        ranges: dict[int, list[tuple[int, int]]] = {}
        for index, p in enumerate(params):
            for key in optimizer._state_keys:
                state = optimizer.state[p][key]
                assert state is optimizer.state_dict()["state"][index][key]
                if state is None:
                    continue
                assert state.shape == p.shape
                assert state.stride() == torch.empty_like(p).stride()
                assert state.dtype == p.dtype and state.device.type == "cpu"
                assert torch.count_nonzero(state) == 0
                start = state.storage_offset() * state.element_size()
                ranges.setdefault(state.untyped_storage().data_ptr(), []).append(
                    (start, start + state.numel() * state.element_size())
                )
            assert optimizer.state[p]["step"].dtype == torch.float32
            assert optimizer.state[p]["step"].item() == 0
        for spans in ranges.values():
            spans.sort()
            assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        state = optimizer.state[params[0]]["exp_avg"]
        state.fill_(7)
        assert torch.count_nonzero(optimizer.state[params[0]]["exp_avg_sq"]) == 0
        assert next(iter(optimizer.state_dict()["state"].values()))["exp_avg"] is state
        del optimizer
        gc.collect()
        assert torch.all(state == 7)  # surviving view retains the slab

    def test_zero_only_shards(self, cpu_allocation: list[int]) -> None:
        param = torch.nn.Parameter(torch.empty(0, 3))
        optimizer = SwapAdamW([param], amsgrad=True)
        assert cpu_allocation == [0]
        assert all(optimizer.state[param][key].shape == (0, 3) for key in optimizer._state_keys)

    def test_full_save_and_dcp_do_not_multiply_slab_payload(self, tmp_path: Path) -> None:
        params = [torch.nn.Parameter(torch.empty(2048)), torch.nn.Parameter(torch.empty(1024))]
        optimizer = SwapAdamW(params)
        state_dict = optimizer.state_dict()
        moments = {f"{i}.{key}": state_dict["state"][i][key] for i in range(2) for key in ("exp_avg", "exp_avg_sq")}
        for i, tensor in enumerate(moments.values()):
            tensor.fill_(i + 1)
        logical = sum(t.numel() * t.element_size() for t in moments.values())
        assert _payload_bytes(state_dict) == logical + 2 * 4  # step scalars
        # Document raw single-view torch.save's storage contract, rather than hide it with a hot-path clone.
        assert _payload_bytes(next(iter(moments.values()))) == logical
        path = tmp_path / "dcp"
        dcp.save(moments, checkpoint_id=path)
        written = sum(p.stat().st_size for p in path.glob("*.distcp"))
        assert logical <= written < logical + 4096 * len(moments)
        restored = {key: torch.zeros_like(value) for key, value in moments.items()}
        dcp.load(restored, checkpoint_id=path)
        for key, value in moments.items():
            torch.testing.assert_close(restored[key], value)

    @pytest.mark.parametrize("mesh_device", ["cpu", "cuda"])
    def test_cpu_dtensor_metadata_and_local_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mesh_device: str
    ) -> None:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,))
            local = torch.empty(3, 5).t()
            param = torch.nn.Parameter(
                DTensor.from_local(local, mesh, [Shard(0)], shape=local.shape, stride=local.stride(), run_check=False)
            )
            # Simulate an offloaded parameter retaining its accelerator mesh without requiring a GPU.
            monkeypatch.setattr(type(mesh), "device_type", property(lambda _: mesh_device))
            optimizer = SwapAdamW([param])
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.state[param][key]
                assert isinstance(state, DTensor)
                assert state.device.type == "cpu" and state.to_local().device.type == "cpu"
                assert state.device_mesh == param.device_mesh
                assert state.placements == param.placements
                assert state.shape == param.shape and state.stride() == param.stride()
                assert state.to_local().stride() == local.stride()
                assert optimizer.state_dict()["state"][0][key]._local_tensor is state._local_tensor
        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_dtensor_checkpoint_preserves_live_state_and_hooks(
        self, fake_cpu_mesh: DeviceMesh, cpu_allocation: list[int], amsgrad: bool
    ) -> None:
        params = [
            torch.nn.Parameter(distribute_tensor(torch.empty(rows, 3), fake_cpu_mesh, [Shard(0)])) for rows in (5, 1)
        ]
        optimizer = SwapAdamW(params, amsgrad=amsgrad)
        originals = [optimizer.state[param].copy() for param in params]
        allocations = cpu_allocation.copy()
        for state in originals:
            for index, key in enumerate(optimizer._state_keys):
                if state[key] is not None:
                    state[key]._local_tensor.fill_(index + 1)
        assert any(
            isinstance(value, DTensor) and value.storage_offset() > 0
            for state in originals
            for value in state.values()
        )
        events: list[str] = []
        optimizer.register_state_dict_pre_hook(lambda _: events.append("pre-last"))
        optimizer.register_state_dict_pre_hook(lambda _: events.append("pre-first"), prepend=True)

        def replace_post_hook(optim: SwapAdamW, state: dict[str, Any]) -> dict[str, Any]:
            events.append("post-first")
            assert state["state"][0] is optim.state[params[0]]
            return {**state, "hook_marker": 7}

        def inspect_post_hook(optim: SwapAdamW, state: dict[str, Any]) -> None:
            events.append("post-last")
            assert state["hook_marker"] == 7
            assert state["state"][0] is optim.state[params[0]]

        optimizer.register_state_dict_post_hook(inspect_post_hook)
        optimizer.register_state_dict_post_hook(replace_post_hook, prepend=True)
        checkpoint = optimizer.state_dict()
        assert events == ["pre-first", "pre-last", "post-first", "post-last"]
        assert checkpoint["hook_marker"] == 7
        assert cpu_allocation == allocations
        for index, (param, original) in enumerate(zip(params, originals)):
            saved = checkpoint["state"][index]
            assert saved is not optimizer.state[param]
            for key, value in original.items():
                assert optimizer.state[param][key] is value
                if isinstance(value, DTensor):
                    assert saved[key]._spec is value._spec
                    assert saved[key]._local_tensor is value._local_tensor
                    assert saved[key].storage_offset() == 0
                else:
                    assert saved[key] is value
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        assert restored["param_groups"] == checkpoint["param_groups"]
        assert restored["hook_marker"] == 7
        for index, original in enumerate(originals):
            for key, value in original.items():
                actual = restored["state"][index][key]
                if isinstance(value, DTensor):
                    torch.testing.assert_close(actual.to_local(), value.to_local())
                    assert actual.shape == value.shape and actual.stride() == value.stride()
                    assert actual.placements == value.placements and actual.device_mesh == value.device_mesh
                    assert actual._local_tensor.storage_offset() == value._local_tensor.storage_offset()
                elif value is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(actual, value)
        saved_moments = restored["state"][0]
        assert (
            saved_moments["exp_avg"]._local_tensor.untyped_storage()
            is saved_moments["exp_avg_sq"]._local_tensor.untyped_storage()
        )
        # Restore the serialized checkpoint through main's in-place loader, not just torch.load.
        for state in optimizer.state.values():
            for value in state.values():
                if value is not None:
                    optimizer._to_local_tensor(value).zero_()
        events.clear()
        optimizer.register_load_state_dict_pre_hook(lambda optim, state: events.append("load-pre"))
        optimizer.register_load_state_dict_post_hook(lambda optim: events.append("load-post"))
        optimizer.load_state_dict(restored)
        assert events == ["load-pre", "load-post"]
        assert cpu_allocation == allocations
        for index, (param, original) in enumerate(zip(params, originals)):
            for key, value in original.items():
                assert optimizer.state[param][key] is value
                if value is not None:
                    torch.testing.assert_close(
                        optimizer._to_local_tensor(value), optimizer._to_local_tensor(restored["state"][index][key])
                    )

    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_multiple_steps_match_adamw(self, amsgrad: bool) -> None:
        _check_math("cpu", amsgrad)

    def test_integration_load_preserves_state_and_group_identity(self) -> None:
        params = [torch.nn.Parameter(torch.empty(3, 5).t()), torch.nn.Parameter(torch.empty(0, 3))]
        optimizer = SwapAdamW(params, amsgrad=True)
        originals = [optimizer.state[p].copy() for p in params]
        group = optimizer.param_groups[0]
        incoming = optimizer.state_dict()
        incoming["state"] = {
            i: {key: torch.full_like(value, 2) for key, value in state.items()}
            for i, state in incoming["state"].items()
        }
        incoming["param_groups"] = [{**group, "params": [0, 1], "lr": 0.025}]
        optimizer.load_state_dict(incoming)
        assert optimizer.param_groups[0] is group and group["lr"] == 0.025
        for p, original in zip(params, originals):
            for key, value in original.items():
                assert optimizer.state[p][key] is value
                assert torch.all(value == 2)
        optimizer.load_state_dict({**incoming, "state": {}})
        assert all(torch.count_nonzero(t) == 0 for state in optimizer.state.values() for t in state.values())


def _check_math(device: str, amsgrad: bool) -> None:
    torch.manual_seed(42)
    params = [
        torch.nn.Parameter(torch.randn(3, 5, device=device).t()),
        torch.nn.Parameter(torch.randn(17, device=device)),
    ]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    kwargs = {"lr": 0.003, "amsgrad": amsgrad, "foreach": False}
    optimizer = SwapAdamW(params, **kwargs)
    expected = torch.optim.AdamW(reference, **kwargs)
    for step in range(4):
        if step == 2:
            stream = io.BytesIO()
            torch.save(optimizer.state_dict(), stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=False)
            originals = [optimizer.state[p].copy() for p in params]
            optimizer.load_state_dict({**restored, "state": {}})
            optimizer.load_state_dict(restored)
            for p, original in zip(params, originals):
                assert all(optimizer.state[p][key] is value for key, value in original.items())
        for i, (p, q) in enumerate(zip(params, reference)):
            p.grad = None if step == 1 and i == 0 else torch.randn_like(p)
            q.grad = None if p.grad is None else p.grad.clone()
        optimizer.step()
        expected.step()
        for p, q in zip(params, reference):
            torch.testing.assert_close(p, q)
            for key, value in expected.state[q].items():
                torch.testing.assert_close(optimizer.state[p][key], value.cpu())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSwapAdamWStorageGPU:
    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_multiple_steps_and_real_pinning(self, amsgrad: bool) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        _check_math("cuda", amsgrad)
        optimizer = SwapAdamW([torch.nn.Parameter(torch.empty(16, device="cuda"))])
        assert optimizer.state_dict()["state"][0]["exp_avg"].is_pinned()

    def test_nondefault_stream_and_allocator_lifetime(self) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        optimizer = SwapAdamW([torch.nn.Parameter(torch.empty(1 << 16, device="cuda"))])
        # Pick the interior moment: base pointers alone cannot protect its allocator lifetime.
        source = next(iter(optimizer.state.values()))["exp_avg_sq"]
        assert source.is_pinned() and source.storage_offset() > 0
        source.fill_(7)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            copied = source.to("cuda", non_blocking=True)
        del source, optimizer
        gc.collect()
        # Exercise host cache reuse while the transfer is pending.
        churn = [torch.empty(1 << 17, pin_memory=True).fill_(3) for _ in range(4)]
        stream.synchronize()
        assert torch.all(copied == 7)
        del churn

    def test_dtensor_uneven_and_empty_shards(self, tmp_path: Path) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cuda", (dist.get_world_size(),))
            # Reuse one process group: destroy/reinitialize with the same torchrun store can race across ranks.
            for rows in (1, 5):
                param = torch.nn.Parameter(distribute_tensor(torch.ones(rows, 3, device="cuda"), mesh, [Shard(0)]))
                optimizer = SwapAdamW([param], amsgrad=True, foreach=False)
                reference = torch.nn.Parameter(param.to_local().detach().clone())
                expected = torch.optim.AdamW([reference], amsgrad=True, foreach=False)
                for _ in range(3):
                    param.grad = torch.ones_like(param)
                    reference.grad = torch.ones_like(reference)
                    optimizer.step()
                    expected.step()
                    torch.testing.assert_close(param.to_local(), reference)
                    for key in optimizer._state_keys:
                        state = optimizer.state[param][key]
                        assert state.shape == param.shape and state.placements == param.placements
                        assert state.to_local().shape == param.to_local().shape
                        torch.testing.assert_close(state.to_local(), expected.state[reference][key].cpu())
        finally:
            dist.destroy_process_group()
