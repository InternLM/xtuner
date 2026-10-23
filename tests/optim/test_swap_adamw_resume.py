"""SwapAdamW must retain CPU canonical state and exact next-update semantics."""

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import xtuner.v1.optim.swap_adamw as module


def _capture_buffers(optimizer):
    return {param: dict(state) for param, state in optimizer.state.items()}


def _assert_buffers_unchanged(optimizer, buffers):
    for param, state in buffers.items():
        for key, value in state.items():
            assert optimizer.state[param][key] is value
            if value is not None and key != "step":
                assert (
                    optimizer._param_to_cpu_states_map[param][key].data_ptr()
                    == optimizer._to_local_tensor(value).data_ptr()
                )


class _NoNewStorage(TorchDispatchMode):
    def __init__(self, *states):
        super().__init__()
        self.pointers = {
            (value.device, value.untyped_storage().data_ptr())
            for state in states
            for value in tree_leaves(state)
            if isinstance(value, torch.Tensor)
        }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor):
                assert (value.device, value.untyped_storage().data_ptr()) in self.pointers, func
        return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("restore", ["optimizer", "dcp"])
def test_resume_matches_continuous_update(tmp_path, monkeypatch, device, amsgrad, restore):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(module, "DEVICE", device)
    if device == "cpu":
        monkeypatch.setattr(
            module, "DEVICE_MODULE", SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None)
        )
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    else:
        monkeypatch.setattr(module, "DEVICE_MODULE", torch.cuda)
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 2, bias=False, device=device)
    reference = copy.deepcopy(model)
    optimizer = module.SwapAdamW(model.parameters(), lr=0.1, amsgrad=amsgrad, fused=False)
    standard = torch.optim.AdamW(reference.parameters(), lr=0.1, amsgrad=amsgrad, fused=False)
    for i in range(3):
        for m, opt in ((model, optimizer), (reference, standard)):
            m.weight.grad = torch.full_like(m.weight, (i + 1) * 0.25)
            opt.step()
    optimizer.param_groups[0]["lr"] = standard.param_groups[0]["lr"] = 0.025
    restored = copy.deepcopy(model)
    resumed = module.SwapAdamW(restored.parameters(), lr=0.2, amsgrad=amsgrad, fused=False)
    buffers = _capture_buffers(resumed)
    if restore == "optimizer":
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    else:
        options = StateDictOptions(cpu_offload=True)
        state = {"optimizer": get_optimizer_state_dict(model, optimizer, options=options)}
        dcp.save(state, checkpoint_id=tmp_path / "weights")
        state = {"optimizer": get_optimizer_state_dict(restored, resumed, options=options)}
        dcp.load(state, checkpoint_id=tmp_path / "weights")
        set_optimizer_state_dict(restored, resumed, state["optimizer"], options=options)
    _assert_buffers_unchanged(resumed, buffers)
    p = restored.weight
    assert resumed._param_to_group_map[p] is resumed.param_groups[0]
    for key in resumed._state_keys:
        value = resumed.state[p][key]
        assert resumed._param_to_cpu_states_map[p][key] is value
        if value is not None:
            assert value.device.type == "cpu"
            if device == "cuda":
                assert value.is_pinned()
            torch.testing.assert_close(value, optimizer.state[model.weight][key])
    assert resumed.state[p]["step"].device.type == "cpu"
    assert resumed.state[p]["step"].item() == 3
    for m, opt in ((model, optimizer), (reference, standard), (restored, resumed)):
        m.weight.grad = torch.full_like(m.weight, -0.75)
        opt.step()
    torch.testing.assert_close(restored.weight, reference.weight)
    torch.testing.assert_close(model.weight, reference.weight)
    assert resumed.state[p]["step"].item() == 4
    for key in resumed._state_keys:
        if resumed.state[p][key] is not None:
            torch.testing.assert_close(resumed.state[p][key], optimizer.state[model.weight][key])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_state_dict_never_stages_moments_on_device(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(module, "DEVICE", device)
    if device == "cpu":
        monkeypatch.setattr(
            module, "DEVICE_MODULE", SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None)
        )
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    else:
        monkeypatch.setattr(module, "DEVICE_MODULE", torch.cuda)
    torch.manual_seed(7)
    model = torch.nn.Linear(5, 3, bias=False, device=device)
    optimizer = module.SwapAdamW(model.parameters(), lr=0.1, fused=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()

    staged = []
    original = torch.optim.Optimizer._process_value_according_to_param_policy

    def spy(param, value, param_id, param_groups, key=None):
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            staged.append((param_id, key))
        return original(param, value, param_id, param_groups, key=key)

    monkeypatch.setattr(torch.optim.Optimizer, "_process_value_according_to_param_policy", staticmethod(spy))
    restored = copy.deepcopy(model)
    resumed = module.SwapAdamW(restored.parameters(), lr=0.1, fused=False)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    assert staged == []
    p = restored.weight
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[p][key], optimizer.state[model.weight][key])
    assert resumed._param_to_cpu_states_map[p]["exp_avg"] is resumed.state[p]["exp_avg"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [(torch.float32, torch.bfloat16), (torch.float32, torch.float16), (torch.bfloat16, torch.float32)],
)
def test_cross_precision_resume_matches_adamw(monkeypatch, device, amsgrad, source_dtype, target_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(module, "DEVICE", device)
    if device == "cpu":
        monkeypatch.setattr(
            module, "DEVICE_MODULE", SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None)
        )
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    else:
        monkeypatch.setattr(module, "DEVICE_MODULE", torch.cuda)
    source = torch.nn.Parameter(torch.arange(6, device=device, dtype=source_dtype).reshape(2, 3) / 6)
    optimizer = module.SwapAdamW([source], lr=0.01, amsgrad=amsgrad, fused=False)
    for i in range(3):
        source.grad = torch.full_like(source, (i + 1) * 0.25)
        optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored = torch.nn.Parameter(source.detach().to(dtype=target_dtype))
    reference = torch.nn.Parameter(restored.detach().clone())
    resumed = module.SwapAdamW([restored], amsgrad=amsgrad, fused=False)
    standard = torch.optim.AdamW([reference], amsgrad=amsgrad, fused=False)
    buffers = _capture_buffers(resumed)
    with _NoNewStorage(checkpoint, resumed.state_dict()):
        resumed.load_state_dict(checkpoint)
    _assert_buffers_unchanged(resumed, buffers)
    standard.load_state_dict(copy.deepcopy(checkpoint))
    for key in resumed._state_keys:
        value = resumed.state[restored][key]
        if value is not None:
            assert value.dtype == target_dtype
            assert value.device.type == "cpu"
            if device == "cuda":
                assert value.is_pinned()
            torch.testing.assert_close(value, standard.state[reference][key].cpu())
    for param, opt in ((restored, resumed), (reference, standard)):
        param.grad = torch.full_like(param, -0.75)
        opt.step()
    torch.testing.assert_close(restored, reference)
    for key in resumed._state_keys:
        if resumed.state[restored][key] is not None:
            torch.testing.assert_close(resumed.state[restored][key], standard.state[reference][key].cpu())
    assert resumed.state[restored]["step"].item() == 4


def _sharded_resume_worker(rank, checkpoint_dir, rendezvous, amsgrad, asynchronous, device):
    if device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl" if device == "cuda" else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "DEVICE", device))
            if device == "cpu":
                stack.enter_context(
                    patch.object(
                        module, "DEVICE_MODULE", SimpleNamespace(current_device=lambda: rank, synchronize=lambda: None)
                    )
                )
                stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
                # Emulate pinning on CPU-only hosts; CUDA cases check real pinned storage.
                stack.enter_context(patch.object(torch.Tensor, "is_pinned", lambda self: True))
                empty_like = torch.empty_like

                def unpinned_empty_like(*args, **kwargs):
                    kwargs.pop("pin_memory", None)
                    return empty_like(*args, **kwargs)

                stack.enter_context(patch.object(torch, "empty_like", unpinned_empty_like))
            else:
                stack.enter_context(patch.object(module, "DEVICE_MODULE", torch.cuda))
            mesh = init_device_mesh(device, (2,))
            model = torch.nn.Module()
            model.register_parameter(
                "weight",
                torch.nn.Parameter(
                    distribute_tensor(
                        torch.arange(35, dtype=torch.float32, device=device).reshape(7, 5) / 35, mesh, [Shard(0)]
                    )
                ),
            )
            optimizer = module.SwapAdamW(model.parameters(), lr=0.01, amsgrad=amsgrad, fused=False)
            for i in range(3):
                model.weight.grad = torch.ones_like(model.weight) * ((rank + 1) * (i + 1) * 0.25)
                optimizer.step()
            options = StateDictOptions(cpu_offload=True)
            state = {
                "model": get_model_state_dict(model, options=options),
                "optimizer": get_optimizer_state_dict(model, optimizer, options=options),
            }
            from xtuner.v1.patch import InterleavedShardLoadPlanner, InterleavedShardSavePlanner

            kwargs = dict(checkpoint_id=checkpoint_dir, planner=InterleavedShardSavePlanner())
            if asynchronous:
                dcp.async_save(state, **kwargs).result(timeout=60)
            else:
                dcp.save(state, **kwargs)
            metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
            for key in ("exp_avg", "exp_avg_sq"):
                saved = metadata.state_dict_metadata[f"optimizer.state.weight.{key}"]
                assert saved.size == model.weight.shape
                assert len(saved.chunks) == 2
            restored = copy.deepcopy(model)
            resumed = module.SwapAdamW(restored.parameters(), lr=0.2, amsgrad=amsgrad, fused=False)
            buffers = _capture_buffers(resumed)
            state = {
                "model": get_model_state_dict(restored, options=options),
                "optimizer": get_optimizer_state_dict(restored, resumed, options=options),
            }
            dcp.load(state, checkpoint_id=checkpoint_dir, planner=InterleavedShardLoadPlanner())
            loaded_moments = state["optimizer"]["state"]["weight"]
            loaded_ptrs = {
                key: value.to_local().data_ptr()
                for key, value in loaded_moments.items()
                if key in resumed._state_keys and value is not None
            }
            set_model_state_dict(restored, state["model"], options=options)
            set_optimizer_state_dict(restored, resumed, state["optimizer"], options=options)
            _assert_buffers_unchanged(resumed, buffers)
            for key in resumed._state_keys:
                value = resumed.state[restored.weight][key]
                if value is not None:
                    assert isinstance(value, DTensor)
                    assert value.to_local().device.type == "cpu"
                    if device == "cuda":
                        assert value.to_local().is_pinned()
                    assert value.to_local().data_ptr() == loaded_ptrs[key]
                    torch.testing.assert_close(value.to_local(), optimizer.state[model.weight][key].to_local())
                    assert (
                        value.to_local().data_ptr()
                        == resumed._param_to_cpu_states_map[restored.weight][key].data_ptr()
                    )
            direct_state = copy.deepcopy(resumed.state_dict())
            resumed.load_state_dict(direct_state)
            _assert_buffers_unchanged(resumed, buffers)
            bad_state = copy.deepcopy(direct_state)
            bad_moment = bad_state["state"][0]["exp_avg"]
            bad_state["state"][0]["exp_avg"] = DTensor.from_local(
                bad_moment.to_local(),
                mesh,
                [Replicate()],
                run_check=False,
                shape=bad_moment.shape,
                stride=bad_moment.stride(),
            )
            with pytest.raises(ValueError, match="mesh or placements"):
                resumed.load_state_dict(bad_state)
            _assert_buffers_unchanged(resumed, buffers)
            for m, opt in ((model, optimizer), (restored, resumed)):
                m.weight.grad = torch.ones_like(m.weight) * -(rank + 1)
                opt.step()
            torch.testing.assert_close(restored.weight.to_local(), model.weight.to_local())
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    resumed.state[restored.weight][key].to_local(), optimizer.state[model.weight][key].to_local()
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sharded_checkpoint_keeps_every_rank_moment(tmp_path, amsgrad, asynchronous, device):
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    mp.spawn(
        _sharded_resume_worker,
        args=(str(tmp_path / "weights"), str(tmp_path / "rendezvous"), amsgrad, asynchronous, device),
        nprocs=2,
    )


def test_incomplete_optimizer_checkpoint_is_rejected_before_loading():
    from xtuner.v1.engine.train_engine import _validate_optimizer_checkpoint_shapes

    state = {"state": {"weight": {"exp_avg": torch.empty(8, 3), "step": torch.tensor(4.0)}}}
    metadata = SimpleNamespace(
        state_dict_metadata={
            "optimizer.state.weight.exp_avg": SimpleNamespace(size=torch.Size([2, 3])),
            "optimizer.state.weight.step": SimpleNamespace(size=torch.Size([])),
        }
    )
    with pytest.raises(ValueError, match="lost distributed shards"):
        _validate_optimizer_checkpoint_shapes(state, metadata)
    metadata.state_dict_metadata["optimizer.state.weight.exp_avg"].size = torch.Size([8, 3])
    _validate_optimizer_checkpoint_shapes(state, metadata)


@pytest.mark.parametrize("incomplete", [False, True])
def test_load_dcp_reads_metadata_once_and_validates_before_tensor_reads(tmp_path, monkeypatch, incomplete):
    from xtuner.v1.engine.train_engine import TrainEngine

    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cpu, "reset_peak_memory_stats", lambda: None, raising=False)
        monkeypatch.setattr(torch.cpu, "max_memory_allocated", lambda: 0, raising=False)
    model = torch.nn.Linear(3, 8, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), fused=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    options = StateDictOptions(cpu_offload=True)
    state = {
        "model": get_model_state_dict(model, options=options),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=options),
    }
    saved = copy.deepcopy(state)
    if incomplete:
        saved["optimizer"]["state"]["weight"]["exp_avg"] = torch.ones(2, 3)
    checkpoint_dir = tmp_path / "weights"
    dcp.save(saved, checkpoint_id=checkpoint_dir)
    with torch.no_grad():
        model.weight.zero_()
        optimizer.state[model.weight]["exp_avg"].zero_()

    calls = {"metadata": 0, "data": 0}
    read_metadata = dcp.FileSystemReader.read_metadata
    read_data = dcp.FileSystemReader.read_data

    def counted_metadata(reader, *args, **kwargs):
        calls["metadata"] += 1
        return read_metadata(reader, *args, **kwargs)

    def counted_data(reader, *args, **kwargs):
        calls["data"] += 1
        return read_data(reader, *args, **kwargs)

    monkeypatch.setattr(dcp.FileSystemReader, "read_metadata", counted_metadata)
    monkeypatch.setattr(dcp.FileSystemReader, "read_data", counted_data)
    engine = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        has_freeze_params=False,
        _get_dcp_state_dict=lambda **kwargs: state,
    )
    if incomplete:
        # DCP wraps planner failures in a collective CheckpointException.
        with pytest.raises(dcp.CheckpointException, match="lost distributed shards"):
            TrainEngine.load_dcp(engine, checkpoint_dir)
        assert calls == {"metadata": 1, "data": 0}
        assert torch.count_nonzero(model.weight) == 0
        assert torch.count_nonzero(optimizer.state[model.weight]["exp_avg"]) == 0
    else:
        TrainEngine.load_dcp(engine, checkpoint_dir)
        assert calls == {"metadata": 1, "data": 1}
        torch.testing.assert_close(model.weight, saved["model"]["weight"])
        torch.testing.assert_close(
            optimizer.state[model.weight]["exp_avg"], saved["optimizer"]["state"]["weight"]["exp_avg"]
        )


@pytest.fixture
def cpu_optimizer(monkeypatch):
    monkeypatch.setattr(module, "DEVICE", "cpu")
    monkeypatch.setattr(module, "DEVICE_MODULE", SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None))
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    params = [torch.nn.Parameter(torch.ones(2, 3)) for _ in range(2)]
    optimizer = module.SwapAdamW([{"params": [p], "lr": 0.01 * (i + 1)} for i, p in enumerate(params)], fused=False)
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    return optimizer


@pytest.mark.parametrize(
    "invalid",
    [
        "groups",
        "params",
        "duplicate_ids",
        "missing_param",
        "extra_param",
        "missing_moment",
        "extra_moment",
        "shape",
        "type",
        "dtype",
        "step_shape",
        "amsgrad",
    ],
)
def test_invalid_checkpoint_does_not_partially_update(cpu_optimizer, invalid):
    optimizer = cpu_optimizer
    before = copy.deepcopy(optimizer.state_dict())
    incoming = copy.deepcopy(before)
    incoming["param_groups"][0]["lr"] = 0.5
    incoming["state"][0]["exp_avg"].fill_(123)
    # Corrupt the second parameter to catch validation interleaved with writes.
    state = incoming["state"][1]
    if invalid == "groups":
        incoming["param_groups"].pop()
    elif invalid == "params":
        incoming["param_groups"][1]["params"].append(2)
    elif invalid == "duplicate_ids":
        incoming["param_groups"][1]["params"] = [0]
    elif invalid == "missing_param":
        del incoming["state"][1]
    elif invalid == "extra_param":
        incoming["state"][2] = copy.deepcopy(state)
    elif invalid == "missing_moment":
        del state["exp_avg_sq"]
    elif invalid == "extra_moment":
        state["unexpected"] = None
    elif invalid == "shape":
        state["exp_avg_sq"] = torch.ones(1, 3)  # Broadcastable, but invalid.
    elif invalid == "type":
        state["exp_avg_sq"] = [1, 2]
    elif invalid == "dtype":
        state["exp_avg_sq"] = torch.ones(2, 3, dtype=torch.int64)
    elif invalid == "step_shape":
        state["step"] = torch.ones(2)
    elif invalid == "amsgrad":
        incoming["param_groups"][1]["amsgrad"] = True
    buffers = _capture_buffers(optimizer)
    with pytest.raises(ValueError):
        optimizer.load_state_dict(incoming)
    _assert_buffers_unchanged(optimizer, buffers)
    after = optimizer.state_dict()
    assert after["param_groups"] == before["param_groups"]
    for param_id, values in before["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(after["state"][param_id][key], value)


def test_empty_state_resets_in_place_and_preserves_hooks(cpu_optimizer, monkeypatch):
    optimizer = cpu_optimizer
    buffers = _capture_buffers(optimizer)
    groups = list(optimizer.param_groups)
    state_object = optimizer.state
    incoming = copy.deepcopy(optimizer.state_dict())
    incoming["state"] = {}
    events = []

    def pre_hook(opt, state):
        events.append("pre")
        replacement = copy.deepcopy(state)
        replacement["param_groups"][0]["lr"] = 0.125
        return replacement

    def post_hook(opt):
        events.append("post")
        assert opt.param_groups[0]["lr"] == 0.125
        for state in opt.state.values():
            for value in state.values():
                if value is not None:
                    assert torch.count_nonzero(value) == 0

    def forbidden_init():
        pytest.fail("load_state_dict must not reinitialize buffers")

    monkeypatch.setattr(optimizer, "_init_swap_states", forbidden_init)
    optimizer.register_load_state_dict_pre_hook(pre_hook)
    optimizer.register_load_state_dict_post_hook(post_hook)
    optimizer.load_state_dict(incoming)
    assert events == ["pre", "post"]
    assert optimizer.state is state_object
    for group, original in zip(optimizer.param_groups, groups):
        assert group is original
        assert optimizer._param_to_group_map[group["params"][0]] is group
    _assert_buffers_unchanged(optimizer, buffers)
    assert incoming["param_groups"][0]["lr"] == 0.01
