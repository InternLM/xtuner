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
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import xtuner.v1.optim.swap_adamw as module


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
    if restore == "optimizer":
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    else:
        options = StateDictOptions(cpu_offload=True)
        state = {"optimizer": get_optimizer_state_dict(model, optimizer, options=options)}
        dcp.save(state, checkpoint_id=tmp_path / "weights")
        state = {"optimizer": get_optimizer_state_dict(restored, resumed, options=options)}
        dcp.load(state, checkpoint_id=tmp_path / "weights")
        set_optimizer_state_dict(restored, resumed, state["optimizer"], options=options)
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
            state = {
                "model": get_model_state_dict(restored, options=options),
                "optimizer": get_optimizer_state_dict(restored, resumed, options=options),
            }
            dcp.load(state, checkpoint_id=checkpoint_dir, planner=InterleavedShardLoadPlanner())
            set_model_state_dict(restored, state["model"], options=options)
            set_optimizer_state_dict(restored, resumed, state["optimizer"], options=options)
            for key in resumed._state_keys:
                value = resumed.state[restored.weight][key]
                if value is not None:
                    assert isinstance(value, DTensor)
                    assert value.to_local().device.type == "cpu"
                    if device == "cuda":
                        assert value.to_local().is_pinned()
                    torch.testing.assert_close(value.to_local(), optimizer.state[model.weight][key].to_local())
                    assert (
                        value.to_local().data_ptr()
                        == resumed._param_to_cpu_states_map[restored.weight][key].data_ptr()
                    )
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
