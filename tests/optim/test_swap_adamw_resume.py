"""SwapAdamW must retain CPU canonical state and exact next-update semantics."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

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
