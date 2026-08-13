from __future__ import annotations

import tempfile

import pytest
import torch
import torch.distributed.checkpoint as dcp
from packaging.version import Version
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from xtuner.v1.engine.train_engine import _prune_uncheckpointed_optimizer_state


@pytest.mark.skipif(
    Version(torch.__version__) < Version("2.10"), reason="requires sparse optimizer loading in PyTorch 2.10+"
)
def test_sparse_adamw_dcp_state_can_resume_and_materialize_later() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.mean().backward()
    optimizer.step()

    saved_state = {
        "model": get_model_state_dict(model),
        "optimizer": get_optimizer_state_dict(model, optimizer),
    }

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        dcp.save(saved_state, checkpoint_id=checkpoint_dir)

        restored_model = torch.nn.Linear(4, 2)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters())
        load_target = {
            "model": get_model_state_dict(restored_model),
            "optimizer": get_optimizer_state_dict(restored_model, restored_optimizer),
        }
        metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
        saved_keys = {str(key) for key in metadata.state_dict_metadata}

        removed = _prune_uncheckpointed_optimizer_state(load_target["optimizer"], saved_keys)
        assert set(removed) == {
            "optimizer.state.bias.step",
            "optimizer.state.bias.exp_avg",
            "optimizer.state.bias.exp_avg_sq",
        }

        dcp.load(load_target, checkpoint_id=checkpoint_dir)
        set_model_state_dict(restored_model, load_target["model"])
        set_optimizer_state_dict(
            restored_model,
            restored_optimizer,
            optim_state_dict=load_target["optimizer"],
            options=StateDictOptions(strict=False),
        )

    assert restored_model.weight in restored_optimizer.state
    assert restored_model.bias not in restored_optimizer.state
    torch.testing.assert_close(
        restored_optimizer.state[restored_model.weight]["exp_avg"],
        optimizer.state[model.weight]["exp_avg"],
    )

    restored_optimizer.zero_grad(set_to_none=True)
    restored_model.bias.mean().backward()
    restored_optimizer.step()
    assert restored_model.bias in restored_optimizer.state


def test_partially_saved_optimizer_state_is_rejected() -> None:
    optimizer_state = {
        "state": {
            "weight": {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
        }
    }
    saved_keys = {
        "optimizer.state.weight.step",
        "optimizer.state.weight.exp_avg",
    }

    with pytest.raises(RuntimeError, match="Incomplete optimizer state"):
        _prune_uncheckpointed_optimizer_state(optimizer_state, saved_keys)
