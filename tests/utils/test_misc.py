import os

import torch
from torch._inductor import config as inductor_config

import xtuner.v1.utils.misc as misc


def test_set_deterministic_disables_inductor_dynamic_scale_rblock(monkeypatch):
    original_dynamic_scale_rblock = inductor_config.dynamic_scale_rblock
    original_debug_mode = torch.get_deterministic_debug_mode()
    try:
        monkeypatch.setattr(misc, "XTUNER_DETERMINISTIC", True)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.delenv("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", raising=False)
        inductor_config.dynamic_scale_rblock = True

        misc.set_deterministic()

        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
        assert os.environ["TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"] == "0"
        assert inductor_config.dynamic_scale_rblock is False
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        inductor_config.dynamic_scale_rblock = original_dynamic_scale_rblock
        torch.set_deterministic_debug_mode(original_debug_mode)


def test_set_deterministic_keeps_user_rblock_config_when_disabled(monkeypatch):
    original_dynamic_scale_rblock = inductor_config.dynamic_scale_rblock
    original_debug_mode = torch.get_deterministic_debug_mode()
    try:
        monkeypatch.setattr(misc, "XTUNER_DETERMINISTIC", False)
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        monkeypatch.setenv("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "1")
        inductor_config.dynamic_scale_rblock = True
        torch.set_deterministic_debug_mode(0)

        misc.set_deterministic()

        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert os.environ["TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"] == "1"
        assert inductor_config.dynamic_scale_rblock is True
        assert torch.get_deterministic_debug_mode() == 0
    finally:
        inductor_config.dynamic_scale_rblock = original_dynamic_scale_rblock
        torch.set_deterministic_debug_mode(original_debug_mode)
