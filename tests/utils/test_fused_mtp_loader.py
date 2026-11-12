"""Actual safetensors I/O and native loader; no allocator or model stubs."""

import json

import pytest
import torch
from safetensors.torch import save_file

from xtuner.v1.utils.loader import HFCheckpointLoader


PREFIX = "mtp.layers.0.mlp.experts"


def checkpoint(path, tensors, indexed=False):
    save_file(tensors, str(path / "model.safetensors"))
    if indexed:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
        )
    return HFCheckpointLoader(str(path))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("indexed", [False, True])
def test_fused_mtp_expert_reads(tmp_path, dtype, indexed):
    gate_up = torch.arange(24).reshape(2, 4, 3).to(dtype)
    down = torch.arange(12).reshape(2, 3, 2).to(dtype)
    loader = checkpoint(tmp_path, {PREFIX + ".gate_up_proj": gate_up, PREFIX + ".down_proj": down}, indexed)
    for expert in range(2):
        for part, expected in [("gate", gate_up[expert, :2]), ("up", gate_up[expert, 2:]), ("down", down[expert])]:
            key = f"{PREFIX}.{expert}.{part}_proj.weight"
            assert loader.is_key_exist(key)
            torch.testing.assert_close(loader.load(key), expected, rtol=0, atol=0)
    assert not loader.is_key_exist(f"{PREFIX}.0.gate_proj.weight_scale_inv")


def test_direct_export_keys_take_precedence(tmp_path):
    direct = f"{PREFIX}.0.gate_proj.weight"
    loader = checkpoint(tmp_path, {PREFIX + ".gate_up_proj": torch.zeros(2, 4, 3), direct: torch.ones(2, 3)})
    torch.testing.assert_close(loader.load(direct), torch.ones(2, 3), rtol=0, atol=0)


def test_missing_and_non_mtp_keys_preserve_behavior(tmp_path):
    loader = checkpoint(tmp_path, {PREFIX + ".gate_up_proj": torch.zeros(2, 4, 3)})
    for key in [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        f"{PREFIX}.0.unknown_proj.weight",
        "mtp.layers.1.mlp.experts.0.gate_proj.weight",
    ]:
        assert not loader.is_key_exist(key)
        assert loader.load(key) is None


def test_bin_checkpoint_direct_path_preserved(tmp_path):
    value = torch.arange(6).reshape(2, 3)
    torch.save({"ordinary.weight": value}, tmp_path / "part.bin")
    (tmp_path / "model.bin.index.json").write_text(json.dumps({"weight_map": {"ordinary.weight": "part.bin"}}))
    loader = HFCheckpointLoader(str(tmp_path))
    assert loader.is_key_exist("ordinary.weight")
    torch.testing.assert_close(loader.load("ordinary.weight"), value)
    assert loader.load(f"{PREFIX}.0.gate_proj.weight") is None


@pytest.mark.parametrize(
    "tensor,expert,message",
    [
        (torch.zeros(2, 4), 0, "shape/index"),
        (torch.zeros(2, 4, 3), 2, "shape/index"),
        (torch.zeros(2, 5, 3), 0, "even"),
        (torch.zeros(2, 4, 3, dtype=torch.int64), 0, "unquantized"),
    ],
)
def test_invalid_fused_inputs_fail_explicitly(tmp_path, tensor, expert, message):
    loader = checkpoint(tmp_path, {PREFIX + ".gate_up_proj": tensor})
    with pytest.raises(ValueError, match=message):
        loader.load(f"{PREFIX}.{expert}.gate_proj.weight")
