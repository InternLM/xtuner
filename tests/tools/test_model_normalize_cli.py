import json

import pytest

from xtuner.tools.model_normalize.cli import _reference_predicate


def test_reference_policy_selects_scale_companions(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.up_proj.weight_scale_inv": "model-00001-of-00001.safetensors",
                    "model.layers.0.mtp.fc.weight": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    predicate = _reference_predicate(reference)
    assert predicate("model.layers.0.mlp.up_proj.weight")
    assert not predicate("model.layers.0.mtp.fc.weight")


def test_mtp_keys_are_not_filtered_by_reference_policy(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mtp.fc.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mtp.fc.weight_scale_inv": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    predicate = _reference_predicate(reference)
    assert predicate("model.layers.0.mtp.fc.weight")


def test_repack_preserves_mtp_keys(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    import torch

    from xtuner.tools.model_normalize.repack import repack

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "model.layers.0.mtp.fc.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
    }
    save_file(tensors, source / "engine-rank0.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": dict.fromkeys(tensors, "engine-rank0.safetensors"),
            }
        )
    )
    repack(source, output, shard_size_bytes=1024)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert (output / "model-00001-of-00001.safetensors").is_file()
