import json
import os
import subprocess
import sys
from itertools import chain
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


SCRIPT = Path("xtuner/tools/model_converters/make_glm52_30b_hf.py")
GLM52_30B_MTP_PATH = Path(
    os.environ.get("GLM5_2_30B_MTP_PATH", "/mnt/shared-storage-user/zhaopenghao/slime0701/ckpts/GLM-5.2-30B-MTP")
)
GLM52_30B_NO_MTP_PATH = Path(
    os.environ.get(
        "GLM5_2_30B_NO_MTP_PATH", "/mnt/shared-storage-user/zhaopenghao/slime0701/ckpts/GLM-5.2-30B-NoMTP"
    )
)


def _write_fake_glm52_hf_checkpoint(path: Path) -> None:
    path.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 8,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
        "indexer_types": ["full", "full", "full", "shared", "shared", "shared", "full", "shared"],
        "vocab_size": 32,
        "hidden_size": 4,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text("{}")

    tensors = {
        "model.embed_tokens.weight": torch.zeros(2, 2),
        "lm_head.weight": torch.ones(2, 2),
    }
    for layer_idx in range(9):
        tensors[f"model.layers.{layer_idx}.input_layernorm.weight"] = torch.full((2,), layer_idx)
        tensors[f"model.layers.{layer_idx}.self_attn.q_a_proj.weight"] = torch.full((2, 2), layer_idx)
        tensors[f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"] = torch.full((2, 2), layer_idx)
    tensors["model.layers.8.eh_proj.weight"] = torch.full((2, 4), 8)
    tensors["model.layers.8.enorm.weight"] = torch.full((2,), 8)
    tensors["model.layers.8.hnorm.weight"] = torch.full((2,), 8)

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {key: shard for key in tensors},
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def _run_crop(source: Path, save: Path, profile: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--save",
            str(save),
            "--profile",
            profile,
            "--overwrite",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _load_output(path: Path) -> tuple[dict, dict]:
    return (
        json.loads((path / "config.json").read_text()),
        json.loads((path / "model.safetensors.index.json").read_text()),
    )


def test_glm52_30b_with_mtp_crop_keeps_main_layers_and_remaps_final_mtp(tmp_path):
    source = tmp_path / "source"
    save = tmp_path / "glm52-30b-mtp"
    _write_fake_glm52_hf_checkpoint(source)

    _run_crop(source, save, "30b-with-mtp")

    config, index = _load_output(save)
    weight_map = index["weight_map"]

    assert config["num_hidden_layers"] == 5
    assert config["num_nextn_predict_layers"] == 1
    assert config["mlp_layer_types"] == ["dense", "dense", "dense", "sparse", "sparse"]
    assert config["indexer_types"] == ["full", "full", "full", "shared", "shared", "full"]

    assert "model.layers.4.input_layernorm.weight" in weight_map
    assert "model.layers.5.eh_proj.weight" in weight_map
    assert "model.layers.5.enorm.weight" in weight_map
    assert "model.layers.5.self_attn.indexer.wq_b.weight" in weight_map
    assert "model.layers.8.eh_proj.weight" not in weight_map
    assert "model.layers.6.input_layernorm.weight" not in weight_map
    assert (save / "tokenizer_config.json").exists()


def test_glm52_30b_without_mtp_crop_keeps_six_main_layers_and_disables_mtp(tmp_path):
    source = tmp_path / "source"
    save = tmp_path / "glm52-30b-no-mtp"
    _write_fake_glm52_hf_checkpoint(source)

    _run_crop(source, save, "30b-no-mtp")

    config, index = _load_output(save)
    weight_map = index["weight_map"]

    assert config["num_hidden_layers"] == 6
    assert config["num_nextn_predict_layers"] == 0
    assert config["mlp_layer_types"] == ["dense", "dense", "dense", "sparse", "sparse", "sparse"]
    assert config["indexer_types"] == ["full", "full", "full", "shared", "shared", "shared"]

    assert "model.layers.5.input_layernorm.weight" in weight_map
    assert "model.layers.6.input_layernorm.weight" not in weight_map
    assert "model.layers.6.eh_proj.weight" not in weight_map
    assert "model.layers.6.self_attn.indexer.wq_b.weight" not in weight_map
    assert "model.layers.8.eh_proj.weight" not in weight_map


@pytest.mark.parametrize(
    "path,num_layers,nextn",
    [
        (GLM52_30B_MTP_PATH, 5, 1),
        (GLM52_30B_NO_MTP_PATH, 6, 0),
    ],
)
def test_generated_glm52_30b_checkpoint_matches_xtuner_hf_keys(path, num_layers, nextn):
    if not path.is_dir():
        pytest.skip(f"{path} is not available")

    from xtuner.v1.model import get_model_config_from_hf

    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_keys = set(index["weight_map"])

    assert config["num_hidden_layers"] == num_layers
    assert config["num_nextn_predict_layers"] == nextn
    assert len(config["mlp_layer_types"]) == num_layers
    assert len(config["indexer_types"]) == num_layers + nextn
    assert "shared" in config["indexer_types"]

    with torch.device("meta"):
        model_config = get_model_config_from_hf(path)
        model_config.compile_cfg = False
        model = model_config.build()
    expected_keys = set(chain.from_iterable(model.to_hf_key_list(name) for name in model.state_dict()))

    assert expected_keys == weight_keys
    assert model_config.layers_type == ["full_attention"] * num_layers
    if nextn:
        assert f"model.layers.{num_layers}.eh_proj.weight" in weight_keys
        assert f"model.layers.{num_layers}.self_attn.indexer.wq_b.weight" in weight_keys
        assert model_config.mtp_config is not None
    else:
        assert not any(key.startswith(f"model.layers.{num_layers}.") for key in weight_keys)
        assert model_config.mtp_config is None
