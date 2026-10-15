"""F0 的 25B 裁剪 checkpoint 构造器，见 doc/xtuner_glm5p3flash_design.md 3.2。

TestDequantizeFp8Block
    test_matches_naive_per_block_multiply        逐块反量化与朴素实现一致
TestCropLayout
    test_keeps_5_main_layers_and_remaps_mtp      保留 5 层主栈并把 MTP 层重编号
TestCropWeightDtypes
    test_dequantizes_fp8_weights_to_bf16         FP8 权重被反量化成 bf16
    test_passes_through_native_bf16_weights_unchanged  原生 bf16 权重逐位不变
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from xtuner.tools.model_converters.make_glm53_25b_hf import dequantize_fp8_block


SCRIPT = Path("xtuner/tools/model_converters/make_glm53_25b_hf.py")

# Mirrors the real checkpoint's first 6 + final (MTP) layers: 3 dense(KDA) + DSA + KDA + DSA(MTP).
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
]
MLP_LAYER_TYPES = ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse"]
INDEXER_TYPES = ["full"] * len(LAYER_TYPES)
ORIGINAL_MAIN_LAYERS = 6  # layers 0..5 are the main stack; layer 6 is the original MTP layer.


def _write_fake_glm53_hf_checkpoint(path: Path, *, quantize: bool = True) -> None:
    """``quantize=False`` mirrors the official native-BF16 GLM-5.3-Flash release (no
    ``quantization_config`` / ``*_scale_inv`` tensors): the crop script must pass such weights
    through untouched rather than attempting to dequantize them."""
    path.mkdir()
    config = {
        "text_config": {
            "num_hidden_layers": ORIGINAL_MAIN_LAYERS,
            "first_k_dense_replace": 3,
            "num_nextn_predict_layers": 1,
            "layer_types": LAYER_TYPES,
            "mlp_layer_types": MLP_LAYER_TYPES,
            "indexer_types": INDEXER_TYPES,
            "linear_attn_config": {
                "num_heads": 4,
                "kda_layers": [0, 1, 2, 4],
                "full_attn_layers": [3, 6],
            },
        },
        "vision_config": {"hidden_size": 4},
    }
    if quantize:
        config["quantization_config"] = {"fmt": "e4m3", "activation_scheme": "dynamic"}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text("{}")

    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.zeros(2, 2),
        "lm_head.weight": torch.ones(2, 2),
        "model.visual.patch_embed.proj.weight": torch.full((2, 2), 99.0),
    }
    for layer_idx in range(ORIGINAL_MAIN_LAYERS + 1):
        prefix = f"model.language_model.layers.{layer_idx}."
        tensors[prefix + "input_layernorm.weight"] = torch.full((2,), float(layer_idx))
        if quantize:
            # A small FP8-quantized tensor with a single 128-block (block is larger than the
            # tensor, exercising the "crop scale to weight shape" path).
            tensors[prefix + "mlp.gate_proj.weight"] = torch.full((2, 2), float(layer_idx)).to(torch.float8_e4m3fn)
            tensors[prefix + "mlp.gate_proj.weight_scale_inv"] = torch.full((1, 1), 2.0)
        else:
            tensors[prefix + "mlp.gate_proj.weight"] = torch.full((2, 2), float(layer_idx), dtype=torch.bfloat16)
    tensors["model.language_model.layers.6.eh_proj.weight"] = torch.full((2, 4), 6.0)
    tensors["model.language_model.layers.6.enorm.weight"] = torch.full((2,), 6.0)
    tensors["model.language_model.layers.6.hnorm.weight"] = torch.full((2,), 6.0)

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": dict.fromkeys(tensors, shard),
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def _run_crop(source: Path, save: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--save", str(save), "--overwrite"],
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


class TestDequantizeFp8Block:
    """FP8 逐块反量化。"""

    def test_matches_naive_per_block_multiply(self):
        # 逐 128x128 块乘 scale_inv 的实现必须与朴素双重循环一致。
        weight = torch.randn(300, 200).to(torch.float8_e4m3fn)
        scale_inv = torch.rand(3, 2) + 0.1  # ceil(300/128)=3, ceil(200/128)=2

        got = dequantize_fp8_block(weight, scale_inv, block=128)

        expected = torch.zeros(300, 200)
        for i in range(3):
            for j in range(2):
                r0, r1 = i * 128, min((i + 1) * 128, 300)
                c0, c1 = j * 128, min((j + 1) * 128, 200)
                expected[r0:r1, c0:c1] = weight[r0:r1, c0:c1].float() * scale_inv[i, j]
        torch.testing.assert_close(got, expected.bfloat16())


class TestCropLayout:
    """裁剪后的层布局。"""

    def test_keeps_5_main_layers_and_remaps_mtp(self, tmp_path):
        # 裁剪保留前 5 层主栈，并把原 MTP 层重编号到主栈之后。
        source = tmp_path / "source"
        save = tmp_path / "glm53-25b"
        _write_fake_glm53_hf_checkpoint(source)

        _run_crop(source, save)

        config, index = _load_output(save)
        tc = config["text_config"]
        weight_map = index["weight_map"]

        assert tc["num_hidden_layers"] == 5
        assert tc["first_k_dense_replace"] == 3
        assert tc["num_nextn_predict_layers"] == 1
        assert tc["layer_types"] == LAYER_TYPES[:5]
        assert tc["mlp_layer_types"] == MLP_LAYER_TYPES[:5]
        assert tc["indexer_types"] == INDEXER_TYPES[:5]
        assert tc["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4]
        assert tc["linear_attn_config"]["full_attn_layers"] == [3]
        assert "quantization_config" not in config

        assert "model.language_model.layers.4.input_layernorm.weight" in weight_map
        assert "model.language_model.layers.5.eh_proj.weight" in weight_map
        assert "model.language_model.layers.5.enorm.weight" in weight_map
        assert "model.language_model.layers.6.eh_proj.weight" not in weight_map
        assert "model.language_model.layers.5.mlp.gate_proj.weight" in weight_map
        assert "model.language_model.layers.5.mlp.gate_proj.weight_scale_inv" not in weight_map
        assert "model.visual.patch_embed.proj.weight" in weight_map
        assert "model.language_model.embed_tokens.weight" in weight_map
        assert "lm_head.weight" in weight_map
        assert (save / "tokenizer_config.json").exists()


class TestCropWeightDtypes:
    """裁剪时的权重精度处理。"""

    def test_dequantizes_fp8_weights_to_bf16(self, tmp_path):
        # 源 checkpoint 的 FP8 权重要被反量化成 bf16 写出。
        source = tmp_path / "source"
        save = tmp_path / "glm53-25b"
        _write_fake_glm53_hf_checkpoint(source)

        _run_crop(source, save)

        _, index = _load_output(save)
        shard = index["weight_map"]["model.language_model.layers.0.mlp.gate_proj.weight"]
        with safe_open(save / shard, framework="pt", device="cpu") as reader:
            tensor = reader.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")
            assert tensor.dtype == torch.bfloat16
            # source fp8 value was 0.0 (layer_idx=0) with scale_inv=2.0 -> dequantized value 0.0
            torch.testing.assert_close(tensor, torch.zeros(2, 2, dtype=torch.bfloat16))

            shard1 = index["weight_map"]["model.language_model.layers.1.mlp.gate_proj.weight"]
        with safe_open(save / shard1, framework="pt", device="cpu") as reader:
            tensor1 = reader.get_tensor("model.language_model.layers.1.mlp.gate_proj.weight")
            # source fp8 value was 1.0 (layer_idx=1) with scale_inv=2.0 -> dequantized value 2.0
            torch.testing.assert_close(tensor1, torch.full((2, 2), 2.0, dtype=torch.bfloat16))

    def test_passes_through_native_bf16_weights_unchanged(self, tmp_path):
        # 已经是 bf16 的权重必须逐位透传，不做多余转换。
        """Official native-BF16 GLM-5.3-Flash release has no FP8 tensors; the crop must copy
        weights through byte-identical rather than attempting a (nonexistent) dequant step."""
        source = tmp_path / "source"
        save = tmp_path / "glm53-25b"
        _write_fake_glm53_hf_checkpoint(source, quantize=False)

        _run_crop(source, save)

        config, index = _load_output(save)
        assert "quantization_config" not in config

        shard1 = index["weight_map"]["model.language_model.layers.1.mlp.gate_proj.weight"]
        with safe_open(save / shard1, framework="pt", device="cpu") as reader:
            tensor1 = reader.get_tensor("model.language_model.layers.1.mlp.gate_proj.weight")
            assert tensor1.dtype == torch.bfloat16
            torch.testing.assert_close(tensor1, torch.full((2, 2), 1.0, dtype=torch.bfloat16))
