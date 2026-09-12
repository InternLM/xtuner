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


MIT_LICENSE = """\
MIT License

Copyright (c) 2026 Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def test_rewrite_license_holder_replaces_copyright_line():
    from xtuner.tools.model_normalize.cli import _rewrite_license_holder

    rewritten, replaced = _rewrite_license_holder(MIT_LICENSE)
    assert replaced
    assert "Copyright 2025-2026 Shanghai AI Laboratory" in rewritten
    # The original holder is gone.
    assert "Zhipu AI" not in rewritten
    # The MIT permission body is preserved verbatim.
    assert "Permission is hereby granted, free of charge" in rewritten
    assert "THE SOFTWARE IS PROVIDED" in rewritten


def test_rewrite_license_holder_no_copyright_line():
    from xtuner.tools.model_normalize.cli import _rewrite_license_holder

    text = "Some license body\nwithout a copyright line\n"
    rewritten, replaced = _rewrite_license_holder(text)
    assert not replaced
    assert rewritten == text


def test_rewrite_license_holder_only_first_copyright_line():
    from xtuner.tools.model_normalize.cli import _rewrite_license_holder

    text = "Copyright (c) 2026 Zhipu AI\nsome line\nCopyright (c) 2030 Other\n"
    rewritten, replaced = _rewrite_license_holder(text)
    assert replaced
    # Only the first Copyright line is rewritten; the second is left as-is.
    assert rewritten == "Copyright 2025-2026 Shanghai AI Laboratory\nsome line\nCopyright (c) 2030 Other\n"


def test_apply_base_model_assets_copies_generation_config(tmp_path):
    from xtuner.tools.model_normalize.cli import _apply_base_model_assets

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "generation_config.json").write_text(json.dumps({"a": 1}))
    output = tmp_path / "output"
    output.mkdir()
    # A pre-existing generation_config is overwritten.
    (output / "generation_config.json").write_text(json.dumps({"old": True}))

    _apply_base_model_assets(base_dir, output)

    assert json.loads((output / "generation_config.json").read_text()) == {"a": 1}


def test_apply_base_model_assets_rewrites_license(tmp_path):
    from xtuner.tools.model_normalize.cli import _apply_base_model_assets

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "LICENSE").write_text(MIT_LICENSE)
    output = tmp_path / "output"
    output.mkdir()

    _apply_base_model_assets(base_dir, output)

    license_text = (output / "LICENSE").read_text()
    assert "Copyright 2025-2026 Shanghai AI Laboratory" in license_text
    assert "Zhipu AI" not in license_text
    assert "Permission is hereby granted" in license_text


def test_apply_base_model_assets_none_is_noop(tmp_path):
    from xtuner.tools.model_normalize.cli import _apply_base_model_assets

    output = tmp_path / "output"
    output.mkdir()
    (output / "generation_config.json").write_text(json.dumps({"keep": True}))

    # No exception, no change to output.
    _apply_base_model_assets(None, output)
    assert json.loads((output / "generation_config.json").read_text()) == {"keep": True}
    assert not (output / "LICENSE").exists()


def test_apply_base_model_assets_missing_files_is_noop(tmp_path):
    from xtuner.tools.model_normalize.cli import _apply_base_model_assets

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    _apply_base_model_assets(base_dir, output)
    assert not (output / "generation_config.json").exists()
    assert not (output / "LICENSE").exists()


def test_apply_base_model_assets_missing_dir_raises(tmp_path):
    from xtuner.tools.model_normalize.cli import _apply_base_model_assets

    with pytest.raises(FileNotFoundError):
        _apply_base_model_assets(tmp_path / "does-not-exist", tmp_path / "output")
