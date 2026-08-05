import json
from pathlib import Path

import pytest

import transformers
from transformers import AutoConfig, BertConfig
from xtuner._testing import HFConfigFieldDependency, check_hf_config_save


class _HFConfigExporter:
    def __init__(self, source_hf_dir: Path, *, drop_field: str | None = None) -> None:
        self.config = AutoConfig.from_pretrained(source_hf_dir)
        self.drop_field = drop_field

    def save_hf(self, output_dir: Path) -> None:
        self.config.save_pretrained(output_dir)
        if self.drop_field is not None:
            config_path = output_dir / "config.json"
            config = json.loads(config_path.read_text())
            config.pop(self.drop_field)
            config_path.write_text(json.dumps(config))


def test_check_hf_config_save_uses_transformers_round_trip_as_reference(tmp_path: Path):
    source_hf_dir = tmp_path / "source"
    BertConfig(hidden_size=16, runtime_mode="special").save_pretrained(source_hf_dir)
    source_config_path = source_hf_dir / "config.json"
    source_config = json.loads(source_config_path.read_text())
    source_config["transformers_version"] = "0.0.0"
    source_config_path.write_text(json.dumps(source_config))

    report = check_hf_config_save(
        _HFConfigExporter(source_hf_dir),
        source_hf_dir,
        engine_dependencies=(
            HFConfigFieldDependency(
                engine="example-engine",
                version="1.2.3",
                path="/runtime_mode",
                expected="special",
                reason="The engine selects its runtime path from this field.",
                source="https://example.com/example-engine/v1.2.3/model.py#L1",
            ),
        ),
    )

    assert report.transformers_version == transformers.__version__
    assert report.transformers_normalized_fields == ("/transformers_version",)
    assert report.checked_engine_versions == ("example-engine==1.2.3",)


def test_check_hf_config_save_reports_dropped_fields(tmp_path: Path):
    source_hf_dir = tmp_path / "source"
    BertConfig(hidden_size=16).save_pretrained(source_hf_dir)

    with pytest.raises(AssertionError) as error:
        check_hf_config_save(
            _HFConfigExporter(source_hf_dir, drop_field="hidden_size"),
            source_hf_dir,
            engine_dependencies=(
                HFConfigFieldDependency(
                    engine="example-engine",
                    version="1.2.3",
                    path="/hidden_size",
                    expected=16,
                    reason="The engine uses this field to construct parameter shapes.",
                    source="https://example.com/example-engine/v1.2.3/model.py#L2",
                ),
            ),
        )

    message = str(error.value)
    assert "Transformers direct round-trip" in message
    assert "example-engine==1.2.3" in message
    assert "/hidden_size" in message
