import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoConfig


_MISSING = object()


@dataclass(frozen=True)
class HFConfigFieldDependency:
    """A versioned inference-engine dependency on one exported config field."""

    engine: str
    version: str
    path: str
    expected: Any
    reason: str
    source: str


@dataclass(frozen=True)
class HFConfigSaveReport:
    """Successful HF config export check and the versions it covered."""

    transformers_version: str
    transformers_normalized_fields: tuple[str, ...]
    allowed_export_differences: tuple[str, ...]
    checked_engine_versions: tuple[str, ...]


@dataclass(frozen=True)
class _JSONDifference:
    path: str
    expected: Any
    actual: Any


def _json_differences(expected: Any, actual: Any, path: str = "") -> list[_JSONDifference]:
    differences: list[_JSONDifference] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            escaped_key = key.replace("~", "~0").replace("/", "~1")
            child_path = f"{path}/{escaped_key}"
            if key not in expected:
                differences.append(_JSONDifference(child_path, _MISSING, actual[key]))
            elif key not in actual:
                differences.append(_JSONDifference(child_path, expected[key], _MISSING))
            else:
                differences.extend(_json_differences(expected[key], actual[key], child_path))
        return differences

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [_JSONDifference(path or "/", expected, actual)]
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_json_differences(expected_item, actual_item, f"{path}/{index}"))
        return differences

    # JSON has one number type. Treat equal integer/float representations as the same,
    # while keeping bool distinct from 0/1.
    if type(expected) is type(actual):
        equal = expected == actual
    else:
        equal = type(expected) in (int, float) and type(actual) in (int, float) and expected == actual
    if not equal:
        differences.append(_JSONDifference(path or "/", expected, actual))
    return differences


def _read_json_pointer(document: Any, path: str) -> Any:
    if path == "":
        return document
    if not path.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {path!r}")

    value = document
    for raw_part in path[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _format_json_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    representation = repr(value)
    if len(representation) > 240:
        return f"{representation[:237]}..."
    return representation


def check_hf_config_save(
    model_config: Any,
    source_hf_dir: str | Path,
    *,
    engine_dependencies: Sequence[HFConfigFieldDependency] = (),
    allowed_export_differences: Mapping[str, str] | None = None,
    trust_remote_code: bool = False,
) -> HFConfigSaveReport:
    """Check an XTuner ``save_hf`` result against HF and engine contracts.

    The reference is the source ``config.json`` loaded and saved directly by the
    installed Transformers version. This separates Transformers normalization
    from fields lost specifically during the XTuner ``from_hf -> save_hf`` path.
    Engine dependencies are checked independently because an inference runtime
    may require a compatibility field that Transformers itself does not use.
    """

    source_hf_dir = Path(source_hf_dir)
    with open(source_hf_dir / "config.json", encoding="utf-8") as file:
        source_config = json.load(file)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        transformers_dir = tmpdir / "transformers"
        exported_dir = tmpdir / "exported"

        hf_config = AutoConfig.from_pretrained(source_hf_dir, trust_remote_code=trust_remote_code)
        hf_config.save_pretrained(transformers_dir)
        model_config.save_hf(exported_dir)

        with open(transformers_dir / "config.json", encoding="utf-8") as file:
            transformers_config = json.load(file)
        with open(exported_dir / "config.json", encoding="utf-8") as file:
            exported_config = json.load(file)

    normalization_differences = _json_differences(source_config, transformers_config)
    export_differences = _json_differences(transformers_config, exported_config)
    allowed_export_differences = allowed_export_differences or {}
    dependency_failures: list[tuple[HFConfigFieldDependency, Any]] = []
    satisfied_dependency_paths: set[str] = set()
    for dependency in engine_dependencies:
        actual = _read_json_pointer(exported_config, dependency.path)
        if _json_differences(dependency.expected, actual, dependency.path):
            dependency_failures.append((dependency, actual))
        else:
            satisfied_dependency_paths.add(dependency.path)

    allowed_paths = {*allowed_export_differences, *satisfied_dependency_paths}

    def is_allowed(path: str) -> bool:
        return any(path == allowed_path or path.startswith(f"{allowed_path}/") for allowed_path in allowed_paths)

    unexpected_differences = [difference for difference in export_differences if not is_allowed(difference.path)]

    if unexpected_differences or dependency_failures:
        lines = [f"HF config save check failed with Transformers {transformers.__version__}."]
        if unexpected_differences:
            lines.append("Unexpected differences from the Transformers direct round-trip:")
            lines.extend(
                f"- {difference.path}: expected {_format_json_value(difference.expected)}, "
                f"exported {_format_json_value(difference.actual)}"
                for difference in unexpected_differences
            )
        if dependency_failures:
            lines.append("Inference-engine field contract failures:")
            for dependency, actual in dependency_failures:
                lines.append(
                    f"- {dependency.engine}=={dependency.version} {dependency.path}: "
                    f"expected {_format_json_value(dependency.expected)}, exported {_format_json_value(actual)}; "
                    f"{dependency.reason} Source: {dependency.source}"
                )
        raise AssertionError("\n".join(lines))

    checked_engine_versions = tuple(
        dict.fromkeys(f"{dependency.engine}=={dependency.version}" for dependency in engine_dependencies)
    )
    return HFConfigSaveReport(
        transformers_version=transformers.__version__,
        transformers_normalized_fields=tuple(difference.path for difference in normalization_differences),
        allowed_export_differences=tuple(
            difference.path for difference in export_differences if is_allowed(difference.path)
        ),
        checked_engine_versions=checked_engine_versions,
    )
