#!/usr/bin/env python3
"""Scan xtuner/v1/model for all Config classes and output model info as JSON."""

import json
import re
import sys
from pathlib import Path


def scan_file(path: Path) -> list[dict[str, str | list[str]]]:
    text = path.read_text()
    # Match class definitions like: class FooConfig(BarConfig):
    pattern = r"^class\s+(\w+Config)\s*\(([^)]+)\):"
    results: list[dict[str, str | list[str]]] = []
    for m in re.finditer(pattern, text, re.MULTILINE):
        class_name = m.group(1)
        parents = [p.strip() for p in m.group(2).split(",")]
        results.append({"class": class_name, "parents": parents, "file": str(path)})
    return results


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    model_dir = root / "xtuner" / "v1" / "model"
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    all_configs = []
    for py_file in sorted(model_dir.rglob("*.py")):
        all_configs.extend(scan_file(py_file))

    # Build parent -> children map
    children: dict[str, list[str]] = {}
    for cfg in all_configs:
        for p in cfg["parents"]:
            if p.endswith("Config"):
                children.setdefault(p, []).append(cfg["class"])

    # Deduplicate
    for k in children:
        children[k] = sorted(set(children[k]))

    output = {
        "configs": all_configs,
        "children": children,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
