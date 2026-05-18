#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


IMG_CONTEXT_LINE_RE = re.compile(r"(?m)^<IMG_CONTEXT>\s*\n?")
LEADING_IMG_CONTEXT_RE = re.compile(r"^(?:(?:<IMG_CONTEXT>\s*\n)|(?:<IMG_CONTEXT>\s*))+")


def clean_img_context(text: str) -> str:
    """Remove <IMG_CONTEXT> marker lines and leftover standalone markers."""
    if "<IMG_CONTEXT>" not in text:
        return text

    cleaned = IMG_CONTEXT_LINE_RE.sub("", text)
    cleaned = cleaned.replace("<IMG_CONTEXT>\n", "")
    cleaned = cleaned.replace("<IMG_CONTEXT>", "")
    # Remove any leading whitespace left after marker cleanup so text starts
    # with meaningful content instead of spaces/tabs/newlines.
    return cleaned.lstrip()


def starts_with_img_context(text: str) -> bool:
    return bool(LEADING_IMG_CONTEXT_RE.match(text))


def reorder_content_items(content_items: List[Any]) -> Tuple[List[Any], int]:
    """
    If text content starts with <IMG_CONTEXT>, move image blocks before text.
    Keeps relative order among image items and among non-image items.
    """
    has_prefixed_img_context = False
    for item in content_items:
        if isinstance(item, dict) and item.get("__img_context_prefix__") is True:
            has_prefixed_img_context = True
            break

    if not has_prefixed_img_context:
        return content_items, 0

    images: List[Any] = []
    others: List[Any] = []
    for item in content_items:
        if isinstance(item, dict) and item.get("type") == "image":
            images.append(item)
        else:
            others.append(item)

    reordered = images + others
    return reordered, 1 if reordered != content_items else 0


def remove_internal_flags(obj: Any) -> Any:
    if isinstance(obj, list):
        return [remove_internal_flags(item) for item in obj]
    if isinstance(obj, dict):
        return {
            key: remove_internal_flags(value)
            for key, value in obj.items()
            if key != "__img_context_prefix__"
        }
    return obj


def transform_obj(obj: Any) -> Tuple[Any, int]:
    """
    Transform one JSON-compatible object.
    Returns transformed object and number of applied conversions.
    """
    changes = 0

    if isinstance(obj, list):
        new_list: List[Any] = []
        for item in obj:
            new_item, c = transform_obj(item)
            changes += c
            new_list.append(new_item)

        reordered_list, c = reorder_content_items(new_list)
        changes += c
        new_list = reordered_list

        return new_list, changes

    if isinstance(obj, dict):
        new_dict: Dict[str, Any] = {}
        for key, value in obj.items():
            new_value, c = transform_obj(value)
            changes += c
            new_dict[key] = new_value

        # {"type":"image_url","image_url":{...}} -> {"type":"image","image":{...}}
        if (
            new_dict.get("type") == "image_url"
            and isinstance(new_dict.get("image_url"), dict)
        ):
            new_dict["type"] = "image"
            new_dict["image"] = new_dict.pop("image_url")
            changes += 1

        # Remove IMG_CONTEXT markers from common textual fields
        for text_key in ("text", "content"):
            if isinstance(new_dict.get(text_key), str):
                if text_key == "text" and starts_with_img_context(new_dict[text_key]):
                    new_dict["__img_context_prefix__"] = True
                cleaned = clean_img_context(new_dict[text_key])
                if cleaned != new_dict[text_key]:
                    new_dict[text_key] = cleaned
                    changes += 1

        return new_dict, changes

    return obj, changes


def convert_jsonl(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """Convert one jsonl file, returning (line_count, change_count)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    change_count = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw_line in fin:
            stripped = raw_line.strip()
            if not stripped:
                fout.write(raw_line)
                continue

            line_count += 1
            record = json.loads(stripped)
            new_record, c = transform_obj(record)
            new_record = remove_internal_flags(new_record)
            change_count += c
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    return line_count, change_count


def make_output_jsonl_path(
    input_jsonl: str, out_dir: Path, dataset_name: str, index: int
) -> Path:
    """
    Build clean deterministic output path under out_dir.
    Example:
    /a/b/c/data.jsonl -> out/converted_jsonl/<dataset_name>/001_data.jsonl
    """
    src = Path(input_jsonl)
    filename = src.name if src.name else f"annotation_{index}.jsonl"
    return out_dir / "converted_jsonl" / dataset_name / f"{index:03d}_{filename}"


def convert_config(config: Dict[str, Any], out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert config data in-memory and write converted jsonl files.
    Only nodes that contain truthy `media_root` and valid `annotation` (str/list)
    are converted.
    """
    summary = {
        "datasets_total": 0,
        "datasets_converted": 0,
        "jsonl_files_converted": 0,
        "records_processed": 0,
        "changes_applied": 0,
    }

    new_config = json.loads(json.dumps(config))

    if not isinstance(new_config, dict):
        raise ValueError("Top-level config must be a JSON object.")

    for dataset_name, dataset_cfg in new_config.items():
        if not isinstance(dataset_cfg, dict):
            continue

        summary["datasets_total"] += 1
        ann = dataset_cfg.get("annotation")
        has_media_root = bool(dataset_cfg.get("media_root"))

        if not has_media_root:
            continue

        annotation_was_str = isinstance(ann, str)
        if annotation_was_str:
            ann_list: List[Any] = [ann]
        elif isinstance(ann, list):
            ann_list = ann
        else:
            continue

        new_annotations: List[str] = []
        for ann_idx, ann_path in enumerate(ann_list, start=1):
            if not isinstance(ann_path, str):
                new_annotations.append(ann_path)
                continue

            in_jsonl = Path(ann_path)
            out_jsonl = make_output_jsonl_path(
                input_jsonl=ann_path,
                out_dir=out_dir,
                dataset_name=dataset_name,
                index=ann_idx,
            )
            lines, changes = convert_jsonl(in_jsonl, out_jsonl)

            summary["jsonl_files_converted"] += 1
            summary["records_processed"] += lines
            summary["changes_applied"] += changes
            new_annotations.append(str(out_jsonl))

        dataset_cfg["annotation"] = new_annotations[0] if annotation_was_str else new_annotations
        summary["datasets_converted"] += 1
        print(
            f"[converted] dataset={dataset_name} files={len(new_annotations)}"
            f" annotation_type={'str' if annotation_was_str else 'list'}"
        )

    return new_config, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert non-standard VL jsonl format into standard image/text format."
    )
    parser.add_argument(
        "-i",
        "--input-config",
        required=True,
        help="Input config JSON path, e.g. vl-debug-mmk12.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for converted jsonl and converted config.",
    )
    parser.add_argument(
        "--output-config-name",
        default=None,
        help="Optional output config filename. Defaults to input config basename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_config = Path(args.input_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_config.open("r", encoding="utf-8") as f:
        config_data = json.load(f)

    converted_config, summary = convert_config(config_data, out_dir)

    output_config_name = args.output_config_name or input_config.name
    output_config_path = out_dir / output_config_name
    with output_config_path.open("w", encoding="utf-8") as f:
        json.dump(converted_config, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print("\n=== Conversion Summary ===")
    print(f"Input config:  {input_config}")
    print(f"Output config: {output_config_path}")
    print(f"Datasets total:      {summary['datasets_total']}")
    print(f"Datasets converted:  {summary['datasets_converted']}")
    print(f"Jsonl converted:     {summary['jsonl_files_converted']}")
    print(f"Records processed:   {summary['records_processed']}")
    print(f"Changes applied:     {summary['changes_applied']}")


if __name__ == "__main__":
    main()
