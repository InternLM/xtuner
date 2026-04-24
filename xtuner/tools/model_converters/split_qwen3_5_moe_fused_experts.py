# Copyright (c) OpenMMLab. All rights reserved.
"""Split fused Qwen3.5 MoE expert tensors in HuggingFace safetensors checkpoints."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_GATE_UP_PROJ = "gate_up_proj"
_DOWN_PROJ = "down_proj"
_GATE_UP_PROJ_BIAS = "gate_up_proj_bias"
_DOWN_PROJ_BIAS = "down_proj_bias"
_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_SAFETENSORS_NAME = "model.safetensors"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Qwen3.5 fused MoE expert tensors in a HuggingFace safetensors checkpoint "
            "to per-expert split tensors."
        )
    )
    parser.add_argument("src_dir", type=Path, help="Directory containing the original HuggingFace checkpoint.")
    parser.add_argument("dst_dir", type=Path, help="Directory to write the converted checkpoint.")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Expected routed expert count. By default it is inferred from each fused tensor's first dimension.",
    )
    parser.add_argument(
        "--key-prefix",
        default="model.language_model",
        help=(
            "HF key prefix before '.layers'. The default matches "
            "xtuner/v1/model/moe/qwen3_5_text.py."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty destination directory.",
    )
    return parser.parse_args()


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _validate_dirs(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    src_dir = _safe_resolve(src_dir)
    dst_dir = _safe_resolve(dst_dir)

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {src_dir}")
    if src_dir == dst_dir:
        raise ValueError("Source and destination directories must be different.")
    if dst_dir.is_relative_to(src_dir):
        raise ValueError("Destination directory must not be placed inside the source directory.")
    if dst_dir.exists() and any(dst_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination directory is not empty: {dst_dir}. Pass --overwrite to reuse it."
        )


def _load_index(src_dir: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    index_path = src_dir / _INDEX_NAME
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        return index["weight_map"], index

    safetensors_path = src_dir / _SINGLE_SAFETENSORS_NAME
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Cannot find {_INDEX_NAME} or {_SINGLE_SAFETENSORS_NAME} under {src_dir}."
        )

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        weight_map = {key: _SINGLE_SAFETENSORS_NAME for key in f.keys()}
    return weight_map, None


def _copy_sidecar_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_path in src_dir.iterdir():
        if src_path.name == _INDEX_NAME or src_path.suffix == ".safetensors":
            continue

        dst_path = dst_dir / src_path.name
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def _group_keys_by_file(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped_keys: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        grouped_keys[filename].append(key)
    return grouped_keys


def _build_fused_prefix(key_prefix: str) -> str:
    return f"{key_prefix}.layers."


def _is_fused_expert_key(key: str, key_prefix: str) -> bool:
    if not key.startswith(_build_fused_prefix(key_prefix)):
        return False
    return key.endswith(
        (
            f".mlp.experts.{_GATE_UP_PROJ}",
            f".mlp.experts.{_DOWN_PROJ}",
            f".mlp.experts.{_GATE_UP_PROJ_BIAS}",
            f".mlp.experts.{_DOWN_PROJ_BIAS}",
        )
    )


def _expert_key_base(key: str, fused_name: str) -> str:
    return key[: -len(fused_name)].rstrip(".")


def _validate_num_experts(key: str, tensor: torch.Tensor, num_experts: int | None) -> int:
    inferred_num_experts = tensor.shape[0]
    if num_experts is not None and inferred_num_experts != num_experts:
        raise ValueError(
            f"{key} has {inferred_num_experts} experts, but --num-experts is {num_experts}."
        )
    return inferred_num_experts


def _split_gate_up_proj(
    key: str,
    tensor: torch.Tensor,
    num_experts: int | None,
) -> dict[str, torch.Tensor]:
    if tensor.ndim != 3:
        raise ValueError(f"{key} should be a 3D tensor, got shape {tuple(tensor.shape)}.")
    if tensor.shape[1] % 2 != 0:
        raise ValueError(f"{key} second dimension should be divisible by 2, got shape {tuple(tensor.shape)}.")

    expert_count = _validate_num_experts(key, tensor, num_experts)
    expert_dim = tensor.shape[1] // 2
    base = _expert_key_base(key, _GATE_UP_PROJ)
    split_tensors = {}

    for expert_idx in range(expert_count):
        split_tensors[f"{base}.{expert_idx}.gate_proj.weight"] = tensor[expert_idx, :expert_dim, :].contiguous()
        split_tensors[f"{base}.{expert_idx}.up_proj.weight"] = tensor[expert_idx, expert_dim:, :].contiguous()
    return split_tensors


def _split_down_proj(
    key: str,
    tensor: torch.Tensor,
    num_experts: int | None,
) -> dict[str, torch.Tensor]:
    if tensor.ndim != 3:
        raise ValueError(f"{key} should be a 3D tensor, got shape {tuple(tensor.shape)}.")

    expert_count = _validate_num_experts(key, tensor, num_experts)
    base = _expert_key_base(key, _DOWN_PROJ)
    split_tensors = {}

    for expert_idx in range(expert_count):
        split_tensors[f"{base}.{expert_idx}.down_proj.weight"] = tensor[expert_idx].contiguous()
    return split_tensors


def _split_gate_up_proj_bias(
    key: str,
    tensor: torch.Tensor,
    num_experts: int | None,
) -> dict[str, torch.Tensor]:
    if tensor.ndim != 2:
        raise ValueError(f"{key} should be a 2D tensor, got shape {tuple(tensor.shape)}.")
    if tensor.shape[1] % 2 != 0:
        raise ValueError(f"{key} second dimension should be divisible by 2, got shape {tuple(tensor.shape)}.")

    expert_count = _validate_num_experts(key, tensor, num_experts)
    expert_dim = tensor.shape[1] // 2
    base = _expert_key_base(key, _GATE_UP_PROJ_BIAS)
    split_tensors = {}

    for expert_idx in range(expert_count):
        split_tensors[f"{base}.{expert_idx}.gate_proj.bias"] = tensor[expert_idx, :expert_dim].contiguous()
        split_tensors[f"{base}.{expert_idx}.up_proj.bias"] = tensor[expert_idx, expert_dim:].contiguous()
    return split_tensors


def _split_down_proj_bias(
    key: str,
    tensor: torch.Tensor,
    num_experts: int | None,
) -> dict[str, torch.Tensor]:
    if tensor.ndim != 2:
        raise ValueError(f"{key} should be a 2D tensor, got shape {tuple(tensor.shape)}.")

    expert_count = _validate_num_experts(key, tensor, num_experts)
    base = _expert_key_base(key, _DOWN_PROJ_BIAS)
    split_tensors = {}

    for expert_idx in range(expert_count):
        split_tensors[f"{base}.{expert_idx}.down_proj.bias"] = tensor[expert_idx].contiguous()
    return split_tensors


def _split_fused_expert_tensor(
    key: str,
    tensor: torch.Tensor,
    num_experts: int | None,
) -> dict[str, torch.Tensor]:
    if key.endswith(f".{_GATE_UP_PROJ}"):
        return _split_gate_up_proj(key, tensor, num_experts)
    if key.endswith(f".{_DOWN_PROJ}"):
        return _split_down_proj(key, tensor, num_experts)
    if key.endswith(f".{_GATE_UP_PROJ_BIAS}"):
        return _split_gate_up_proj_bias(key, tensor, num_experts)
    if key.endswith(f".{_DOWN_PROJ_BIAS}"):
        return _split_down_proj_bias(key, tensor, num_experts)
    raise ValueError(f"Unsupported fused expert key: {key}")


def _convert_safetensors_file(
    src_file: Path,
    dst_file: Path,
    keys: list[str],
    key_prefix: str,
    num_experts: int | None,
) -> tuple[dict[str, str], int, int]:
    output_tensors = {}
    output_weight_map = {}
    fused_tensor_count = 0
    split_tensor_count = 0

    with safe_open(src_file, framework="pt", device="cpu") as f:
        actual_keys = set(f.keys())
        missing_keys = set(keys) - actual_keys
        if missing_keys:
            raise KeyError(f"{src_file} is missing keys declared in the index: {sorted(missing_keys)}")
        metadata = f.metadata()

        for key in keys:
            tensor = f.get_tensor(key)
            if _is_fused_expert_key(key, key_prefix):
                split_tensors = _split_fused_expert_tensor(key, tensor, num_experts)
                output_tensors.update(split_tensors)
                output_weight_map.update({split_key: dst_file.name for split_key in split_tensors})
                fused_tensor_count += 1
                split_tensor_count += len(split_tensors)
            else:
                output_tensors[key] = tensor.contiguous()
                output_weight_map[key] = dst_file.name

    # Preserve the original file grouping so existing shard sizes remain predictable; the speedup comes
    # from making each expert slice independently addressable by safetensors.
    save_file(output_tensors, dst_file, metadata=metadata)
    return output_weight_map, fused_tensor_count, split_tensor_count


def _write_index(dst_dir: Path, index: dict[str, Any] | None, weight_map: dict[str, str]) -> None:
    if index is None:
        if set(weight_map.values()) == {_SINGLE_SAFETENSORS_NAME}:
            return
        new_index = {"metadata": {}, "weight_map": weight_map}
    else:
        new_index = index.copy()
        new_index["weight_map"] = weight_map

    with open(dst_dir / _INDEX_NAME, "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)
        f.write("\n")


@torch.no_grad()
def _main() -> None:
    args = _parse_args()
    src_dir = _safe_resolve(args.src_dir)
    dst_dir = _safe_resolve(args.dst_dir)

    _validate_dirs(src_dir, dst_dir, args.overwrite)
    weight_map, index = _load_index(src_dir)
    grouped_keys = _group_keys_by_file(weight_map)

    _copy_sidecar_files(src_dir, dst_dir)

    new_weight_map = {}
    fused_tensor_count = 0
    split_tensor_count = 0

    for filename, keys in grouped_keys.items():
        src_file = src_dir / filename
        dst_file = dst_dir / filename
        file_weight_map, file_fused_count, file_split_count = _convert_safetensors_file(
            src_file=src_file,
            dst_file=dst_file,
            keys=keys,
            key_prefix=args.key_prefix,
            num_experts=args.num_experts,
        )
        new_weight_map.update(file_weight_map)
        fused_tensor_count += file_fused_count
        split_tensor_count += file_split_count

    _write_index(dst_dir, index, new_weight_map)
    print(
        f"Converted {fused_tensor_count} fused expert tensors into {split_tensor_count} split tensors "
        f"under {dst_dir}."
    )


if __name__ == "__main__":
    _main()
