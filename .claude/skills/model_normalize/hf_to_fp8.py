r"""Per-block FP8 quantization of an HF-format safetensors checkpoint.

The module exposes both a library API and a small CLI. The library form is
preferred when you want to drive the quantization policy from another script
(e.g. ``model_normalize.py``); the CLI form is kept for ad-hoc use.

Library:
    from hf_to_fp8 import convert
    convert(source, target, should_quantize=lambda name: ...)

CLI (single pattern):
    python hf_to_fp8.py <bf16-path> <fp8-path> \
        'model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|down|up)_proj\.weight$'

CLI (multiple patterns, OR'd):
    python hf_to_fp8.py <bf16-path> <fp8-path> \
        -p 'model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj\.weight$' \
        -p 'model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$'
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def convert(
    source: Path,
    target: Path,
    should_quantize: Callable[[str], bool],
    *,
    block_size: int = 128,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
    max_workers: int = 16,
) -> None:
    """Quantize an HF safetensors checkpoint to FP8 in-place into ``target``.

    Args:
        source (Path): Source checkpoint directory containing
            ``model.safetensors.index.json`` and the referenced shards.
        target (Path): Destination directory. Will be created if missing.
            Shard filenames are preserved from the source — call the repacker
            afterwards if the source uses non-standard naming.
        should_quantize (Callable[[str], bool]): Predicate that, given a
            tensor name, returns whether that tensor should be quantized.
        block_size (int): Block size for per-block scaling (default 128).
        float8_dtype (torch.dtype): Target FP8 dtype (default ``e4m3fn``).
        max_workers (int): Max parallel save workers.
    """
    with open(source / "model.safetensors.index.json") as f:
        index = json.load(f)

    target.mkdir(parents=True, exist_ok=True)

    original_weight_map = index.pop("weight_map")

    # Decide quantization status per module up front so we can emit a complete
    # ``modules_to_not_convert`` list in the HF quantization_config.
    modules_to_not_convert: set[str] = set()
    for param_name in original_weight_map:
        module_name = param_name.rsplit(".", 1)[0]
        if not should_quantize(param_name):
            modules_to_not_convert.add(module_name)

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [block_size, block_size],
        "modules_to_not_convert": sorted(modules_to_not_convert),
    }

    executor = ProcessPoolExecutor(max_workers=max_workers)
    new_weight_map: dict[str, str] = {}

    for filename in tqdm(sorted(set(original_weight_map.values()))):
        filepath = source / filename
        safetensor_fh = safe_open(filepath, framework="pt")
        new_shard: dict[str, torch.Tensor] = {}

        for key in safetensor_fh.keys():
            new_weight_map[key] = filename
            tensor = safetensor_fh.get_tensor(key)
            if not should_quantize(key):
                new_shard[key] = tensor
                continue

            fp8_tensor, scale = per_block_quant_torch(
                tensor.cuda(), block_size=block_size, float8_dtype=float8_dtype
            )
            scale_key = f"{key}_scale_inv"
            new_shard[key] = fp8_tensor.cpu()
            new_shard[scale_key] = scale.cpu()
            new_weight_map[scale_key] = filename

        executor.submit(save_file, new_shard, target / filename)

    executor.shutdown()
    _copy_others(source, target)

    index["weight_map"] = new_weight_map
    with open(target / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    with open(source / "config.json") as f:
        hf_config = json.load(f)
    hf_config["quantization_config"] = quantization_config
    with open(target / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)


def compile_union(patterns: list[str]) -> re.Pattern:
    """Compile a list of regexes into a single union pattern.

    Each input is wrapped in a non-capturing group so caller-side groups don't
    bleed across patterns, then OR'd together.

    Args:
        patterns (list[str]): Regex strings to union.

    Returns:
        re.Pattern: A compiled regex equivalent to ``(?:p1)|(?:p2)|...``.
    """
    if len(patterns) == 1:
        return re.compile(patterns[0])
    return re.compile("|".join(f"(?:{p})" for p in patterns))


@torch.no_grad()
def per_block_quant_torch(
    tensor: torch.Tensor,
    block_size: int = 128,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block FP8 quantization for 2D weights and 3D fused-expert weights.

    For 3D inputs of shape ``(num_experts, dim0, dim1)``, each expert slice is
    quantized independently and the per-slice scale is stacked along a leading
    dim, matching the layout vLLM / SGLang expect for fused MoE expert
    weights.

    Args:
        tensor (torch.Tensor): Input weight tensor (bf16/fp16/fp32), 2D or 3D.
        block_size (int): Block size on each quantized dim.
        float8_dtype (torch.dtype): Target FP8 dtype.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(fp8_tensor, scale_inv)``.
    """
    if tensor.dim() == 2:
        return _per_block_quant_2d(tensor, block_size, float8_dtype)
    if tensor.dim() == 3:
        fp8_slices: list[torch.Tensor] = []
        scale_slices: list[torch.Tensor] = []
        for expert_idx in range(tensor.shape[0]):
            t, s = _per_block_quant_2d(tensor[expert_idx], block_size, float8_dtype)
            fp8_slices.append(t)
            scale_slices.append(s)
        return torch.stack(fp8_slices, dim=0), torch.stack(scale_slices, dim=0)
    raise ValueError(f"per_block_quant_torch only supports 2D or 3D tensors, got shape {tuple(tensor.shape)}")


def _per_block_quant_2d(tensor: torch.Tensor, block_size: int, float8_dtype: torch.dtype):
    dim0, dim1 = tensor.shape
    tensor_pad = _pad_for_block(tensor, (0, 1), block_size)
    dim0_pad, dim1_pad = tensor_pad.shape
    tensor_pad = (
        tensor_pad.view(dim0_pad // block_size, block_size, dim1_pad // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    amax = tensor_pad.abs().amax(-1, True).to(torch.float64)
    scales = (amax / torch.finfo(float8_dtype).max).to(torch.float32)
    tensor_pad_scaled = tensor_pad.float() / scales
    fp8 = _to_fp8_saturated(tensor_pad_scaled, float8_dtype)
    fp8 = (
        fp8.view(dim0_pad // block_size, dim1_pad // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dim0_pad, dim1_pad)
    )
    scales = scales.view(dim0_pad // block_size, dim1_pad // block_size)
    return fp8[:dim0, :dim1], scales


def _pad_for_block(tensor: torch.Tensor, dims, block_size: int) -> torch.Tensor:
    assert tensor.dim() == 2
    if isinstance(dims, int):
        dims = (dims,)
    dim1, dim2 = tensor.shape
    dim1_aligned = _align(dim1, block_size) if 0 in dims else dim1
    dim2_aligned = _align(dim2, block_size) if 1 in dims else dim2
    return torch.nn.functional.pad(tensor, (0, dim2_aligned - dim2, 0, dim1_aligned - dim1))


def _align(size: int, alignment: int) -> int:
    return (1 + ((size - 1) // alignment)) * alignment


def _to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype) -> torch.Tensor:
    # PyTorch's default cast to float8_e4m3fn / e5m2 does not saturate; we
    # clamp first so that out-of-range values become +/- max instead of NaN.
    if float8_dtype not in FP8_TYPES:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")
    max_value = torch.finfo(float8_dtype).max
    return x.clamp(min=-max_value, max=max_value).to(float8_dtype)


def _copy_others(source: Path, target: Path) -> None:
    # Tensor shards, index, and config.json are produced by the conversion
    # itself, so they are excluded. Everything else at the top level is
    # mirrored over — files via copy2 to preserve mtime, directories
    # recursively (e.g. ``figs/`` in HF model dirs). Existing entries in
    # target are overwritten so reruns are idempotent.
    for entry in source.iterdir():
        if entry.name.endswith("safetensors"):
            continue
        if entry.name.startswith("."):
            continue
        if entry.name == "model.safetensors.index.json":
            continue
        if entry.name == "config.json":
            continue
        dst = target / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dst)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF bf16 -> FP8 per-block quantization")
    parser.add_argument("source", type=Path, help="source HF model directory")
    parser.add_argument("target", type=Path, help="target HF model directory")
    parser.add_argument(
        "regex",
        nargs="*",
        type=str,
        help="one or more regex patterns; tensor is quantized iff any pattern matches",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        action="append",
        default=[],
        help="alternative to positional regex; can be repeated",
    )
    args = parser.parse_args()
    args.regex = list(args.regex) + list(args.pattern)
    if not args.regex:
        parser.error("at least one regex pattern is required (positional or via --pattern)")
    return args


def main() -> None:
    args = _get_args()
    pattern = compile_union(args.regex)
    convert(args.source, args.target, should_quantize=lambda k: pattern.search(k) is not None)


if __name__ == "__main__":
    main()
