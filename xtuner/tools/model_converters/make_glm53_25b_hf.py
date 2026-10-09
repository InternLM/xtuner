# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python3
"""Build a ~25B GLM-5.3-Flash HF checkpoint for single-node validation.

Crops the published FP8 checkpoint down to a handful of main-stack layers (keeping the
original MTP layer, renumbered), dequantizes FP8 block-scaled weights to BF16, and rewrites
the nested ``text_config`` schedule fields so the result loads as a normal ``glm5_next``
checkpoint. See ``doc/xtuner_glm5p3flash_design.md`` section 3.2 (F0) for the design.
"""

import argparse
import json
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"
LAYER_KEY_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
FP8_BLOCK = 128


@dataclass(frozen=True)
class CropProfile:
    num_main_layers: int
    include_mtp: bool


# 3 dense(KDA) + layer3(DSA, sparse) + layer4(KDA, sparse) + original layer 45 (MTP) ~= 24.9B.
PROFILE_25B = CropProfile(num_main_layers=5, include_mtp=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original GLM-5.3-Flash FP8 checkpoint dir")
    parser.add_argument("--save", type=Path, required=True, help="Directory to write the cropped HF checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected tensors and estimated size")
    return parser.parse_args()


def _layer_id(name: str) -> int | None:
    match = LAYER_KEY_RE.match(name)
    return int(match.group(1)) if match is not None else None


def target_name(name: str, *, num_main_layers: int, original_main_layers: int, include_mtp: bool) -> str | None:
    """``model.visual.*`` / ``lm_head`` / embeddings pass through untouched; the main stack is
    truncated to ``num_main_layers`` and the original MTP layer (``original_main_layers``) is
    renumbered to ``layers.{num_main_layers}``."""
    layer_id = _layer_id(name)
    if layer_id is None:
        return name
    if layer_id < num_main_layers:
        return name
    if include_mtp and layer_id == original_main_layers:
        return LAYER_KEY_RE.sub(f"model.language_model.layers.{num_main_layers}.", name, count=1)
    return None


def dequantize_fp8_block(weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """Native FP8 (e4m3) + 128x128 block scale -> BF16.

    ``scale_inv`` has shape ``ceil(out/128) x ceil(in/128)``; expand by block and crop back to
    the weight's shape before multiplying (the last block along each axis may be partial).
    """
    scale = scale_inv.repeat_interleave(block, 0).repeat_interleave(block, 1)
    scale = scale[: weight_fp8.shape[0], : weight_fp8.shape[1]]
    return (weight_fp8.float() * scale).bfloat16()


def _copy_metadata_files(source: Path, target: Path):
    for path in source.iterdir():
        if path.name.endswith(".safetensors") or path.name.endswith(".safetensors.index.json"):
            continue
        if path.is_dir():
            shutil.copytree(path, target / path.name, symlinks=False)
        else:
            shutil.copy2(path, target / path.name)


def rewrite_config(config: dict, profile: CropProfile) -> dict:
    """Crop the nested ``text_config`` schedule lists and rewrite the KDA/DSA layer schedule."""
    tc = config["text_config"]
    n = profile.num_main_layers
    original_main_layers = int(tc["num_hidden_layers"])
    if n < 1 or n > original_main_layers:
        raise ValueError(f"num_main_layers must be in [1, {original_main_layers}], got {n}")

    tc["num_hidden_layers"] = n
    tc["first_k_dense_replace"] = min(int(tc.get("first_k_dense_replace", 0)), n)
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        if isinstance(tc.get(key), list):
            tc[key] = tc[key][:n]

    tc["linear_attn_config"]["kda_layers"] = [i for i, t in enumerate(tc["layer_types"]) if t == "linear_attention"]
    tc["linear_attn_config"]["full_attn_layers"] = [
        i for i, t in enumerate(tc["layer_types"]) if t != "linear_attention"
    ]
    tc["num_nextn_predict_layers"] = 1 if profile.include_mtp else 0

    config.pop("quantization_config", None)  # output is plain BF16
    return config


def _selected_weight_map(index_path: Path, *, num_main_layers: int, original_main_layers: int, include_mtp: bool):
    index = json.loads(index_path.read_text(), object_pairs_hook=OrderedDict)
    selected = OrderedDict()
    for source_name, shard in index["weight_map"].items():
        tgt = target_name(
            source_name,
            num_main_layers=num_main_layers,
            original_main_layers=original_main_layers,
            include_mtp=include_mtp,
        )
        if tgt is not None:
            selected[source_name] = (tgt, shard)
    if not selected:
        raise ValueError("no tensors selected; check crop profile")
    return selected


def _group_by_shard(weight_map: "OrderedDict[str, tuple[str, str]]"):
    grouped: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for source_name, (target, shard) in weight_map.items():
        grouped.setdefault(shard, []).append((source_name, target))
    return grouped


def _write_sharded_weights(source: Path, target: Path, grouped: "OrderedDict[str, list[tuple[str, str]]]"):
    """Dequantize FP8 tensors and write out plain-BF16 sharded safetensors, one source shard at
    a time so the crop never materializes all selected tensors in host memory at once."""
    total_size = 0
    new_weight_map = OrderedDict()
    num_shards = len(grouped)

    for shard_id, (source_shard, tensor_names) in enumerate(grouped.items(), 1):
        output_shard = f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source / source_shard, framework="pt", device="cpu") as reader:
            shard_keys = set(reader.keys())
            for source_name, out_name in tensor_names:
                scale_key = f"{source_name}_scale_inv"
                tensor = reader.get_tensor(source_name)
                if scale_key in shard_keys:
                    tensor = dequantize_fp8_block(tensor, reader.get_tensor(scale_key))
                tensors[out_name] = tensor
                total_size += tensor.numel() * tensor.element_size()
                new_weight_map[out_name] = output_shard

        save_file(tensors, target / output_shard)

    index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    (target / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    return total_size, len(new_weight_map), num_shards


def _prepare_target(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main():
    args = parse_args()
    profile = PROFILE_25B
    index_path = args.source / INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")

    source_config = json.loads((args.source / "config.json").read_text())
    original_main_layers = int(source_config["text_config"]["num_hidden_layers"])
    # weight_scale_inv entries are not real parameters; drop them from selection (handled
    # alongside their base tensor in _write_sharded_weights) so they don't leak into the index.
    selected = {
        name: value
        for name, value in _selected_weight_map(
            index_path,
            num_main_layers=profile.num_main_layers,
            original_main_layers=original_main_layers,
            include_mtp=profile.include_mtp,
        ).items()
        if not name.endswith("_scale_inv")
    }
    grouped = _group_by_shard(selected)

    if args.dry_run:
        print(f"source={args.source}")
        print(f"save={args.save}")
        print(f"num_main_layers={profile.num_main_layers}")
        print(f"include_mtp={profile.include_mtp}")
        print(f"selected_tensors={len(selected)}")
        print(f"source_shards={len(grouped)}")
        return

    _prepare_target(args.save, args.overwrite)
    _copy_metadata_files(args.source, args.save)
    config = rewrite_config(source_config, profile)
    (args.save / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    total_size, num_tensors, num_shards = _write_sharded_weights(args.source, args.save, grouped)

    total_params = total_size / 2  # BF16, 2 bytes/param
    print(f"wrote {num_tensors} tensors in {num_shards} shards to {args.save}")
    print(f"total_size={total_size} bytes, total_params={total_params / 1e9:.2f}B")
    if not (0.9 * 24.9e9 <= total_params <= 1.1 * 24.9e9):
        print("WARNING: total_params deviates from the 24.9B design target by more than 10%")


if __name__ == "__main__":
    main()
