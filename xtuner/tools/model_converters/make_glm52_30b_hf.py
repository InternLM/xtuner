# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python3
"""Build GLM-5.2 ~30B HF checkpoints for single-node 32k validation."""

import argparse
import json
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"
LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class CropProfile:
    num_main_layers: int
    include_mtp: bool


PROFILES = {
    # 3 dense + 2 MoE + original final MTP layer ~= 32.9B parameters.
    "30b-with-mtp": CropProfile(
        num_main_layers=5,
        include_mtp=True,
    ),
    # 3 dense + 3 MoE ~= 32.7B parameters.
    "30b-no-mtp": CropProfile(
        num_main_layers=6,
        include_mtp=False,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original GLM-5.2 HF checkpoint directory")
    parser.add_argument("--save", type=Path, required=True, help="Directory to write the cropped HF checkpoint")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="30b-with-mtp")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected tensors and estimated size")
    return parser.parse_args()


def _layer_id(name: str) -> int | None:
    match = LAYER_KEY_RE.match(name)
    return int(match.group(1)) if match is not None else None


def _target_weight_name(name: str, *, num_main_layers: int, original_main_layers: int, include_mtp: bool) -> str | None:
    layer_id = _layer_id(name)
    if layer_id is None:
        return name
    if layer_id < num_main_layers:
        return name
    if include_mtp and layer_id == original_main_layers:
        return LAYER_KEY_RE.sub(f"model.layers.{num_main_layers}.", name, count=1)
    return None


def _copy_metadata_files(source: Path, target: Path):
    for path in source.iterdir():
        if path.name.endswith(".safetensors") or path.name.endswith(".safetensors.index.json"):
            continue
        if path.is_dir():
            shutil.copytree(path, target / path.name, symlinks=False)
        else:
            shutil.copy2(path, target / path.name)


def _write_config(source: Path, target: Path, profile: CropProfile):
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    original_main_layers = int(config["num_hidden_layers"])
    if profile.num_main_layers < 1 or profile.num_main_layers > original_main_layers:
        raise ValueError(f"num_main_layers must be in [1, {original_main_layers}], got {profile.num_main_layers}")

    config["num_hidden_layers"] = profile.num_main_layers
    config["first_k_dense_replace"] = min(int(config.get("first_k_dense_replace", 0)), profile.num_main_layers)
    config["num_nextn_predict_layers"] = 1 if profile.include_mtp else 0

    for key in ("mlp_layer_types", "indexer_types"):
        if isinstance(config.get(key), list):
            config[key] = config[key][: profile.num_main_layers]

    if profile.include_mtp and isinstance(config.get("indexer_types"), list):
        # The original GLM-5.2 MTP layer is stored as model.layers.{num_hidden_layers}
        # and carries its own DSA indexer weights. After remapping it after the cropped
        # main stack, keep an explicit full indexer entry so XTuner builds the same
        # parameter surface for strict HF loading/saving.
        config["indexer_types"].append("full")

    (target / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return original_main_layers


def _selected_weight_map(index_path: Path, *, num_main_layers: int, original_main_layers: int, include_mtp: bool):
    index = json.loads(index_path.read_text(), object_pairs_hook=OrderedDict)
    selected = OrderedDict()
    for source_name, shard in index["weight_map"].items():
        target_name = _target_weight_name(
            source_name,
            num_main_layers=num_main_layers,
            original_main_layers=original_main_layers,
            include_mtp=include_mtp,
        )
        if target_name is not None:
            selected[source_name] = (target_name, shard)
    if not selected:
        raise ValueError("no tensors selected; check crop profile")
    return selected


def _group_by_shard(weight_map: OrderedDict[str, tuple[str, str]]):
    grouped: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    for source_name, (target_name, shard) in weight_map.items():
        grouped.setdefault(shard, []).append((source_name, target_name))
    return grouped


def _write_sharded_weights(source: Path, target: Path, grouped: OrderedDict[str, list[tuple[str, str]]]):
    total_size = 0
    new_weight_map = OrderedDict()
    num_shards = len(grouped)

    for shard_id, (source_shard, tensor_names) in enumerate(grouped.items(), 1):
        output_shard = f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
        tensors = {}
        with safe_open(source / source_shard, framework="pt", device="cpu") as reader:
            for source_name, target_name in tensor_names:
                tensor = reader.get_tensor(source_name)
                tensors[target_name] = tensor
                total_size += tensor.numel() * tensor.element_size()
                new_weight_map[target_name] = output_shard

        # Keep slime's shard-at-a-time pattern so the 30B crop never materializes
        # all selected tensors in host memory at once.
        save_file(tensors, target / output_shard)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
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
    profile = PROFILES[args.profile]
    index_path = args.source / INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")

    source_config = json.loads((args.source / "config.json").read_text())
    original_main_layers = int(source_config["num_hidden_layers"])
    selected = _selected_weight_map(
        index_path,
        num_main_layers=profile.num_main_layers,
        original_main_layers=original_main_layers,
        include_mtp=profile.include_mtp,
    )
    grouped = _group_by_shard(selected)

    if args.dry_run:
        print(f"source={args.source}")
        print(f"save={args.save}")
        print(f"profile={args.profile}")
        print(f"num_main_layers={profile.num_main_layers}")
        print(f"include_mtp={profile.include_mtp}")
        print(f"selected_tensors={len(selected)}")
        print(f"source_shards={len(grouped)}")
        return

    _prepare_target(args.save, args.overwrite)
    _copy_metadata_files(args.source, args.save)
    _write_config(args.source, args.save, profile)
    total_size, num_tensors, num_shards = _write_sharded_weights(args.source, args.save, grouped)

    print(f"profile={args.profile}")
    print(f"wrote {num_tensors} tensors in {num_shards} shards to {args.save}")
    print(f"total_size={total_size} bytes")


if __name__ == "__main__":
    main()
