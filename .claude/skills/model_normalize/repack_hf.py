"""Repack a safetensors model directory into HF-standard ~4GB shards.

Training engines may save safetensors with engine-specific filenames
(e.g. ``model-language-0001-fused-save_rank0.safetensors``). HF inference
backends expect the standard ``model-{i:05d}-of-{n:05d}.safetensors``
layout. This script rewrites shards in place into the standard layout,
preserves all non-shard files (config, tokenizer, etc.), and emits a fresh
``model.safetensors.index.json``.

Library:
    from repack_hf import repack
    repack(source, target, shard_size_bytes=4 * 1024**3)

CLI:
    python repack_hf.py <src-dir> <dst-dir> [--shard-size-gb 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# safetensors dtype string -> bytes per element. Covers the dtypes we emit.
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass
class _TensorEntry:
    name: str
    source_file: str
    nbytes: int


def repack(source: Path, target: Path, shard_size_bytes: int = 4 * 1024**3) -> None:
    """Repack ``source`` safetensors into ~``shard_size_bytes`` standard shards.

    Args:
        source (Path): Source model directory.
        target (Path): Output directory; created if missing. Must differ from
            ``source`` (we never rewrite shards in place to keep the source
            recoverable on failure).
        shard_size_bytes (int): Soft upper bound for each output shard. A
            tensor larger than this bound goes into its own shard.
    """
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("source and target must differ")
    target.mkdir(parents=True, exist_ok=True)

    entries = _scan_tensors(source)
    plan = _plan_shards(entries, shard_size_bytes)
    num_shards = len(plan)

    new_weight_map: dict[str, str] = {}
    total_size = 0

    for shard_idx, shard in enumerate(tqdm(plan, desc="writing shards")):
        shard_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        # Group tensors in this shard by source file so we open each source
        # shard at most once per output shard.
        by_source: dict[str, list[str]] = {}
        for entry in shard:
            by_source.setdefault(entry.source_file, []).append(entry.name)

        tensors = {}
        for src_file, names in by_source.items():
            with safe_open(source / src_file, framework="pt") as f:
                for name in names:
                    tensors[name] = f.get_tensor(name)

        save_file(tensors, target / shard_name)

        for entry in shard:
            new_weight_map[entry.name] = shard_name
            total_size += entry.nbytes

    _copy_non_shard_files(source, target)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(target / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


def _scan_tensors(source: Path) -> list[_TensorEntry]:
    # Prefer the index when available — it gives a stable ordering and avoids
    # scanning files we never intended to read.
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        files_to_keys: dict[str, list[str]] = {}
        for name, file in weight_map.items():
            files_to_keys.setdefault(file, []).append(name)
    else:
        files_to_keys = {p.name: None for p in sorted(source.glob("*.safetensors"))}  # type: ignore[assignment]

    entries: list[_TensorEntry] = []
    for src_file, names in files_to_keys.items():
        with safe_open(source / src_file, framework="pt") as f:
            keys = names if names is not None else list(f.keys())
            for key in keys:
                slice_ = f.get_slice(key)
                shape = slice_.get_shape()
                dtype = slice_.get_dtype()
                numel = 1
                for d in shape:
                    numel *= d
                nbytes = numel * _DTYPE_BYTES[dtype]
                entries.append(_TensorEntry(name=key, source_file=src_file, nbytes=nbytes))
    return entries


def _plan_shards(entries: list[_TensorEntry], shard_size_bytes: int) -> list[list[_TensorEntry]]:
    # Greedy bin-packing. We preserve the input order (which mirrors the
    # source index order) so that produced shards stay roughly contiguous in
    # the original layer order — friendlier for streamed loading.
    shards: list[list[_TensorEntry]] = []
    current: list[_TensorEntry] = []
    current_size = 0
    for entry in entries:
        if current and current_size + entry.nbytes > shard_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += entry.nbytes
    if current:
        shards.append(current)
    return shards


def _copy_non_shard_files(source: Path, target: Path) -> None:
    for file in source.iterdir():
        if file.name.endswith(".safetensors"):
            continue
        if file.name == "model.safetensors.index.json":
            continue
        if file.name.startswith("."):
            continue
        dst = target / file.name
        if file.is_dir():
            if not dst.exists():
                shutil.copytree(file, dst)
        else:
            shutil.copy(file, dst)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack safetensors into 4GB-shard HF layout")
    parser.add_argument("source", type=Path, help="source model directory")
    parser.add_argument("target", type=Path, help="target model directory")
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=4.0,
        help="soft upper bound per output shard, in GiB (default: 4)",
    )
    return parser.parse_args()


def main() -> None:
    args = _get_args()
    shard_size_bytes = int(args.shard_size_gb * 1024**3)
    repack(args.source, args.target, shard_size_bytes=shard_size_bytes)


if __name__ == "__main__":
    main()
