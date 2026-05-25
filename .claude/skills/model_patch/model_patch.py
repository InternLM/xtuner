"""Merge an ``extra`` HF model dir into a ``base`` HF model dir.

Produces a new ``out`` directory whose ``model.safetensors.index.json`` is
``base.index ∪ (extra.index \\ base.index)`` — i.e. ``extra`` only contributes
tensors that ``base`` does not already have. Overlapping keys are kept from
``base``.

Shards in ``out`` follow the HF-standard ``model-{i:05d}-of-{N:05d}.safetensors``
layout:
- Base shards are re-emitted with the new total ``N``. By default the bytes
  are copied (``out`` is fully independent of ``base``); pass ``--hardlink``
  to use ``os.link`` instead, which avoids duplicating bytes on the same
  filesystem at the cost of coupling the two trees.
- Tensors only present in ``extra`` are packed into freshly-written shards
  appended after the base shards.

Non-tensor files (``config.json``, tokenizer, modeling/configuration ``.py``,
generation config, etc.) are mirrored from ``base`` into ``out`` so the
result is loadable on its own. ``extra``'s aux files are ignored on purpose:
the overlap rule says base wins, and that extends to configs — mixing in
``extra``'s config tends to drag along engine-specific edits or modality
keys that don't match base's tensor topology.

CLI::

    python model_patch.py \\
        --base  /path/to/base-model \\
        --extra /path/to/extra-model \\
        --out   /path/to/merged-model
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# safetensors dtype string -> bytes per element. Matches repack_hf.py.
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


def patch(
    base: Path,
    extra: Path,
    out: Path,
    shard_size_bytes: int = 4 * 1024**3,
    hardlink: bool = False,
) -> None:
    """Merge tensors from ``extra`` into ``base`` and write to ``out``.

    Args:
        base (Path): Authoritative source. Every tensor in its index lands in
            ``out`` unchanged.
        extra (Path): Donor. Only tensors *not* present in base's index are
            taken from here; overlapping names are silently ignored.
        out (Path): Output directory. Created if missing. Must differ from
            both ``base`` and ``extra``.
        shard_size_bytes (int): Soft upper bound for shards holding new
            (extra-only) tensors. Base shards are preserved as-is and not
            re-bin-packed.
        hardlink (bool): If True, base shards are placed into ``out`` via
            ``os.link`` (cheap, but ``out`` shares inodes with ``base``).
            Defaults to False — bytes are fully copied, so ``out`` is an
            independent tree.
    """
    base = base.resolve()
    extra = extra.resolve()
    out = out.resolve()
    if out in (base, extra):
        raise ValueError("out must differ from base and extra")
    out.mkdir(parents=True, exist_ok=True)

    base_map = _load_weight_map(base)
    extra_map = _load_weight_map(extra)

    base_shards = _ordered_unique(base_map.values())
    extra_only_keys = [k for k in extra_map.keys() if k not in base_map]

    extra_entries = _entries_for(extra, extra_only_keys, extra_map)
    new_shard_plan = _plan_shards(extra_entries, shard_size_bytes)

    num_base_shards = len(base_shards)
    num_new_shards = len(new_shard_plan)
    total_shards = num_base_shards + num_new_shards

    if total_shards == 0:
        raise ValueError("nothing to write — base and extra are both empty")

    print(
        f"[model_patch] base={len(base_map)} tensors / {num_base_shards} shards, "
        f"extra adds {len(extra_only_keys)} tensors / {num_new_shards} shards "
        f"(ignored {len(extra_map) - len(extra_only_keys)} overlapping keys)"
    )

    new_weight_map: dict[str, str] = {}
    total_size = 0

    # 1) Re-emit base shards under the new -of-N name. Default is a real copy
    #    so that out is independent of base; --hardlink trades that
    #    independence for zero-cost placement on the same filesystem.
    base_rename: dict[str, str] = {}
    place = _hardlink if hardlink else _copy
    desc = "linking base shards" if hardlink else "copying base shards"
    for i, src_name in enumerate(tqdm(base_shards, desc=desc)):
        new_name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        base_rename[src_name] = new_name
        place(base / src_name, out / new_name)

    for key, src_file in base_map.items():
        new_weight_map[key] = base_rename[src_file]
        total_size += _tensor_nbytes(base / src_file, key)

    # 2) Write the new shards from extra-only tensors after the base shards.
    for j, shard in enumerate(tqdm(new_shard_plan, desc="writing extra shards")):
        shard_idx = num_base_shards + j + 1
        shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"

        by_source: dict[str, list[str]] = {}
        for entry in shard:
            by_source.setdefault(entry.source_file, []).append(entry.name)

        tensors = {}
        for src_file, names in by_source.items():
            with safe_open(extra / src_file, framework="pt") as f:
                for name in names:
                    tensors[name] = f.get_tensor(name)
        save_file(tensors, out / shard_name)

        for entry in shard:
            new_weight_map[entry.name] = shard_name
            total_size += entry.nbytes

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Mirror base's non-tensor files (config.json, tokenizer, modeling .py, ...).
    # Base wins on overlap at the tensor level, and the same applies to configs:
    # extra's aux files may carry engine-specific edits or describe a modality
    # whose tensor layout we did not adopt.
    _copy_aux_files(base, out)


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    # Require an explicit index — model_patch operates strictly on index diffs,
    # so a missing index is a hard error rather than a "scan shards" fallback.
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found; model_patch requires an index file")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def _ordered_unique(items) -> list[str]:
    # Preserve first-seen order — base shard order matters for downstream
    # streamed loading and for keeping the new -of-N numbering stable.
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen.keys())


def _entries_for(model_dir: Path, keys: list[str], weight_map: dict[str, str]) -> list[_TensorEntry]:
    if not keys:
        return []
    # Read sizes per source file, opening each file at most once.
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    entries: list[_TensorEntry] = []
    for src_file, names in by_file.items():
        with safe_open(model_dir / src_file, framework="pt") as f:
            for name in names:
                entries.append(_TensorEntry(name=name, source_file=src_file, nbytes=_slice_nbytes(f, name)))
    return entries


def _plan_shards(entries: list[_TensorEntry], shard_size_bytes: int) -> list[list[_TensorEntry]]:
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


def _copy_aux_files(base: Path, out: Path) -> None:
    # Tensor shards and the index are produced by this script; everything else
    # at the top level of base is treated as auxiliary state and mirrored into
    # out. Files are copied with shutil.copy2 to preserve mtime; directories
    # (e.g. ``figs/``) are copied recursively. Existing entries in out are
    # overwritten so reruns are idempotent.
    copied = 0
    for entry in base.iterdir():
        if entry.name == "model.safetensors.index.json":
            continue
        if entry.suffix == ".safetensors":
            continue
        dst = out / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dst)
        copied += 1
    print(f"[model_patch] copied {copied} aux entries from base")


def _copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    shutil.copy(src, dst)


def _hardlink(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        # Cross-device or filesystem without hardlink support — caller opted
        # into hardlink, so degrade to copy rather than fail outright.
        shutil.copy(src, dst)


def _slice_nbytes(opened, key: str) -> int:
    s = opened.get_slice(key)
    shape = s.get_shape()
    numel = 1
    for d in shape:
        numel *= d
    return numel * _DTYPE_BYTES[s.get_dtype()]


def _tensor_nbytes(file_path: Path, key: str) -> int:
    with safe_open(file_path, framework="pt") as f:
        return _slice_nbytes(f, key)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge new tensors from extra HF dir into base HF dir")
    parser.add_argument("--base", type=Path, required=True, help="authoritative base model dir")
    parser.add_argument("--extra", type=Path, required=True, help="donor model dir (only new tensors are taken)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=4.0,
        help="soft upper bound per new shard, in GiB (default: 4)",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="place base shards via os.link instead of copying (default: copy)",
    )
    return parser.parse_args()


def main() -> None:
    args = _get_args()
    patch(
        args.base,
        args.extra,
        args.out,
        shard_size_bytes=int(args.shard_size_gb * 1024**3),
        hardlink=args.hardlink,
    )


if __name__ == "__main__":
    main()
