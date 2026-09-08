"""Command-line entry point for HF checkpoint normalization.

This tool intentionally does conversion/repacking only. Expensive full-model
validation is a separate release concern and is not run automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size-gb", type=float, default=float(os.getenv("MODEL_NORMALIZE_SHARD_SIZE_GB", "4")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generation-config", type=Path, default=None)
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=None,
        help="directory with user-supplied LICENSE and generation_config.json "
        "to apply after conversion; the Copyright line of LICENSE is rewritten "
        "to 'Shanghai AI Laboratory'",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize HF checkpoints for XTuner/HF inference")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("repack", help="re-shard an existing HF checkpoint")
    _add_common(rep)
    rep.set_defaults(handler=_run_repack)

    fp8 = sub.add_parser("to-fp8", help="convert BF16/FP16 weights to per-block FP8")
    _add_common(fp8)
    policy = fp8.add_mutually_exclusive_group(required=True)
    policy.add_argument("--reference", type=Path, help="FP8 reference checkpoint")
    policy.add_argument("--policy", choices=["heuristic"], help="explicit fallback quantization policy")
    fp8.add_argument(
        "--max-save-workers",
        type=int,
        default=int(os.getenv("MODEL_NORMALIZE_MAX_SAVE_WORKERS", "4")),
    )
    fp8.add_argument("--block-size", type=int, default=128)
    fp8.set_defaults(handler=_run_fp8)
    return parser


def _prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if source == output:
        raise ValueError("source and output must be different directories")
    if output.exists():
        if not overwrite and any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty; pass --overwrite: {output}")
        if overwrite:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _copy_generation_config(path: Path | None, output: Path) -> None:
    if path is None:
        return
    if not path.is_file():
        raise FileNotFoundError(f"generation config does not exist: {path}")
    shutil.copy2(path, output / "generation_config.json")


# Fixed copyright line stamped onto a user-supplied LICENSE. The year range is
# a literal, not derived from the input, so the release copyright is stable.
_LICENSE_COPYRIGHT_LINE = "Copyright 2025-2026 Shanghai AI Laboratory"


def _rewrite_license_holder(text: str) -> tuple[str, bool]:
    """Rewrite the Copyright holder line in a LICENSE text.

    Only the first line that starts with ``Copyright`` (after stripping leading
    whitespace) is replaced with the fixed Shanghai AI Laboratory copyright line;
    the rest of the text (e.g. the MIT permission grant) is left untouched.

    Args:
        text (str): Original LICENSE content.

    Returns:
        tuple[str, bool]: The rewritten text and whether a Copyright line was
        found and replaced.
    """
    lines = text.splitlines(keepends=True)
    replaced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Copyright"):
            # Preserve the original line ending style of the replaced line.
            newline = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
            lines[i] = f"{_LICENSE_COPYRIGHT_LINE}{newline}" if newline else _LICENSE_COPYRIGHT_LINE
            replaced = True
            break
    return "".join(lines), replaced


def _apply_base_model_assets(base_dir: Path | None, output: Path) -> None:
    """Apply user-supplied assets from ``base_dir`` into ``output``.

    ``base_dir`` may contain a ``generation_config.json`` (copied verbatim,
    overwriting any existing one) and a ``LICENSE`` (copied with its Copyright
    line rewritten to Shanghai AI Laboratory). Missing files are skipped
    silently; a ``None`` directory is a no-op so existing behavior is preserved.

    Args:
        base_dir (Path | None): User-supplied base-model assets directory.
        output (Path): Conversion product directory.
    """
    if base_dir is None:
        return
    base_dir = base_dir.resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base-model directory does not exist: {base_dir}")

    gen_src = base_dir / "generation_config.json"
    if gen_src.is_file():
        gen_dst = output / "generation_config.json"
        if gen_dst.exists():
            print("[model_normalize] overwriting existing generation_config.json from base-model-dir")
        shutil.copy2(gen_src, gen_dst)

    license_src = base_dir / "LICENSE"
    if license_src.is_file():
        license_dst = output / "LICENSE"
        if license_dst.exists():
            print("[model_normalize] overwriting existing LICENSE from base-model-dir")
        text = license_src.read_text(encoding="utf-8")
        rewritten, replaced = _rewrite_license_holder(text)
        if not replaced:
            print("[model_normalize] warning: no Copyright line found in base-model LICENSE; writing unchanged")
        license_dst.write_text(rewritten, encoding="utf-8")


def _run_repack(args: argparse.Namespace) -> None:
    try:
        from .repack import repack
    except ModuleNotFoundError as exc:
        raise RuntimeError("repack requires the 'safetensors' package") from exc
    _prepare_output(args.source, args.output, args.overwrite)
    repack(args.source, args.output, shard_size_bytes=int(args.shard_size_gb * 1024**3))
    _copy_generation_config(args.generation_config, args.output)
    _apply_base_model_assets(args.base_model_dir, args.output)


def _reference_predicate(reference: Path):
    with open(reference / "model.safetensors.index.json") as f:
        keys = set(json.load(f)["weight_map"])
    suffix = "_scale_inv"
    names = {key[: -len(suffix)] for key in keys if key.endswith(suffix)}
    return names.__contains__


def _run_fp8(args: argparse.Namespace) -> None:
    try:
        import torch

        from .fp8 import convert
        from .heuristics import DEFAULT_QUANTIZE_PATTERNS, build_heuristic_predicate
        from .repack import repack
    except ModuleNotFoundError as exc:
        raise RuntimeError("FP8 conversion requires torch, safetensors, and tqdm") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 conversion requires CUDA; use repack for CPU-only HF sharding")
    if args.max_save_workers < 1:
        raise ValueError("--max-save-workers must be positive")
    if args.block_size < 1:
        raise ValueError("--block-size must be positive")
    _prepare_output(args.source, args.output, args.overwrite)
    if args.reference is not None:
        predicate = _reference_predicate(args.reference)
    else:
        predicate = build_heuristic_predicate()
        print(f"[model_normalize] using heuristic policy ({len(DEFAULT_QUANTIZE_PATTERNS)} patterns)")
    with tempfile.TemporaryDirectory(prefix="model_normalize_fp8_", dir=args.output.parent) as tmp:
        staging = Path(tmp)
        convert(
            args.source,
            staging,
            should_quantize=predicate,
            block_size=args.block_size,
            max_workers=args.max_save_workers,
        )
        repack(staging, args.output, shard_size_bytes=int(args.shard_size_gb * 1024**3))
    _copy_generation_config(args.generation_config, args.output)
    _apply_base_model_assets(args.base_model_dir, args.output)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
