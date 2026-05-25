"""Top-level CLI for normalizing an HF model into FP8 / standard layout.

Subcommands:
    to-fp8   Convert a bf16/fp16 HF checkpoint to FP8.
             Two modes:
               (a) --reference <fp8-model-dir>: derive the quantization rule
                   from a reference FP8 model by reading which tensors carry
                   a ``_scale_inv`` companion in its index.
               (b) no reference: apply the built-in heuristic (quantize only
                   the large attention / MLP / MoE-expert linears; keep
                   norms, routers, embeddings, biases, and vision tower).
    repack   Re-shard a safetensors directory into HF-standard ~4GB shards.

End-to-end (FP8 + standard repack):
    python model_normalize.py to-fp8 \
        --source /path/to/bf16-model \
        --output /path/to/fp8-model \
        --reference /path/to/fp8-reference

If the source has non-standard shard filenames, ``to-fp8`` will repack the
output into the HF-standard layout unless ``--no-repack`` is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from hf_to_fp8 import convert
from heuristics import DEFAULT_QUANTIZE_PATTERNS, build_heuristic_predicate
from repack_hf import repack


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize HF model: FP8 quantize and/or repack to standard shards")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fp8 = sub.add_parser("to-fp8", help="Quantize bf16/fp16 -> FP8")
    p_fp8.add_argument("--source", type=Path, required=True, help="source HF model directory")
    p_fp8.add_argument("--output", type=Path, required=True, help="destination directory")
    p_fp8.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="optional reference FP8 model; quantization rule will be derived from its index",
    )
    p_fp8.add_argument(
        "--no-repack",
        action="store_true",
        help="skip the standard-naming repack step even if source uses non-standard shard names",
    )
    p_fp8.add_argument(
        "--shard-size-gb",
        type=float,
        default=4.0,
        help="shard size for the repack step (default: 4)",
    )
    p_fp8.set_defaults(func=_cmd_to_fp8)

    p_rep = sub.add_parser("repack", help="Re-shard into HF-standard ~4GB layout")
    p_rep.add_argument("--source", type=Path, required=True)
    p_rep.add_argument("--output", type=Path, required=True)
    p_rep.add_argument("--shard-size-gb", type=float, default=4.0)
    p_rep.set_defaults(func=_cmd_repack)

    args = parser.parse_args()
    args.func(args)


def _cmd_to_fp8(args: argparse.Namespace) -> None:
    if args.reference is not None:
        names_to_quantize = _extract_quantized_names(args.reference)
        print(f"[model_normalize] derived {len(names_to_quantize)} quantized tensors from reference")
        predicate = names_to_quantize.__contains__
    else:
        print(f"[model_normalize] using built-in heuristic ({len(DEFAULT_QUANTIZE_PATTERNS)} patterns)")
        predicate = build_heuristic_predicate()

    needs_repack = (not args.no_repack) and _has_nonstandard_shards(args.source)

    if needs_repack:
        # Quantize into a scratch dir, then repack to the final output. Using
        # a temp dir keeps the final output clean: callers see exactly the
        # standard layout, never the intermediate engine-named shards.
        with tempfile.TemporaryDirectory(prefix="fp8_quant_", dir=args.output.parent) as tmp:
            tmp_path = Path(tmp)
            convert(args.source, tmp_path, should_quantize=predicate)
            print(f"[model_normalize] repacking to HF-standard layout ({args.shard_size_gb} GiB shards)")
            repack(tmp_path, args.output, shard_size_bytes=int(args.shard_size_gb * 1024**3))
    else:
        convert(args.source, args.output, should_quantize=predicate)


def _cmd_repack(args: argparse.Namespace) -> None:
    repack(args.source, args.output, shard_size_bytes=int(args.shard_size_gb * 1024**3))


def _extract_quantized_names(reference: Path) -> set[str]:
    """Collect names of FP8-quantized tensors from a reference model index.

    A tensor is treated as quantized iff a sibling ``<name>_scale_inv`` exists
    in the same index — this is the contract HF / vLLM / SGLang use for
    per-block FP8.

    Args:
        reference (Path): Reference FP8 model directory.

    Returns:
        set[str]: Tensor names that should be quantized in the source model.
    """
    with open(reference / "model.safetensors.index.json") as f:
        index = json.load(f)
    keys = set(index["weight_map"].keys())
    suffix = "_scale_inv"
    return {k[: -len(suffix)] for k in keys if k.endswith(suffix)}


_STANDARD_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def _has_nonstandard_shards(source: Path) -> bool:
    # Treat any single-file (``model.safetensors``) or standard sharded
    # checkpoint as already-conformant. Anything else (engine-specific
    # rank-based names, custom prefixes) triggers the repack.
    shards = list(source.glob("*.safetensors"))
    if len(shards) == 1 and shards[0].name == "model.safetensors":
        return False
    return any(not _STANDARD_SHARD_RE.match(p.name) for p in shards)


if __name__ == "__main__":
    main()
