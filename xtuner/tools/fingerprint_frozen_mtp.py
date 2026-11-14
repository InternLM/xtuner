"""Identify frozen draft tensor contents, independently of path/helper hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = args.source / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"]
    keys = sorted(key for key in mapping if key.startswith("mtp."))
    if not keys:
        raise ValueError("No dedicated MTP tensors in source")
    tensors = {}
    for shard in sorted({mapping[key] for key in keys}):
        with safe_open(args.source / shard, framework="pt", device="cpu") as stream:
            for key in keys:
                if mapping[key] != shard:
                    continue
                tensor = stream.get_tensor(key).contiguous()
                raw = tensor.view(torch.uint8).numpy()
                tensors[key] = dict(
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    nbytes=tensor.numel() * tensor.element_size(),
                    sha256=hashlib.sha256(memoryview(raw).cast("B")).hexdigest(),
                )
                print(key, tensors[key]["nbytes"], flush=True)
                del raw, tensor
    # Identity is independent of shard filenames or local mount paths.
    content = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = dict(
        source=str(args.source.resolve()),
        index_sha256=file_hash(index),
        config_sha256=file_hash(args.source / "config.json"),
        mtp_content_sha256=content,
        tensors=tensors,
        total_bytes=sum(item["nbytes"] for item in tensors.values()),
        scope="CPU source identity only; no engine transfer, shared target or GPU correctness claim",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("MTP content identity", content, flush=True)


if __name__ == "__main__":
    main()
