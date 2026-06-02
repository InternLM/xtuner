"""Convert a Step-3.5-Flash HF checkpoint to a *split* per-expert layout.

The released checkpoint stores each MoE layer's experts as three fused 3-D tensors
(`moe.gate_proj/up_proj/down_proj.weight`, shape `(num_experts, *, *)`). XTuner fuses experts
expert-major-interleaved for its grouped GEMM, and the load/save path can only shard a fused
parameter when each HF key is a *contiguous* slice of it. A single fused weight that maps to two
separate HF tensors (gate, up) therefore cannot be sharded across ranks.

This script explodes the fused expert tensors into per-expert 2-D tensors under
`moe.experts.{i}.{gate,up,down}_proj.weight` (Qwen3-MoE style). With that layout XTuner's
`to_hf_key_list` emits the interleaved key order `[gate_0, up_0, gate_1, up_1, ...]`, which lines up
with its expert-major fused weight, so the default sharded load/save works on any number of
GPUs (FSDP and EP) with no per-model checkpoint code.

The MTP layers (`model.layers.45..47`) are dropped — XTuner does not load them.

Usage:
    python .dev_scripts/convert_step3p5_to_split.py <src_hf_dir> <dst_dir>
"""

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SHARD_BYTES = 4 * 1024**3  # ~4GB shards, HF-standard
# Side files needed so the converted dir is loadable (AutoConfig trust_remote_code + tokenizer).
AUX_FILES = [
    "config.json",
    "configuration_step3p5.py",
    "modeling_step3p5.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
]
DROP_LAYER_RE = re.compile(r"^model\.layers\.(4[5-9]|[5-9]\d)\.")  # MTP / out-of-range layers
EXPERT_RE = re.compile(r"^(model\.layers\.\d+)\.moe\.(gate_proj|up_proj|down_proj)\.weight$")


def _iter_converted_tensors(src: Path):
    """Yield (new_key, tensor) for every kept tensor, exploding fused experts into per-expert keys."""
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    # Group keys by source shard so each shard is opened once.
    by_file: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        by_file.setdefault(fname, []).append(key)

    for fname in sorted(by_file):
        with safe_open(str(src / fname), framework="pt") as f:
            for key in by_file[fname]:
                if DROP_LAYER_RE.match(key):
                    continue
                tensor = f.get_tensor(key)
                m = EXPERT_RE.match(key)
                if m is None:
                    yield key, tensor
                    continue
                prefix, proj = m.group(1), m.group(2)
                # gate/up: (n, inter, hidden) -> per expert (inter, hidden)
                # down:    (n, hidden, inter) -> per expert (hidden, inter)
                for i in range(tensor.shape[0]):
                    yield f"{prefix}.moe.experts.{i}.{proj}.weight", tensor[i].contiguous()


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    shard_idx = 1
    shards: list[tuple[str, dict[str, torch.Tensor]]] = []

    def flush():
        nonlocal buffer, buffer_bytes, shard_idx
        if not buffer:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        shards.append((name, buffer))
        for k in buffer:
            weight_map[k] = name
        buffer = {}
        buffer_bytes = 0
        shard_idx += 1

    for key, tensor in _iter_converted_tensors(src):
        buffer[key] = tensor
        buffer_bytes += tensor.numel() * tensor.element_size()
        if buffer_bytes >= SHARD_BYTES:
            flush()
    flush()

    total = sum(t.numel() * t.element_size() for _, b in shards for t in b.values())
    n_shards = len(shards)
    renamed: list[tuple[str, dict[str, torch.Tensor]]] = []
    for i, (_, buf) in enumerate(shards, start=1):
        final = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        for k in buf:
            weight_map[k] = final
        renamed.append((final, buf))
    for name, buf in renamed:
        save_file(buf, str(dst / name), metadata={"format": "pt"})
        print(f"  wrote {name}  ({len(buf)} tensors)")

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    for aux in AUX_FILES:
        srcf = src / aux
        if srcf.exists():
            shutil.copy2(srcf, dst / aux)
    print(f"done: {len(weight_map)} tensors across {n_shards} shards, {total / 1024**3:.1f} GiB -> {dst}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
