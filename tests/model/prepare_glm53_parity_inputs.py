"""Offline shared VLM inputs, using the actual XTuner GLM53 tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from transformers import AutoTokenizer
from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import Glm53VLTokenizeFunction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    MODEL, OUT, ROOT = args.model, args.output, args.media_root
    if args.limit < 1:
        raise ValueError("limit must be positive")
    OUT.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    fn = Glm53VLTokenizeFunction(tok, MODEL, str(args.jsonl), max_pixels=1048576, max_length=16384)
    samples = []
    with args.jsonl.open() as source:
        for index, line in enumerate(source):
            if index == args.limit:
                break
            record = json.loads(line)
            item = fn(record, media_root=str(ROOT))
            tensors = {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long).reshape(1, -1),
                "labels": torch.tensor(item["labels"], dtype=torch.long).reshape(1, -1),
                "pixel_values": item["pixel_values"].cpu().contiguous(),
                "image_grid_thw": item["image_grid_thw"].cpu().contiguous(),
                "mm_token_type_ids": torch.as_tensor(item["mm_token_type_ids"]).reshape(1, -1).contiguous(),
            }
            pos = torch.nonzero(tensors["labels"][0, 1:] != -100).flatten()[-8:]
            tensors["logit_positions"] = pos
            assert pos.numel() > 0
            assert (
                int((tensors["input_ids"] == fn.processor.image_token_id).sum())
                == int(tensors["image_grid_thw"].prod(dim=-1).sum()) // fn.merge_unit
            )
            path = OUT / f"{index:03d}.safetensors"
            save_file(tensors, str(path), metadata={"shift": "raw_unshifted; use input_ids[:,:-1] and labels[:,1:]"})
            sample = {
                "index": index,
                "id": record["id"],
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "length": tensors["input_ids"].numel(),
                "valid_tokens": int((tensors["labels"][:, 1:] != -100).sum()),
                "logit_positions": pos.tolist(),
                "image_grid_thw": tensors["image_grid_thw"].tolist(),
            }
            samples.append(sample)
            sample["tensors"] = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()
            }
            print(json.dumps(sample), flush=True)
    if len(samples) != args.limit:
        raise ValueError("Insufficient input samples")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "contract": "raw input_ids and raw labels; shift exactly once; logits positions index shifted input; final 8 supervised positions",
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
