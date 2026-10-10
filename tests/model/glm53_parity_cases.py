"""Shared raw text/image cases for trainer-free GLM-5.3 parity.

This module deliberately does not import XTuner or AutoModel. Each environment
runs the same preprocessing locally; tensor contracts detect version differences.
"""

import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


TEXT_CASE_COUNT = 4


def build_glm53_parity_cases(checkpoint, processor_python=None):
    """Build raw cases locally or in the existing HF processor environment.

    The optional subprocess rebuilds inputs on every invocation. Its tensor-only
    result travels through a pipe, without creating pretokenized fixture files.
    """
    if processor_python is not None:
        result = subprocess.run(
            [str(processor_python), str(Path(__file__).resolve()), str(checkpoint)],
            check=True,
            stdout=subprocess.PIPE,
        )
        return torch.load(io.BytesIO(result.stdout), map_location="cpu", weights_only=True)
    from PIL import Image

    from transformers import AutoProcessor, AutoTokenizer

    text_list = [
        "数据应该像山间的清泉，自然地流向它该去的地方",
        "当异常来临时，就像秋风中飘落的叶子，应该被温柔地接住，而不是粗暴地丢弃",
        "当函数被调用时，它应该像春天的第一缕阳光，温柔地唤醒沉睡的数据结构",
        "就像老树拥抱归巢的鸟儿，内存管理应该给予每个对象足够的安全感",
    ]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    cases = [(f"text-{i}", dict(tokenizer(text, return_tensors="pt"))) for i, text in enumerate(text_list)]
    processor = AutoProcessor.from_pretrained(checkpoint)
    image_path = Path(__file__).resolve().parents[1] / "resource/mscoco_twocat_000000039769.jpg"
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((224, 224))
    # A single image and two distinct images exercise both placeholder spans.
    for name, images in [
        ("image", [image]),
        ("two-images", [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]),
    ]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                + [{"type": "text", "text": "Describe the cats in the image(s)."}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Two cats are resting on a sofa."}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        batch = dict(processor(text=prompt, images=images, return_tensors="pt"))
        assert len(batch["image_grid_thw"]) == len(images)
        expected_tokens = int((batch["image_grid_thw"].prod(-1) // processor.image_processor.merge_size**2).sum())
        assert int((batch["mm_token_type_ids"] == 1).sum()) == expected_tokens
        cases.append((name, batch))
    # Construct labels/positions once; HF and XT receive identical media and supervision.
    for _, batch in cases:
        batch["labels"] = batch["input_ids"].clone()
        if "mm_token_type_ids" in batch:
            batch["labels"][batch["mm_token_type_ids"] != 0] = -100
        batch["positions"] = torch.nonzero(batch["labels"][0, 1:] != -100).flatten()[-8:]
        assert batch["positions"].numel() > 0
    return cases


def case_contract(name, batch):
    """Hash uncast processor tensors, including supervision and sampled positions."""
    tensors = {}
    for key, value in sorted(batch.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}.{key}: expected a tensor")
        value = value.detach().cpu().contiguous()
        tensors[key] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    positions = batch["positions"].cpu()
    labels = batch["labels"][0, 1:].cpu()
    return {
        "name": name,
        "tensors": tensors,
        "positions": positions.tolist(),
        "targets": labels[positions].tolist(),
        "valid_tokens": int((labels != -100).sum()),
    }


def load_glm53_reference(directory, cases, checkpoint):
    """Validate cross-environment inputs before accepting offline AM predictions."""
    from safetensors.torch import load_file

    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata-rank0.json").read_text())
    model_class = metadata.get("model_class", "")
    native_class = model_class.startswith("nemo_automodel.") and model_class.endswith(
        ".Glm5NextForConditionalGeneration"
    )
    if metadata.get("backend_name") != "automodel" or not native_class:
        raise ValueError("Reference must use the native AutoModel GLM model, not the HF fallback")
    # This local shared-checkpoint contract deliberately rejects relocated checkpoints.
    if Path(metadata.get("checkpoint", "")).resolve() != Path(checkpoint).resolve():
        raise ValueError("Reference checkpoint path differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    contracts = [case_contract(name, batch) for name, batch in cases]
    if manifest.get("schema_version") != 1 or manifest.get("cases") != contracts:
        raise ValueError("Reference input contracts differ: check tokenizer/processor versions and case order")
    results = json.loads((directory / "results.json").read_text())["cases"]
    if len(results) != len(contracts):
        raise ValueError("Reference case count differs")
    reference = []
    for result, contract in zip(results, contracts):
        for key in ("name", "positions", "targets", "valid_tokens"):
            if result.get(key) != contract[key]:
                raise ValueError(f"Reference {key} differs for {contract['name']}")
        loss = float(result["mean_ce"])
        if not math.isfinite(loss) or contract["valid_tokens"] <= 0:
            raise ValueError(f"Invalid reference CE/count for {contract['name']}")
        logits_path = (directory / result["logits_file"]).resolve()
        if not logits_path.is_relative_to(directory):
            raise ValueError("Reference logits must be stored inside the result directory")
        logits = load_file(str(logits_path))["logits"]
        if (
            logits.ndim != 2
            or logits.shape[0] != len(contract["positions"])
            or logits.shape[1] == 0
            or not torch.isfinite(logits).all()
        ):
            raise ValueError(f"Invalid reference logits for {contract['name']}")
        reference.append((loss, logits))
    return reference


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: glm53_parity_cases.py CHECKPOINT")
    torch.save(build_glm53_parity_cases(sys.argv[1]), sys.stdout.buffer)
