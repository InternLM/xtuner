"""Internal single-GPU AutoModel reference, launched by the existing pytest fixture.

No trainer or XTuner imports. Preprocessing runs from raw cases in the pytest
interpreter, so the model environment cannot change the tokenizer/template.
"""

import argparse
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from glm53_parity_cases import build_glm53_parity_cases, case_contract
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = build_glm53_parity_cases(args.checkpoint, processor_python=args.processor_python)
    contracts = [case_contract(name, batch) for name, batch in cases]
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 1, "cases": contracts}))
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration

    backend = BackendConfig(
        attn="cudnn",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        rope_fusion=False,
        gate_precision="float32",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        force_hf=False,
        backend=backend,
        attn_implementation="sdpa",
        use_liger_kernel=False,
        use_sdpa_patching=False,
        text_config={"num_nextn_predict_layers": 0, "output_hidden_states": True},
        distributed_setup=None,
        trust_remote_code=False,
    ).eval()
    if not isinstance(model, Glm5NextForConditionalGeneration):
        raise TypeError(f"Expected native AutoModel GLM implementation, got {type(model)}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Checkpoint contains unmaterialized parameters")
    metadata = {
        "python": sys.executable,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backend_name": "automodel",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "backend": asdict(backend),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
    }
    (args.output / "metadata-rank0.json").write_text(json.dumps(metadata, indent=2, default=str))
    results = []
    with torch.no_grad():
        for (name, batch), contract in zip(cases, contracts):
            ids = batch["input_ids"][:, :-1].contiguous().cuda()
            labels = batch["labels"][:, 1:].contiguous().cuda()
            media = {
                k: batch[k].to("cuda", dtype=torch.bfloat16 if k == "pixel_values" else torch.long)
                for k in ("pixel_values", "image_grid_thw")
                if k in batch
            }
            logits = model(input_ids=ids, use_cache=False, **media).logits
            assert logits.shape[:2] == labels.shape
            valid = labels != -100
            loss = F.cross_entropy(logits[valid].float(), labels[valid])
            selected = logits[0, batch["positions"].cuda()].float().cpu().contiguous()
            if not torch.isfinite(loss) or not torch.isfinite(selected).all():
                raise ValueError(f"Nonfinite AutoModel result for {name}")
            row = {key: contract[key] for key in ("name", "positions", "targets", "valid_tokens")}
            row.update(mean_ce=loss.item(), logits_file=f"{name}.safetensors")
            save_file({"logits": selected}, str(args.output / row["logits_file"]))
            results.append(row)
            print(f"AutoModel reference: {name}, loss={loss.item()}", flush=True)
            del logits
    (args.output / "results.json").write_text(json.dumps({"cases": results}, indent=2))


if __name__ == "__main__":
    main()
