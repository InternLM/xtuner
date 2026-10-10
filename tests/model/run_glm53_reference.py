"""Run the existing raw GLM parity cases with HF or trainer-free AutoModel.

Execute this file directly, not through pytest: the AutoModel environment does
not need XTuner or its pytest conftest. Each run rebuilds inputs from raw cases.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from glm53_parity_cases import build_glm53_parity_cases, case_contract
from safetensors.torch import save_file


def source_metadata(path):
    """Record the checkout actually imported, including local modifications."""
    root = Path(path).resolve().parent

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"file": str(path), "commit": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def load_model(args, world):
    """Load a native model; distributed configuration belongs to AutoModel."""
    cp_mesh = setup = None
    if args.backend == "hf":
        import transformers

        model = transformers.Glm5NextForConditionalGeneration.from_pretrained(
            args.checkpoint, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
        ).eval()
        return model, cp_mesh, {"source": source_metadata(transformers.__file__), "attention": "eager"}

    import nemo_automodel
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration

    backend = BackendConfig(
        attn=args.attn,
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch" if args.ep_size == 1 else "hybridep",
        rope_fusion=False,
        gate_precision="float32",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )
    if world > 1:
        from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config, MoEParallelizerConfig
        from nemo_automodel.components.distributed.mesh import ParallelismSizes

        setup = DistributedSetup.build(
            strategy=FSDP2Config(),
            parallelism_sizes=ParallelismSizes(tp_size=1, pp_size=1, ep_size=args.ep_size, cp_size=args.cp_size),
            moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=False, wrap_outer_model=True),
            activation_checkpointing=False,
            world_size=world,
        )
        cp_mesh = setup.mesh_context.device_mesh["cp"]
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        force_hf=False,
        backend=backend,
        attn_implementation="sdpa",
        use_liger_kernel=False,
        use_sdpa_patching=False,
        text_config={"num_nextn_predict_layers": 0, "output_hidden_states": True},
        distributed_setup=setup,
        trust_remote_code=False,
    ).eval()
    if not isinstance(model, Glm5NextForConditionalGeneration):
        raise TypeError(f"Expected native AutoModel GLM implementation, got {type(model)}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Checkpoint contains unmaterialized parameters")
    details = {"source": source_metadata(nemo_automodel.__file__), "backend": asdict(backend)}
    if setup is not None:
        from torch.distributed.tensor import DTensor

        details["cp_ranks"] = cp_mesh.mesh.tolist()
        details["ep_ranks"] = setup.mesh_context.moe_mesh["ep"].mesh.tolist()
        details["expert_shards"] = [
            {"name": name, "global_shape": list(p.shape), "local_shape": list(p.to_local().shape)}
            for name, p in model.named_parameters()
            if isinstance(p, DTensor) and ".experts." in name
        ][:3]
    return model, cp_mesh, details


def forward_case(model, batch, cp_mesh):
    """Return local logits/next-token labels and the global start of their sequence slice."""
    ids = batch["input_ids"][:, :-1].contiguous().cuda()
    labels = batch["labels"][:, 1:].contiguous().cuda()
    media = {
        key: batch[key].to("cuda", dtype=torch.bfloat16 if key == "pixel_values" else torch.long)
        for key in ("pixel_values", "image_grid_thw")
        if key in batch
    }
    if cp_mesh is None:
        output = model(input_ids=ids, use_cache=False, **media)
        return output.logits, labels, 0
    from nemo_automodel.components.models.glm5_next.cp import shard_batch_for_glm5_next_cp

    _, local, _ = shard_batch_for_glm5_next_cp(
        cp_mesh, None, {"input_ids": ids, "labels": labels, **media}, shard_primary=False
    )
    targets = local.pop("labels")
    context = local["glm5_next_packed_context"]
    return model(**local, logits_to_keep=0, output_hidden_states=False).logits, targets, context.seq_start


def collect_result(logits, labels, positions, start, cp_mesh):
    """Reduce NLL/count and sampled full-vocabulary logits only within the CP group."""
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Local logits/labels mismatch: {logits.shape}, {labels.shape}")
    valid = labels != -100
    nll = (
        F.cross_entropy(logits[valid].float(), labels[valid], reduction="sum") if valid.any() else logits.new_zeros(())
    )
    stats = torch.stack((nll.double(), valid.sum().double()))
    selected = torch.zeros((positions.numel(), logits.shape[-1]), device=logits.device, dtype=torch.float32)
    owned = (positions >= start) & (positions < start + logits.shape[1])
    selected[owned] = logits[0, positions[owned] - start].float()
    owners = owned.to(torch.int32)
    if cp_mesh is not None and cp_mesh.size() > 1:
        dist.all_reduce(stats, group=cp_mesh.get_group())
        dist.all_reduce(selected, group=cp_mesh.get_group())
        dist.all_reduce(owners, group=cp_mesh.get_group())
    if not torch.all(owners == 1):
        raise ValueError("Each sampled position must have exactly one CP owner")
    if stats[1] <= 0 or not torch.isfinite(stats).all() or not torch.isfinite(selected).all():
        raise ValueError("Empty supervision or non-finite forward result")
    span = 0.0
    if dist.is_initialized():
        low, high = stats.clone(), stats.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        span = (high[0] - low[0]).item()
    return stats, selected.cpu().contiguous(), span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "automodel"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--processor-python", help="Run raw-case preprocessing in this Python without writing token files"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--attn", choices=("sdpa", "cudnn"), default="cudnn", help="AutoModel sparse MLA backend")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Rebuild raw cases and write their contract, without GPU/model load",
    )
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if args.ep_size < 1 or args.cp_size < 1 or world % args.ep_size or world % args.cp_size:
        parser.error("EP and CP sizes must be positive divisors of WORLD_SIZE")
    if args.backend == "hf" and (world, args.ep_size, args.cp_size) != (1, 1, 1):
        parser.error("HF is the single-process reference")
    if args.prepare_only and world != 1:
        parser.error("--prepare-only runs without torchrun")
    if args.output.exists():
        raise FileExistsError(f"Use a new result directory: {args.output}")
    cases = build_glm53_parity_cases(args.checkpoint, processor_python=args.processor_python)
    manifest = {"schema_version": 1, "cases": [case_contract(name, batch) for name, batch in cases]}
    if args.prepare_only:
        args.output.mkdir(parents=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if world > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30))
            dist.barrier()
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if world > 1:
            dist.barrier()
        model, cp_mesh, details = load_model(args, world)
        details.update(
            {
                "python": sys.executable,
                "processor_python": args.processor_python or sys.executable,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "model_class": f"{type(model).__module__}.{type(model).__name__}",
                "backend_name": args.backend,
                "world_size": world,
                "ep_size": args.ep_size,
                "cp_size": args.cp_size,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
                "loss": "external FP32 next-token CE; one explicit shift; no trainer",
            }
        )
        (args.output / f"metadata-rank{rank}.json").write_text(json.dumps(details, indent=2, default=str))
        results = []
        with torch.no_grad():
            for name, batch in cases:
                logits, labels, start = forward_case(model, batch, cp_mesh)
                positions = batch["positions"].cuda()
                stats, selected, span = collect_result(logits, labels, positions, start, cp_mesh)
                targets = batch["labels"][0, 1:][batch["positions"]].tolist()
                if int(stats[1]) != int((batch["labels"][:, 1:] != -100).sum()):
                    raise ValueError("CP changed the supervision count")
                row = {
                    "name": name,
                    "mean_ce": (stats[0] / stats[1]).item(),
                    "valid_tokens": int(stats[1]),
                    "positions": positions.tolist(),
                    "targets": targets,
                    "logits_file": f"{name}.safetensors",
                    "dp_nll_span": span,
                }
                if rank == 0:
                    save_file({"logits": selected}, str(args.output / row["logits_file"]))
                    results.append(row)
                    print(json.dumps(row), flush=True)
                del logits
        if rank == 0:
            (args.output / "results.json").write_text(json.dumps({"cases": results}, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
